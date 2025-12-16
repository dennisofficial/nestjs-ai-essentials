import {
  applyDecorators,
  CallHandler,
  createParamDecorator,
  ExecutionContext,
  HttpException,
  Injectable,
  Logger,
  NestInterceptor,
  Scope,
  SetMetadata,
  UseInterceptors,
} from '@nestjs/common';
import { catchError, finalize, Observable, tap } from 'rxjs';
import { Request } from 'express';
import { Reflector } from '@nestjs/core';
import { LangfuseEventAttributes, LangfuseSpan } from '@langfuse/tracing';
import { RunTracer } from './langfuse.tracer';

const OBSERVER_KEY = Symbol('OBSERVER_KEY');
const TRACER_OPTIONS = Symbol('TRACER_OPTIONS');

interface IOptions extends LangfuseEventAttributes {
  name: string;
}

@Injectable({ scope: Scope.REQUEST })
class LangfuseInterceptor implements NestInterceptor {
  private logger = new Logger(LangfuseInterceptor.name);

  constructor(private readonly reflect: Reflector) {}

  async intercept(context: ExecutionContext, next: CallHandler): Promise<Observable<any>> {
    // Get HTTP Context
    const request = context.switchToHttp().getRequest<Request>();

    // Initialize RunTre
    const observer = this.createObserver(context, request);
    const observerProxy = this.getProxyObserver(observer);

    // Set the RunTree Proxy into `reflect` metadata, so the `GetRunTree` decorator can inject it into
    // the request handler.
    Reflect.defineMetadata(OBSERVER_KEY, observerProxy, context.getHandler());

    return next.handle().pipe(
      tap((res) => {
        observer.update({ output: res });
      }),
      catchError((err: HttpException) => {
        observer.update({ statusMessage: [err.message, err.stack ?? 'No Stacktrace'].join('\n') });
        throw err;
      }),
      finalize(async () => {
        observer.end();
      }),
    );
  }

  private createObserver(context: ExecutionContext, request: Request): LangfuseSpan {
    // Initialize config
    const { name, ...options } = this.reflect.get<IOptions>(TRACER_OPTIONS, context.getHandler());
    const observation = RunTracer.traceAsync({ name, ...options });

    const user = (request as any)._user;

    observation.updateTrace({
      metadata: {
        request: {
          hostname: request.hostname,
          ip: request.ip,
        },
        headers: request.headers,
        controller: {
          class: context.getClass().name,
          handler: context.getHandler().name,
        },
      },
      userId: user.uid,
      name: name,
      input: {
        body: request.body,
        query: request.query,
        params: request.params,
      },
      tags: [request.method, request.path],
    });

    return observation;
  }

  private getProxyObserver(observer: LangfuseSpan): LangfuseSpan {
    // Create a RunTree proxy object, to block langsmith functions.
    return new Proxy(observer, {
      get: (target: LangfuseSpan, p: keyof LangfuseSpan, receiver: any): any => {
        // Since these methods are handled from the interceptor, it will break if the user tries to
        // execute these before the interceptor does.
        const blockedMethods: (keyof LangfuseSpan)[] = ['end'];
        if (blockedMethods.includes(p)) {
          return () => this.logMethodBlock(p);
        }
        return Reflect.get(target, p, receiver);
      },
    });
  }

  private logMethodBlock(method: keyof LangfuseSpan): void {
    // Log disabled functions called from using the `RunTree` request instance.
    const stackTrace = new Error().stack?.split('\n').slice(2).join('\n');
    this.logger.warn(
      `The method "${String(
        method,
      )}" did not invoke. It is handled by the Interceptor. Method was called at:\n${stackTrace}`,
    );
  }
}

// RunTree Decorator to use the request RunTree feature
export const UseTracer = (name: string, attributes?: LangfuseEventAttributes) =>
  applyDecorators(
    SetMetadata<symbol, IOptions>(TRACER_OPTIONS, { name, ...attributes }),
    UseInterceptors(LangfuseInterceptor),
  );

// Decorator to get RunTree from request
export const GetTracer = createParamDecorator<unknown, LangfuseSpan>((_, ctx: ExecutionContext) => {
  const r = Reflect.getMetadata(OBSERVER_KEY, ctx.getHandler());
  if (!r) {
    throw new Error(
      'RunTree not found! Either remove @GetTracer() decorator, or add the `@UserTracer() decorator',
    );
  }
  return r;
});
