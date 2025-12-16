import { FactoryProvider } from '@nestjs/common/interfaces/modules/provider.interface';
import { Type } from '@nestjs/common';
import { Runnable } from '@langchain/core/runnables';

export type RunnableInput<T> = T extends Runnable<infer A> ? A : never;
export type RunnableOutput<T> = T extends Runnable<any, infer B> ? B : never;

type RunnableReturn<T> = Runnable<RunnableInput<T>, RunnableOutput<T>>;

export type ChainProvider<T> = Omit<FactoryProvider, 'provide' | 'useFactory'> & {
  provide: Type<T>;
  useFactory(...args: any[]): Promise<RunnableReturn<T>> | RunnableReturn<T>;
};
