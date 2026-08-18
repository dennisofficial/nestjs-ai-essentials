# @dltech/nestjs-langfuse

## 1.1.0

### Minor Changes

- Add `RunTracer`, the NestJS request-tracing layer, and the serializer/parser/media helpers.

  A consumer had grown these independently against the same Langfuse SDK, so the two codebases
  shared a package name and almost no API. This brings that work in:

  - **`RunTracer`** — `trace` / `traceEvent` / `traceAsync` / `from`, wrapping an arbitrary async
    span across all nine observation types. Distinct from `traceSessionTurn`, which handles a
    detached session turn; both are exported. Lives in `langfuse.run-tracer.ts` because
    `langfuse.tracer.ts` is already the session-turn module.
  - **`LangfuseInterceptor` + `UseTracer` / `GetTracer`** — request-scoped tracing for NestJS
    controllers, with a Proxy guard that blocks a premature `.end()`.
  - **`registerIoSerializer`** — hook for redacting or reshaping traced I/O.
  - **`safeModelParams`**, plus internal media and prompt-value parsing helpers.

  **This adds a NestJS peer surface the package did not previously have**: `@nestjs/common`,
  `@nestjs/core`, `express`, `rxjs`, `reflect-metadata` and `class-validator` are now peers. They
  are only required by the interceptor; the tracing and callback APIs remain framework-agnostic.

  Verified against a NestJS/Mongoose consumer with 59 importing files: typecheck clean, build
  clean, 839 tests passing.

## 1.0.1

### Patch Changes

- Add `@swc/core` as a devDependency so `tsup` can honour `emitDecoratorMetadata`. Without a resolvable `@swc/core`, tsup silently degrades `emitDecoratorMetadata` to a no-op warning instead of failing — this is the same root cause fixed in `@dltech/nestjs-core`'s `LoggerInterceptor` DI bug.

  Neither package currently decorates any class within its own source (both only export decorators/utilities for consumers to apply to their own classes), so this build's emitted `design:paramtypes`/`design:type` output is unchanged — it's a defensive fix so a future `@Injectable` provider or internally-decorated class doesn't silently ship broken metadata the way `@dltech/nestjs-core` did.

## 1.0.0

### Major Changes

- First public release.

  Previously consumed as `@workspace/langchain` and `@workspace/langfuse` through a git
  submodule. Both packages now ship compiled type declarations from `dist` rather than
  pointing consumers at their TypeScript sources, and release through CI with npm
  provenance.

  `class-transformer` and `class-validator` remain peer dependencies rather than bundled
  ones: they share a metadata registry with the consuming app's own decorated DTOs, so a
  duplicated instance would silently fail validation instead of erroring.
