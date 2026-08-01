# @dltech/nestjs-langchain

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
