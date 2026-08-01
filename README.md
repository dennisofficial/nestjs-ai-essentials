# nestjs-ai-essentials

Two independently published NestJS/LangChain utility packages:

- [`langchain/`](./langchain) — [`@dltech/nestjs-langchain`](https://www.npmjs.com/package/@dltech/nestjs-langchain)
- [`langfuse/`](./langfuse) — [`@dltech/nestjs-langfuse`](https://www.npmjs.com/package/@dltech/nestjs-langfuse)

See each package's own README for install instructions and usage.

## Development

```bash
pnpm install
pnpm build      # builds both packages
pnpm typecheck  # typechecks both packages
```

Releases are managed with [changesets](https://github.com/changesets/changesets). Run
`pnpm changeset` to record an intended version bump for one or both packages, and
`pnpm release` (`changeset version`) to apply accumulated changesets to `package.json` +
`CHANGELOG.md`. Pushing a resulting version bump to `main` publishes that package to npm —
see `.github/workflows/publish.yml`.
