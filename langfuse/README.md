# @dltech/nestjs-langfuse

LangChain/NestJS integration helpers for the [Langfuse](https://langfuse.com/) v5
OpenTelemetry SDK:

- **`LangfuseCallbackHandler`** — a LangChain `BaseCallbackHandler` that maps chain/LLM/tool/
  retriever/agent run events onto Langfuse OTel observations, filtering out the internal
  `Runnable*` wrapper noise LangChain emits (retries, fallbacks) so traces reflect real work.
- **`LangfusePrompt`** — wraps a Langfuse-managed prompt as a LangChain `Runnable`, with
  automatic version step-down when the caller doesn't yet supply a variable a newer prompt
  version requires.
- **`traceSessionTurn`** / **`captureParentChatTrace`** — helpers for tracing detached
  background work (e.g. an agent session turn) as its own Langfuse trace, optionally linked
  back to the chat trace that spawned it.

## Install

```bash
npm install @dltech/nestjs-langfuse
```

Peer dependencies you're expected to already have in your app: `@langchain/core`,
`@langfuse/client`, `@langfuse/core`, `@langfuse/tracing`, `@opentelemetry/api`.

## Usage

### Tracing a LangChain run

```ts
import { LangfuseCallbackHandler } from '@dltech/nestjs-langfuse';

const handler = new LangfuseCallbackHandler({ userId, sessionId, tags: ['production'] });

await chain.invoke(input, { callbacks: [handler] });
```

### A Langfuse-managed prompt as a chain step

```ts
import { LangfusePrompt } from '@dltech/nestjs-langfuse';

const prompt = new LangfusePrompt<{ topic: string }>('summarize', { label: 'production' });

const chain = prompt.getRunnable().pipe(model).pipe(new StringOutputParser());
```

### Tracing a detached session turn

```ts
import { traceSessionTurn, captureParentChatTrace } from '@dltech/nestjs-langfuse';

const parentTrace = captureParentChatTrace(); // call this while the triggering chat span is active

await traceSessionTurn(() => runAgentTurn(task), {
  name: 'session.turn:claude:execute',
  sessionId,
  input: task,
  metadata: { parentTrace },
});
```
