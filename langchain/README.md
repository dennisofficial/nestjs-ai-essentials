# @dltech/nestjs-langchain

NestJS-friendly utilities for building on top of [LangChain](https://js.langchain.com/):

- **`Ai*` decorators + `classToToolCall`/`ClassToolOutputParser`** — describe a tool/function
  call's arguments as a plain TypeScript class and get a JSON Schema `FunctionDefinition` plus a
  streaming output parser that validates and instantiates the result, instead of hand-writing
  JSON Schema.
- **`llmFallback`** — chain multiple `Runnable`s as ordered fallbacks.
- **`streamToRxjs`** — adapt a LangChain `IterableReadableStream` to an RxJS `Observable`.
- **`ChainProvider`** — a typed NestJS `FactoryProvider` shape for chains built from `Runnable`s.

## Install

```bash
npm install @dltech/nestjs-langchain
```

Peer dependencies you're expected to already have in your app: `@langchain/core`, `langchain`,
`@nestjs/common`, `class-transformer`, `class-validator`, `reflect-metadata`, `rxjs`, `zod`.

## Usage

### Describing a tool call as a class

```ts
import { AiToolCall, AiString, AiNumber, AiOptional, ClassToolOutputParser } from '@dltech/nestjs-langchain';

@AiToolCall({ description: 'Book a restaurant reservation' })
class BookReservationArgs {
  @AiString('Name of the restaurant')
  restaurant: string;

  @AiNumber('Party size')
  partySize: number;

  @AiString('Requested time, ISO 8601')
  @AiOptional
  time?: string;
}

const parser = new ClassToolOutputParser(BookReservationArgs);
const { tool_choice, tools } = parser.bindLLM();

const modelWithTools = model.bind({ tools, tool_choice });
const result = await modelWithTools.pipe(parser).invoke(messages);
// result is a BookReservationArgs, validated with class-validator
```

### Falling back between models

```ts
import { llmFallback } from '@dltech/nestjs-langchain';

const chain = llmFallback(primaryModel, secondaryModel, tertiaryModel);
```

### Streaming a LangChain stream into RxJS

```ts
import { streamToRxjs } from '@dltech/nestjs-langchain';

const stream = await chain.stream(input);
streamToRxjs(stream).subscribe((chunk) => sendToClient(chunk));
```
