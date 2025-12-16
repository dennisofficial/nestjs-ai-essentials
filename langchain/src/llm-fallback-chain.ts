import { Runnable } from '@langchain/core/runnables';

export const llmFallback = <Input, Output>(
  first: Runnable<Input, Output>,
  ...rest: Runnable<Input, Output>[]
) =>
  first.withFallbacks({
    fallbacks: rest,
  });
