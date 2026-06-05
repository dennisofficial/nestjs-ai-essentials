import type { BasePromptValueInterface } from '@langchain/core/prompt_values';
import { ChatPromptTemplate, PromptTemplate } from '@langchain/core/prompts';
import { Runnable, RunnableLambda } from '@langchain/core/runnables';
import { ChatPromptClient, LangfuseClient, TextPromptClient } from '@langfuse/client';

type ResolvedPrompt = TextPromptClient | ChatPromptClient;
type ResolvedTemplate = PromptTemplate | ChatPromptTemplate;

/**
 * Thrown by a version attempt when the caller does not supply every `{{variable}}`
 * the prompt requires. This is the *only* error that triggers a version step-down —
 * see `LangfusePrompt.getRunnable`.
 */
export class MissingPromptVariableError extends Error {
  constructor(
    readonly promptName: string,
    readonly version: number,
    readonly missing: string[],
  ) {
    super(`Prompt "${promptName}" v${version} is missing input variable(s): ${missing.join(', ')}`);
    this.name = 'MissingPromptVariableError';
  }
}

export interface LangfusePromptOptions {
  /** Fetch a specific label (e.g. `production`, `latest`). Mirrors `prompt.get`. */
  label?: string;
  /** Pin a specific version number. When set, no step-down occurs. Mirrors `prompt.get`. */
  version?: number;
  /** Lowest version the step-down will try before giving up. Default 1. */
  minVersion?: number;
  /** Defaults to a process-wide `new LangfuseClient()`; inject one for tests/DI. */
  client?: LangfuseClient;
}

/**
 * Wraps a Langfuse-managed prompt as a drop-in step for a `RunnableSequence`.
 *
 * Declare it once next to the chain's other constants:
 *
 *   const prompt = new LangfusePrompt<PromptInput>('example_prompt', { label });
 *
 * then drop `prompt.getRunnable()` into the sequence in place of the manual
 * fetch / `getLangchainPrompt()` / `ChatPromptTemplate.fromMessages(...)` block.
 *
 * Version step-down (deploy-skew safety): the latest version is tried first. If it
 * requires a `{{variable}}` the caller does not supply — e.g. a new version was
 * published before this build started providing the variable — the attempt throws a
 * `MissingPromptVariableError`, which surfaces as an errored span in the trace, and
 * the next-older version is tried, mirroring how `llmFallback` shows a failed model
 * before its fallback. Resolution is per-invoke; Langfuse's own client cache keeps
 * the fetches cheap and lets new versions be picked up without a redeploy.
 */
export class LangfusePrompt<TInput extends Record<string, unknown> = Record<string, unknown>> {
  private readonly client: LangfuseClient;

  constructor(
    private readonly name: string,
    private readonly options: LangfusePromptOptions = {},
  ) {
    this.client = options.client ?? new LangfuseClient();
  }

  /**
   * A runnable that formats `TInput` into a `BasePromptValueInterface` for the downstream LLM.
   *
   * Trace linking: the custom `LangfuseCallbackHandler` links a prompt to its
   * generation via `metadata.langfusePrompt`, registered when this runnable starts —
   * but which version wins is only known mid-invoke. We therefore pass a thunk
   * (`() => holder.current`) that the handler evaluates later, at generation start,
   * once the winning attempt below has populated `holder`.
   */
  getRunnable(): Runnable<TInput, BasePromptValueInterface> {
    const holder: { current?: ResolvedPrompt } = {};

    return RunnableLambda.from<TInput, BasePromptValueInterface>(async (input, config) => {
      const startVersion = await this.resolveStartVersion();
      const minVersion = this.options.minVersion ?? 1;

      let lastError: unknown;
      for (let version = startVersion; version >= minVersion; version--) {
        try {
          // Each version is its own runnable, so a failed attempt is a visible,
          // errored span in the trace before the next-older version takes over.
          return await this.versionAttempt(version, holder).invoke(input, config);
        } catch (error) {
          lastError = error;
          // Only a missing variable steps down a version; anything else is a real
          // failure and is surfaced immediately.
          if (error instanceof MissingPromptVariableError && version > minVersion) continue;
          throw error;
        }
      }
      throw lastError ?? new Error(`No usable version found for prompt "${this.name}"`);
    }).withConfig({
      runName: `LangfusePrompt:${this.name}`,
      metadata: { langfusePrompt: () => holder.current },
    });
  }

  /** A runnable that formats one specific version, throwing if the input can't fill it. */
  private versionAttempt(
    version: number,
    holder: { current?: ResolvedPrompt },
  ): Runnable<TInput, BasePromptValueInterface> {
    return RunnableLambda.from<TInput, BasePromptValueInterface>(async (input, config) => {
      const prompt = await this.client.prompt.get(this.name, { version });

      const missing = this.missingVariables(prompt, input);
      if (missing.length > 0) {
        // Throw from inside the runnable so the trace shows *why* this version failed.
        throw new MissingPromptVariableError(this.name, version, missing);
      }

      const value = await this.buildTemplate(prompt).invoke(input, config);
      holder.current = prompt; // linked into the generation only on success
      return value;
    }).withConfig({ runName: `${this.name}@v${version}` });
  }

  private async resolveStartVersion(): Promise<number> {
    if (this.options.version !== undefined) return this.options.version;
    const latest = await this.client.prompt.get(
      this.name,
      this.options.label !== undefined ? { label: this.options.label } : {},
    );
    return latest.version;
  }

  private buildTemplate(prompt: ResolvedPrompt): ResolvedTemplate {
    if (prompt.type === 'chat') {
      const messages = prompt.getLangchainPrompt() as unknown as Array<{
        role: string;
        content: string;
      }>;
      return ChatPromptTemplate.fromMessages(
        messages.map((msg) => [msg.role, msg.content] as [string, string]),
      );
    }
    return PromptTemplate.fromTemplate(prompt.getLangchainPrompt() as unknown as string);
  }

  /** Variables the prompt requires (raw `{{handle}}` tokens) that `input` does not provide. */
  private missingVariables(prompt: ResolvedPrompt, input: TInput): string[] {
    // Scan the raw Langfuse `{{var}}` handles rather than LangChain's f-string
    // inference, which mis-reads literal `{ }` (e.g. JSON examples) in the body.
    const raw = typeof prompt.prompt === 'string' ? prompt.prompt : JSON.stringify(prompt.prompt);
    const required = new Set([...raw.matchAll(/\{\{\s*([\w.]+)\s*\}\}/g)].map((match) => match[1]));
    return [...required].filter((name) => !(name in input));
  }
}
