import type { EvaluatorResult, SimpleEvaluator } from 'openevals';
import type { BaseCallbackHandler } from '@langchain/core/callbacks/base';

export type { SimpleEvaluator } from 'openevals';

export interface EvalCase<In, Out = unknown> {
  input: In;
  /** Reference output passed to evaluators as `reference_outputs`. */
  expected?: Out;
}

type Invocable<In, Out> = {
  invoke(input: In, config?: { callbacks?: BaseCallbackHandler[] }): Promise<Out> | Out;
};

export interface EvalDatasetOptions<In, Out> {
  dataset: EvalCase<In, Out>[];
  /** Invoke the chain or agent under test. */
  run: (input: In) => Promise<Out>;
  /** `createLLMAsJudge`, `createTrajectoryLLMAsJudge`, or plain programmatic functions. */
  evaluators: SimpleEvaluator[];
  /**
   * Optional LangChain callbacks forwarded to each `chain.invoke` call.
   * Plug in your tracer here — all three implement `BaseCallbackHandler`:
   *
   *   LangSmith:  new LangChainTracer()            (from 'langsmith/callbacks')
   *   LangFuse:   new CallbackHandler(...)          (from '@langfuse/langchain')
   *   PostHog:    new PostHogLangChain({ client })  (from 'posthog-node/langchain')
   *
   * You can also enable LangSmith tracing without passing anything here by
   * setting LANGCHAIN_TRACING_V2=true in your environment.
   */
  callbacks?: BaseCallbackHandler[];
  /** Max concurrent cases — keep low to respect rate limits. Default: 5. */
  concurrency?: number;
}

export interface EvalDatasetResult {
  /** Average score (0..1) per metric key across all successful cases. */
  scores: Record<string, number>;
  total: number;
}

export async function evalDataset<In, Out>(
  options: EvalDatasetOptions<In, Out>,
): Promise<EvalDatasetResult> {
  const { dataset, run, evaluators, callbacks, concurrency = 5 } = options;

  void callbacks;

  const allScores: EvaluatorResult[] = [];
  const caseErrors: Array<{ index: number; error: Error }> = [];
  let cursor = 0;

  const worker = async () => {
    while (cursor < dataset.length) {
      const index = cursor++;
      const { input, expected } = dataset[index];

      try {
        const output = await run(input);

        const results = (
          await Promise.all(
            evaluators.map((evaluate) =>
              evaluate({ inputs: input, outputs: output, reference_outputs: expected }),
            ),
          )
        ).flat() as EvaluatorResult[];

        allScores.push(...results);
      } catch (e) {
        caseErrors.push({
          index,
          error: e instanceof Error ? e : new Error(String(e)),
        });
      }
    }
  };

  await Promise.all(Array.from({ length: Math.min(concurrency, dataset.length) }, worker));

  if (caseErrors.length > 0) {
    const details = caseErrors
      .sort((a, b) => a.index - b.index)
      .map(({ index, error }) => `  [${index + 1}/${dataset.length}] ${error.message}`)
      .join('\n');
    throw new Error(`${caseErrors.length} case(s) failed:\n${details}`);
  }

  const totals = new Map<string, { sum: number; count: number }>();
  for (const { key, score } of allScores) {
    const normalized = typeof score === 'boolean' ? (score ? 1 : 0) : Number(score);
    const t = totals.get(key) ?? { sum: 0, count: 0 };
    t.sum += normalized;
    t.count += 1;
    totals.set(key, t);
  }

  const scores = Object.fromEntries(
    Array.from(totals.entries()).map(([key, { sum, count }]) => [key, sum / count]),
  );

  return { scores, total: dataset.length };
}
