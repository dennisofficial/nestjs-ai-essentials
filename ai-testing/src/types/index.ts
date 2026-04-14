import { LangfuseClient } from '@langfuse/client';

export interface TestResult {
  key: string;
  grade: number;
}

// Intentionally erased to avoid leaking a concrete LangChain callback type across package boundaries.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AITestingTracer = any;

export interface AITestingRunnable<In, Out> {
  invoke(input: In): Promise<Out> | Out;
  withConfig(config: Record<string, any>): AITestingRunnable<In, Out>;
}

export type TestCase<In, Out> = (
  input: In,
  run_out: Out,
  expected: Out,
  tracer: AITestingTracer,
) => Promise<TestResult | TestResult[]> | TestResult | TestResult[];
export interface Example<In, Out> {
  input: In;
  output: Out;
}

export interface AITestingModule<In, Out> {
  // The name of the Testing Module
  name: string;
  // Local DataSet
  datasetLoader: (client: LangfuseClient) => Promise<Example<In, Out>[]> | Example<In, Out>[];
  // A function that returns a langchain runnable
  runnableFactory: () => AITestingRunnable<In, Out> | Promise<AITestingRunnable<In, Out>>;
  // List of evaluators
  evaluators: TestCase<In, Out>[];
  playground?: (client: LangfuseClient) => Promise<any>;
}
