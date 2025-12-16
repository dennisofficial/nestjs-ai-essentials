import { LangfuseClient } from '@langfuse/client';
import { Runnable } from '@langchain/core/runnables';
import { LangfuseSpan } from '@langfuse/tracing';

export interface TestResult {
  key: string;
  grade: number;
}

export type TestCase<In, Out> = (
  input: In,
  run_out: Out,
  expected: Out,
  tracer: LangfuseSpan,
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
  runnableFactory: () => Runnable<In, Out> | Promise<Runnable<In, Out>>;
  // List of evaluators
  evaluators: TestCase<In, Out>[];
  playground?: (client: LangfuseClient) => Promise<any>;
}
