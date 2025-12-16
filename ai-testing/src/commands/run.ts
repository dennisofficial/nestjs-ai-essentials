import { singleton } from 'tsyringe';
import { ArgumentsCamelCase, CommandBuilder, CommandModule, Options } from 'yargs';
import { existsSync } from 'fs-extra';
import ts from 'typescript';
import { resolve } from 'path';
import { register } from 'ts-node';
import { AITestingModule, Example, TestResult } from '../types';
import { Sema } from 'async-sema';
import { isDefined } from 'class-validator';
import { progressBar } from '../modules/progress-bar';
import { proxyConsole, restoreConsole } from '../modules/console-proxy';
import { LangfuseClient } from '@langfuse/client';
import { RunTracer } from '@fb/langfuse';

interface Args extends Options {
  file_name: string;
  playground?: boolean;
}

@singleton()
export class RunCommand<T extends object, U extends Args> implements CommandModule<T, U> {
  command: string = '$0';
  describe: string = 'Run Test On Specific File';
  builder: CommandBuilder<T, U> = {
    file_name: {
      describe: 'Name of the file to run the tests on',
      type: 'string',
      demandOption: true,
      alias: 'f',
    },
    playground: {
      describe: 'Playground function, with ENV loaded, and LangFuse Client Loaded',
      alias: 'p',
    },
  };

  handler = async (args: ArgumentsCamelCase<U>) => {
    if (args._.length) {
      console.error(`Option not allowed: ${args._}`);
      process.exit(1);
    }

    // Resolve TS-Config file
    const tsConfig = ts.findConfigFile(process.cwd(), existsSync);
    if (!tsConfig) {
      console.error('TS Config not found');
      process.exit(1);
    }

    // Execute test
    register({
      cwd: process.cwd(),
      project: tsConfig,
      require: ['tsconfig-paths/register'],
    });

    // eslint-disable-next-line @typescript-eslint/no-var-requires,@typescript-eslint/no-require-imports
    const module: AITestingModule<any, any> = require(
      resolve(process.cwd(), args.file_name),
    ).default;

    const client = new LangfuseClient({
      publicKey: process.env.LANGFUSE_PUBLIC_KEY,
      secretKey: process.env.LANGFUSE_SECRET_KEY,
      baseUrl: process.env.LANGFUSE_BASE_URL,
    });

    if (args.playground) {
      return module.playground?.(client);
    }

    await this.executeExperiment(module, client);
  };

  executeExperiment = async (
    module: AITestingModule<Record<string, any>, Record<string, any>>,
    client: LangfuseClient,
  ) => {
    // Fetch Dataset
    const datasetResult = module.datasetLoader(client);
    let dataset: Example<any, any>[];
    if (datasetResult instanceof Promise) {
      dataset = await datasetResult;
    } else {
      dataset = datasetResult;
    }

    if (!dataset.length) {
      console.error('Dataset Empty');
      process.exit(1);
    }

    // Start Progress Bar
    // Every LLM and Eval resolve will trigger an increment
    const dataLen = dataset.length;
    const evalLen = module.evaluators.length;
    progressBar.start(dataLen + dataLen * evalLen, 0, { name: module.name });
    proxyConsole();

    // Create RunTree for tracing
    const run_tree = RunTracer.traceAsync({
      name: module.name,
    });

    // Build Runnable
    const runnable = await module.runnableFactory();

    const sema = new Sema(dataset.length);
    const testResultsMatrix = await Promise.all(
      dataset.map(async ({ input, output }, idx) => {
        return await RunTracer.trace(
          async (test_run) => {
            // Execute Chain with test data Input
            await sema.acquire();
            let runResult: any;
            try {
              runResult = await runnable
                .withConfig({
                  callbacks: [RunTracer.from({ parent: test_run })],
                })
                .invoke(input);
            } catch (e: any) {
              test_run.update({
                level: 'ERROR',
                output: e,
              });
              return [];
            } finally {
              sema.release();
              progressBar.increment();
            }

            try {
              // Run evaluators
              const testResults = await RunTracer.trace(
                async (evalTree) => {
                  return await Promise.all(
                    module.evaluators.map(
                      async (fn): Promise<TestResult | TestResult[] | undefined> => {
                        await sema.acquire();
                        try {
                          const result = fn(input, runResult, output, evalTree as any);
                          let testResult;
                          if (result instanceof Promise) {
                            testResult = await result;
                          } else {
                            testResult = result;
                          }

                          // Add scores to the test run observation
                          if (Array.isArray(testResult)) {
                            for (const result of testResult) {
                              client.score.create({
                                observationId: test_run.id,
                                name: result.key,
                                value: result.grade,
                              });
                            }
                          } else {
                            client.score.create({
                              observationId: test_run.id,
                              name: testResult.key,
                              value: testResult.grade,
                            });
                          }

                          return testResult;
                        } catch (e) {
                          console.error(e);
                        } finally {
                          sema.release();
                          progressBar.increment();
                        }
                      },
                    ),
                  );
                },
                {
                  name: 'Evaluators',
                  type: 'evaluator',
                  parent: test_run,
                  input: { runResult, expected: output },
                },
              );

              // Filter out failed results
              return testResults.filter(isDefined);
            } catch (e: any) {
              return [];
            }
          },
          {
            name: `Test Case ${idx}`,
            parent: run_tree,
            input,
          },
        );
      }),
    );
    const testResults = testResultsMatrix.flat();

    // Tally up the result totals
    const averages = this.consolidateTestResults(testResults);

    // Upload Test Results to root trace
    for (const [key, score] of Object.entries(averages)) {
      client.score.create({
        observationId: run_tree.id,
        name: key,
        value: score,
      });
    }

    // End the root trace
    run_tree.end();

    progressBar.stop();
    restoreConsole();

    console.table(averages);

    try {
      // Construct Langfuse trace URL
      const baseUrl = process.env.LANGFUSE_BASE_URL || 'https://cloud.langfuse.com';
      const url = `${baseUrl}/trace/${run_tree.traceId}`;
      console.log(`View Langfuse Trace: ${url}`);
    } catch (e: any) {
      console.log(`Failed to get Langfuse Trace URL: ${e.message}`);
    }
  };

  private consolidateTestResults = (
    results: (TestResult | TestResult[])[],
  ): Record<string, number> => {
    // Tally up the result totals
    const resultTotals = results.flat().reduce(
      (acc, curr) => {
        const item = acc[curr.key];
        if (item) {
          item.count += 1;
          item.total += curr.grade;
        } else {
          acc[curr.key] = {
            count: 1,
            total: curr.grade,
          };
        }
        return acc;
      },
      {} as Record<string, { count: number; total: number }>,
    );

    // Average out results
    return Object.entries(resultTotals).reduce(
      (acc, [key, { count, total }]) => {
        acc[key] = total / count;
        return acc;
      },
      {} as Record<string, number>,
    );
  };
}
