import { defineConfig, Options } from 'tsup';
import { readFileSync } from 'fs';
import { join } from 'path';

export function createTsupConfig(options: Options[]): ReturnType<typeof defineConfig> {
  // Read package.json to auto-detect externals
  const packageJsonPath = join(process.cwd(), 'package.json');
  const packageJson = JSON.parse(readFileSync(packageJsonPath, 'utf-8'));

  const external = Array.from(
    new Set([
      ...Object.keys(packageJson.peerDependencies || {}),
      ...Object.keys(packageJson.devDependencies || {}).filter(
        (dep) =>
          !dep.startsWith('@types/') && !['typescript', 'tsup', 'jest', 'ts-jest'].includes(dep),
      ),
    ]),
  );

  return defineConfig(
    options.map((o) => ({
      format: ['cjs', 'esm'],
      dts: false, // Types point to source files
      splitting: true, // Enable code splitting to preserve module dependencies
      sourcemap: true, // Enable source maps for better stack traces
      clean: true,
      outDir: 'dist',
      external,
      treeshake: true,
      minify: false,
      target: 'es2022',
      ...o,
    })),
  );
}
