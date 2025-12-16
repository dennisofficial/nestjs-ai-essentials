import { createTsupConfig } from '../tsup.config.base';

export default createTsupConfig([
  {
    entry: {
      index: 'src/index.ts',
      instruments: 'src/langfuse.instruments.ts',
    },
  },
]);
