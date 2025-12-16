import { createTsupConfig } from '../tsup.config.base';

export default createTsupConfig([
  {
    entry: {
      index: 'src/index.ts',
      'types/index': 'src/types/index.ts',
    },
    // Preserve shebang for CLI bin file
    banner: {
      js: '#!/usr/bin/env node',
    },
  },
]);
