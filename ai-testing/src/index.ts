import 'reflect-metadata';
import { hideBin } from 'yargs/helpers';
import yargs from 'yargs';
import { container } from 'tsyringe';
import { config } from 'dotenv';

config({
  path: ['.env.vault', '.env', '.env.local'],
  override: true,
  DOTENV_KEY: process.env.DOTENV_KEY,
  encoding: 'utf-8',
  quiet: true,
});
process.env.NODE_ENV = 'test';

async function main() {
  const { RunCommand } = await import('./commands/run.js');

  yargs(hideBin(process.argv))
    .scriptName('ai-testing')
    .command(container.resolve(RunCommand))
    .showHelpOnFail(true)
    .help()
    .parse();
}

void main();
