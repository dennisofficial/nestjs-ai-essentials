import chalk from 'chalk';
import { SingleBar } from 'cli-progress';
import prettyMs from 'pretty-ms';

interface BarPayload {
  name: string;
}

const prettyTime = (seconds: number) => {
  const durationPretty = prettyMs(seconds * 1000).padStart(6);
  return chalk.green(durationPretty);
};

export const progressBar: SingleBar = new SingleBar({
  hideCursor: true,
  format: (options, params, payload: BarPayload) => {
    const name = chalk.bgBlueBright(` ${payload.name} `);

    const barWidth = options.barsize ?? 40;
    const barProgress = Math.floor(params.progress * barWidth);
    const filledBar = String().padStart(barProgress);
    const restBar = String().padEnd(barWidth - filledBar.length);
    const bar = chalk.bgWhiteBright(filledBar) + chalk.bgGray(restBar);

    const percentage = Math.round(params.progress * 100);
    const progress = chalk.green(`${percentage}%`.padStart(4));

    const duration = prettyTime(
      Math.ceil(((params.stopTime ?? Date.now()) - params.startTime) / 1000),
    );

    const valuePadded = `${params.value}`.padStart(`${params.total}`.length);
    const remaining = chalk.yellow(`${valuePadded}/${params.total}`);

    if (params.total === params.value) {
      const doneMsg = chalk.bgGreenBright(' DONE ');
      return `${name} [${bar}] ${progress} | ${duration} | ${doneMsg}`;
    }

    return `${name} [${bar}] ${progress} | ${duration} | ${remaining}`;
  },
});
