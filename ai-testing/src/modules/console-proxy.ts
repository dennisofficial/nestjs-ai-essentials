const consoleOg = console;

interface ConsoleExecutionLog {
  method: keyof typeof console;
  arguments: any[];
}
const logs: ConsoleExecutionLog[] = [];

export const proxyConsole = () => {
  console = new Proxy(console, {
    get: (target: typeof console, prop: keyof typeof console) => {
      const func = target[prop];
      if (typeof func === 'function') {
        if (prop === 'error') {
          return target[prop];
        }

        return (...args: any[]) => {
          logs.push({ method: prop, arguments: args });
        };
      }
      return target[prop];
    },
  });
};

export const restoreConsole = () => {
  console = consoleOg;
  logs.forEach((log) => {
    const func = console[log.method];
    if (typeof func.apply === 'function') {
      (func as (...args: any[]) => void).apply(console, log.arguments);
    }
  });
};
