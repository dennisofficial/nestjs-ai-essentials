import { IterableReadableStream } from '@langchain/core/utils/stream';
import { Observable } from 'rxjs';

export const streamToRxjs = <T>(stream: IterableReadableStream<T>) => {
  return new Observable<T>((subscriber) => {
    (async () => {
      try {
        for await (const chunk of stream) {
          subscriber.next(chunk);
        }
        subscriber.complete();
      } catch (error) {
        subscriber.error(error);
      }
    })();
  });
};
