import {
  circularTransformer,
  serializeBaseMessage,
  serializeLangfuseMedia,
  serializePrompts,
} from './langfuse.parser';

export const serializeInputsOutputs = async (data: any): Promise<any> => {
  return circularTransformer(data, [
    serializeBaseMessage,
    serializePrompts,
    serializeLangfuseMedia,
  ]);
};
