import { circularTransformer, serializeLangfuseMedia, serializePrompts } from './langfuse.parser';

export const serializeInputsOutputs = async (data: any): Promise<any> => {
  return circularTransformer(data, [serializePrompts, serializeLangfuseMedia]);
};
