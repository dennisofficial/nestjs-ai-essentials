import {
  circularTransformer,
  CircularTransformerFn,
  serializeLangfuseMedia,
  serializePrompts,
} from './langfuse.parser';

const _GLOBAL_HANDLERS = new Set<CircularTransformerFn>();

/**
 * Serializes Langfuse IO values by recursively walking the full value graph.
 *
 * Custom serializers are evaluated for every traversed value:
 * - the root value
 * - nested object values
 * - array items
 *
 * They do not run on object keys.
 *
 * If a serializer returns a defined value, that branch is considered handled
 * and recursion stops for that branch.
 */
export const serializeInputsOutputs = async (data: any): Promise<any> => {
  return circularTransformer(data, [
    serializePrompts,
    serializeLangfuseMedia,
    ...Array.from(_GLOBAL_HANDLERS),
  ]);
};

/**
 * Registers a global IO serializer for this process.
 *
 * The serializer is called for every traversed value during input/output
 * serialization, so it should stay fast and predictable.
 */
export const registerIoSerializer = (fn: CircularTransformerFn) => {
  _GLOBAL_HANDLERS.add(fn);
};
