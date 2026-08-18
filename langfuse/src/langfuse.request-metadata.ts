import { IncomingHttpHeaders } from 'http';

type TraceHeaderValue = string | string[];

const REDACTED_HEADER_VALUE = '[REDACTED]';
const MAX_PROPAGATED_METADATA_VALUE_LENGTH = 200;

const SENSITIVE_HEADER_NAMES = new Set([
  'authorization',
  'cookie',
  'set-cookie',
  'proxy-authorization',
  'x-api-key',
]);

const PROPAGATED_HEADER_TO_METADATA_KEY = {
  'x-request-id': 'requestId',
  'x-correlation-id': 'correlationId',
  traceparent: 'traceparent',
  'user-agent': 'userAgent',
  origin: 'origin',
  referer: 'referer',
} as const;

const normalizeHeaderValue = (value: IncomingHttpHeaders[string]): TraceHeaderValue | undefined => {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === 'string');
  return undefined;
};

const sanitizeHeaderValue = (
  name: string,
  value: IncomingHttpHeaders[string],
): TraceHeaderValue | undefined => {
  const normalizedValue = normalizeHeaderValue(value);
  if (normalizedValue === undefined || normalizedValue.length === 0) return undefined;

  return SENSITIVE_HEADER_NAMES.has(name.toLowerCase()) ? REDACTED_HEADER_VALUE : normalizedValue;
};

const asPropagatedString = (value: TraceHeaderValue | undefined): string | undefined => {
  if (value === undefined) return undefined;

  const stringValue = Array.isArray(value) ? value.join(', ') : value;
  return stringValue.length <= MAX_PROPAGATED_METADATA_VALUE_LENGTH ? stringValue : undefined;
};

export const sanitizeHeadersForObservation = (
  headers: IncomingHttpHeaders,
): Record<string, TraceHeaderValue> => {
  return Object.entries(headers).reduce<Record<string, TraceHeaderValue>>((result, [name, value]) => {
    const sanitizedValue = sanitizeHeaderValue(name, value);
    if (sanitizedValue !== undefined) {
      result[name] = sanitizedValue;
    }
    return result;
  }, {});
};

type RequestTraceMetadataParams = {
  headers: IncomingHttpHeaders;
  hostname?: string;
  ip?: string;
  controllerClass: string;
  controllerHandler: string;
};

export const buildObservationTraceMetadata = ({
  headers,
  hostname,
  ip,
  controllerClass,
  controllerHandler,
}: RequestTraceMetadataParams) => ({
  request: {
    ...(hostname ? { hostname } : {}),
    ...(ip ? { ip } : {}),
  },
  headers: sanitizeHeadersForObservation(headers),
  controller: {
    class: controllerClass,
    handler: controllerHandler,
  },
});

export const buildPropagatedTraceMetadata = ({
  headers,
  hostname,
  ip,
  controllerClass,
  controllerHandler,
}: RequestTraceMetadataParams): Record<string, string> => {
  const sanitizedHeaders = sanitizeHeadersForObservation(headers);
  const metadata: Record<string, string> = {
    controllerClass,
    controllerHandler,
  };

  if (hostname) {
    metadata.requestHostname = hostname;
  }

  if (ip) {
    metadata.requestIp = ip;
  }

  for (const [headerName, metadataKey] of Object.entries(PROPAGATED_HEADER_TO_METADATA_KEY)) {
    const value = asPropagatedString(sanitizedHeaders[headerName]);
    if (value !== undefined) {
      metadata[metadataKey] = value;
    }
  }

  return metadata;
};
