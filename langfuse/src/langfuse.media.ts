import { LangfuseMedia } from '@langfuse/core';
import { LangfuseClient } from '@langfuse/client';
import { circularTransformer } from './langfuse.parser';

export const handleMedia = async (
  media: any,
  field: 'input' | 'output' | 'metadata',
  traceId: string,
  observerId: string,
) => {
  const handleMediaClass = async () => {
    if (!(media instanceof LangfuseMedia)) return;

    void uploadMediaToLangfuse(media, field, traceId, observerId);

    return media.getTag();
  };

  return circularTransformer(media, [handleMediaClass]);
};

export const uploadMediaToLangfuse = async (
  media: LangfuseMedia,
  field: 'input' | 'output' | 'metadata',
  traceId: string,
  observerId: string,
) => {
  try {
    const client = new LangfuseClient();

    const id = await media.getId();
    const sha256Hash = await media.getSha256Hash();
    const contentBytes = media._contentBytes;
    const contentType = media._contentType ?? 'application/octet-stream';

    if (!id || !sha256Hash || !media.contentLength || !contentBytes) return;

    const { uploadUrl, mediaId } = await client.api.media.getUploadUrl({
      field,
      traceId,
      observationId: observerId,
      sha256Hash,
      contentLength: media.contentLength,
      contentType,
    });

    if (!uploadUrl) return media.getTag();

    const start = Date.now();
    let status: number;
    let uploadHttpError = '';

    try {
      // Create a proper Buffer from the Uint8Array for S3 upload
      // Using the underlying ArrayBuffer ensures exact byte length preservation
      const buffer = Buffer.from(
        contentBytes.buffer,
        contentBytes.byteOffset,
        contentBytes.byteLength,
      );

      const response = await fetch(uploadUrl, {
        method: 'PUT',
        headers: {
          'Content-Type': contentType,
          'Content-Length': buffer.byteLength.toString(),
        },
        body: buffer as any,
      });
      status = response.status;

      // Check if the response was successful
      if (!response.ok) {
        uploadHttpError = `HTTP ${status}: ${response.statusText}`;
      }
    } catch (error) {
      status = 0;
      uploadHttpError = error instanceof Error ? error.message : String(error);
    }

    await client.api.media.patch(mediaId, {
      uploadedAt: new Date().toISOString(),
      uploadHttpStatus: status,
      uploadTimeMs: Date.now() - start,
      uploadHttpError,
    });
  } catch (e) {
    console.error('Error uploading media to Langfuse', e);
  }
};
