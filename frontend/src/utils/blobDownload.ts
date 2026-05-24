/**
 * Trigger a browser download for a Blob with the given filename.
 *
 * Creates a temporary object URL, programmatically clicks an anchor
 * element to invoke the browser's save dialog, then revokes the URL
 * so the blob can be garbage-collected.
 */
export function triggerBlobDownload(
  blob: Blob,
  filename: string,
): void {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}
