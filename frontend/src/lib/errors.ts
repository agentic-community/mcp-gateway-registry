import { AxiosError } from 'axios';

/**
 * Structured API error type
 */
export interface ApiError {
  message: string;
  status: number;
  code?: string;
  detail?: string;
  isNetworkError: boolean;
  isTimeout: boolean;
  originalError?: unknown;
}

/**
 * Check if an error is an AxiosError
 */
export function isAxiosError(error: unknown): error is AxiosError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'isAxiosError' in error &&
    (error as AxiosError).isAxiosError === true
  );
}

/**
 * Check if an error is an ApiError
 */
export function isApiError(error: unknown): error is ApiError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'message' in error &&
    'status' in error &&
    'isNetworkError' in error
  );
}

/**
 * Extract detail message from various error response shapes
 */
function extractDetailMessage(data: unknown): string | undefined {
  if (!data || typeof data !== 'object') return undefined;

  const record = data as Record<string, unknown>;

  // Try common error response formats
  if (typeof record.detail === 'string') {
    return record.detail;
  }

  if (typeof record.message === 'string') {
    return record.message;
  }

  if (typeof record.error === 'string') {
    return record.error;
  }

  // Handle FastAPI validation errors
  if (Array.isArray(record.detail)) {
    const messages = record.detail
      .map((item: unknown) => {
        if (typeof item === 'string') return item;
        if (typeof item === 'object' && item !== null) {
          const obj = item as Record<string, unknown>;
          return obj.msg || obj.message || JSON.stringify(obj);
        }
        return String(item);
      })
      .filter(Boolean);
    if (messages.length > 0) {
      return messages.join('; ');
    }
  }

  return undefined;
}

/**
 * Get human-readable message for HTTP status codes
 */
function getStatusMessage(status: number): string {
  const messages: Record<number, string> = {
    400: 'Bad request',
    401: 'Authentication required',
    403: 'Access denied',
    404: 'Not found',
    409: 'Conflict',
    422: 'Validation error',
    429: 'Too many requests',
    500: 'Server error',
    502: 'Bad gateway',
    503: 'Service unavailable',
    504: 'Gateway timeout',
  };

  return messages[status] || `HTTP error ${status}`;
}

/**
 * Normalize any error into an ApiError
 */
export function normalizeError(error: unknown): ApiError {
  // Already normalized
  if (isApiError(error)) {
    return error;
  }

  // Axios error
  if (isAxiosError(error)) {
    const isNetworkError = !error.response && error.code !== 'ECONNABORTED';
    const isTimeout = error.code === 'ECONNABORTED';
    const status = error.response?.status || 0;

    let message: string;
    if (isNetworkError) {
      message = 'Network error. Please check your connection.';
    } else if (isTimeout) {
      message = 'Request timed out. Please try again.';
    } else {
      const detail = extractDetailMessage(error.response?.data);
      message = detail || getStatusMessage(status);
    }

    return {
      message,
      status,
      code: error.code,
      detail: extractDetailMessage(error.response?.data),
      isNetworkError,
      isTimeout,
      originalError: error,
    };
  }

  // Standard Error
  if (error instanceof Error) {
    return {
      message: error.message || 'An unexpected error occurred',
      status: 0,
      isNetworkError: false,
      isTimeout: false,
      originalError: error,
    };
  }

  // Unknown error type
  return {
    message: typeof error === 'string' ? error : 'An unexpected error occurred',
    status: 0,
    isNetworkError: false,
    isTimeout: false,
    originalError: error,
  };
}

/**
 * Get a user-friendly error message
 */
export function getErrorMessage(error: unknown): string {
  const normalized = normalizeError(error);
  return normalized.message;
}

/**
 * Check if error is a specific HTTP status
 */
export function isHttpStatus(error: unknown, status: number): boolean {
  if (isApiError(error)) {
    return error.status === status;
  }
  if (isAxiosError(error)) {
    return error.response?.status === status;
  }
  return false;
}

/**
 * Check if error is a 401 Unauthorized
 */
export function isUnauthorized(error: unknown): boolean {
  return isHttpStatus(error, 401);
}

/**
 * Check if error is a 403 Forbidden
 */
export function isForbidden(error: unknown): boolean {
  return isHttpStatus(error, 403);
}

/**
 * Check if error is a 404 Not Found
 */
export function isNotFound(error: unknown): boolean {
  return isHttpStatus(error, 404);
}

/**
 * Check if error is a 503 Service Unavailable
 */
export function isServiceUnavailable(error: unknown): boolean {
  return isHttpStatus(error, 503);
}
