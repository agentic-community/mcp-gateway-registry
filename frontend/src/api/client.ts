import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  InternalAxiosRequestConfig,
} from 'axios';
import { v4 as uuidv4 } from 'uuid';
import { ApiError, normalizeError } from '@/lib/errors';

// CSRF token storage
let csrfToken: string | null = null;
let csrfFetchPromise: Promise<string> | null = null;

/**
 * Fetch CSRF token from the server
 * Uses a shared promise to avoid multiple concurrent fetches
 */
async function fetchCsrfToken(client: AxiosInstance): Promise<string> {
  if (csrfFetchPromise) {
    return csrfFetchPromise;
  }

  csrfFetchPromise = (async () => {
    try {
      const response = await client.get<{ csrf_token: string }>('/api/auth/csrf', {
        // Skip CSRF header for this request
        headers: { 'X-Skip-CSRF': 'true' },
      });
      csrfToken = response.data.csrf_token;
      return csrfToken;
    } finally {
      csrfFetchPromise = null;
    }
  })();

  return csrfFetchPromise;
}

/**
 * Clear the cached CSRF token (call on logout or 403 CSRF errors)
 */
export function clearCsrfToken(): void {
  csrfToken = null;
  csrfFetchPromise = null;
}

/**
 * Get the current CSRF token (for testing purposes)
 */
export function getCsrfToken(): string | null {
  return csrfToken;
}

/**
 * Create the API client with interceptors
 */
function createApiClient(): AxiosInstance {
  const client = axios.create({
    withCredentials: true, // Include cookies in requests
    timeout: 30000, // 30 second timeout
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor
  client.interceptors.request.use(
    async (config: InternalAxiosRequestConfig) => {
      // Add request ID for correlation
      config.headers['X-Request-Id'] = uuidv4();

      // Skip CSRF for GET requests or if explicitly skipped
      const skipCsrf =
        config.method?.toUpperCase() === 'GET' ||
        config.headers['X-Skip-CSRF'] === 'true';

      if (config.headers['X-Skip-CSRF']) {
        delete config.headers['X-Skip-CSRF'];
      }

      // Add CSRF token for state-changing requests
      if (!skipCsrf) {
        if (!csrfToken) {
          await fetchCsrfToken(client);
        }
        if (csrfToken) {
          config.headers['X-CSRF-Token'] = csrfToken;
        }
      }

      return config;
    },
    (error) => Promise.reject(error)
  );

  // Response interceptor
  client.interceptors.response.use(
    (response) => response,
    async (error: AxiosError) => {
      const originalRequest = error.config as AxiosRequestConfig & {
        _csrfRetry?: boolean;
      };

      // Handle CSRF token errors - retry once with fresh token
      if (
        error.response?.status === 403 &&
        !originalRequest._csrfRetry &&
        originalRequest.method?.toUpperCase() !== 'GET'
      ) {
        const responseData = error.response?.data as { detail?: string } | undefined;
        const errorMessage = responseData?.detail?.toLowerCase() || '';

        if (errorMessage.includes('csrf')) {
          originalRequest._csrfRetry = true;
          clearCsrfToken();
          await fetchCsrfToken(client);

          if (csrfToken && originalRequest.headers) {
            originalRequest.headers['X-CSRF-Token'] = csrfToken;
          }

          return client(originalRequest);
        }
      }

      // Normalize the error for consistent handling
      const normalizedError = normalizeError(error);
      return Promise.reject(normalizedError);
    }
  );

  return client;
}

// Export the singleton API client
export const apiClient = createApiClient();

/**
 * Type-safe API request helper
 */
export async function apiRequest<T>(
  config: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.request<T>(config);
  return response.data;
}

/**
 * GET request helper
 */
export async function apiGet<T>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> {
  return apiRequest<T>({ ...config, method: 'GET', url });
}

/**
 * POST request helper
 */
export async function apiPost<T, D = unknown>(
  url: string,
  data?: D,
  config?: AxiosRequestConfig
): Promise<T> {
  return apiRequest<T>({ ...config, method: 'POST', url, data });
}

/**
 * PUT request helper
 */
export async function apiPut<T, D = unknown>(
  url: string,
  data?: D,
  config?: AxiosRequestConfig
): Promise<T> {
  return apiRequest<T>({ ...config, method: 'PUT', url, data });
}

/**
 * PATCH request helper
 */
export async function apiPatch<T, D = unknown>(
  url: string,
  data?: D,
  config?: AxiosRequestConfig
): Promise<T> {
  return apiRequest<T>({ ...config, method: 'PATCH', url, data });
}

/**
 * DELETE request helper
 */
export async function apiDelete<T>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> {
  return apiRequest<T>({ ...config, method: 'DELETE', url });
}

// Re-export error types
export type { ApiError };
