import { describe, it, expect } from 'vitest';
import { AxiosError } from 'axios';
import {
  normalizeError,
  getErrorMessage,
  isAxiosError,
  isApiError,
  isUnauthorized,
  isForbidden,
  isNotFound,
  isServiceUnavailable,
  isHttpStatus,
} from './errors';

describe('Error utilities', () => {
  describe('normalizeError', () => {
    it('normalizes string errors', () => {
      const error = normalizeError('Something went wrong');

      expect(error.message).toBe('Something went wrong');
      expect(error.status).toBe(0);
      expect(error.isNetworkError).toBe(false);
      expect(error.isTimeout).toBe(false);
    });

    it('normalizes Error objects', () => {
      const error = normalizeError(new Error('Test error message'));

      expect(error.message).toBe('Test error message');
      expect(error.status).toBe(0);
      expect(error.isNetworkError).toBe(false);
    });

    it('normalizes unknown error types', () => {
      const error = normalizeError({ weird: 'object' });

      expect(error.message).toBe('An unexpected error occurred');
      expect(error.status).toBe(0);
    });

    it('normalizes axios error with detail field', () => {
      const axiosError = {
        isAxiosError: true,
        response: {
          status: 400,
          data: { detail: 'Invalid input data' },
        },
        code: undefined,
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('Invalid input data');
      expect(error.status).toBe(400);
      expect(error.detail).toBe('Invalid input data');
    });

    it('normalizes axios error with message field', () => {
      const axiosError = {
        isAxiosError: true,
        response: {
          status: 422,
          data: { message: 'Validation failed' },
        },
        code: undefined,
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('Validation failed');
      expect(error.status).toBe(422);
    });

    it('normalizes axios error with error field', () => {
      const axiosError = {
        isAxiosError: true,
        response: {
          status: 500,
          data: { error: 'Internal server error' },
        },
        code: undefined,
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('Internal server error');
      expect(error.status).toBe(500);
    });

    it('normalizes axios error with FastAPI validation errors', () => {
      const axiosError = {
        isAxiosError: true,
        response: {
          status: 422,
          data: {
            detail: [
              { loc: ['body', 'email'], msg: 'invalid email format' },
              { loc: ['body', 'password'], msg: 'too short' },
            ],
          },
        },
        code: undefined,
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('invalid email format; too short');
      expect(error.status).toBe(422);
    });

    it('handles network errors', () => {
      const axiosError = {
        isAxiosError: true,
        response: undefined,
        code: 'ERR_NETWORK',
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('Network error. Please check your connection.');
      expect(error.status).toBe(0);
      expect(error.isNetworkError).toBe(true);
    });

    it('handles timeout errors', () => {
      const axiosError = {
        isAxiosError: true,
        response: undefined,
        code: 'ECONNABORTED',
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('Request timed out. Please try again.');
      expect(error.status).toBe(0);
      expect(error.isTimeout).toBe(true);
    });

    it('returns default message for unknown status codes', () => {
      const axiosError = {
        isAxiosError: true,
        response: {
          status: 418, // I'm a teapot
          data: {},
        },
        code: undefined,
      } as unknown as AxiosError;

      const error = normalizeError(axiosError);

      expect(error.message).toBe('HTTP error 418');
      expect(error.status).toBe(418);
    });

    it('preserves already normalized errors', () => {
      const apiError = {
        message: 'Already normalized',
        status: 404,
        isNetworkError: false,
        isTimeout: false,
      };

      const error = normalizeError(apiError);

      expect(error).toBe(apiError);
    });
  });

  describe('getErrorMessage', () => {
    it('returns message from string', () => {
      expect(getErrorMessage('Test error')).toBe('Test error');
    });

    it('returns message from Error', () => {
      expect(getErrorMessage(new Error('Error message'))).toBe('Error message');
    });

    it('returns message from ApiError', () => {
      const apiError = {
        message: 'API error message',
        status: 500,
        isNetworkError: false,
        isTimeout: false,
      };
      expect(getErrorMessage(apiError)).toBe('API error message');
    });
  });

  describe('Type guards', () => {
    it('isAxiosError identifies axios errors', () => {
      expect(isAxiosError({ isAxiosError: true })).toBe(true);
      expect(isAxiosError({ isAxiosError: false })).toBe(false);
      expect(isAxiosError(new Error())).toBe(false);
      expect(isAxiosError(null)).toBe(false);
    });

    it('isApiError identifies API errors', () => {
      const apiError = {
        message: 'Test',
        status: 400,
        isNetworkError: false,
        isTimeout: false,
      };
      expect(isApiError(apiError)).toBe(true);
      expect(isApiError(new Error())).toBe(false);
      expect(isApiError(null)).toBe(false);
    });
  });

  describe('HTTP status checks', () => {
    it('isUnauthorized checks for 401', () => {
      const error = { message: 'Test', status: 401, isNetworkError: false, isTimeout: false };
      expect(isUnauthorized(error)).toBe(true);
      expect(isUnauthorized({ ...error, status: 403 })).toBe(false);
    });

    it('isForbidden checks for 403', () => {
      const error = { message: 'Test', status: 403, isNetworkError: false, isTimeout: false };
      expect(isForbidden(error)).toBe(true);
      expect(isForbidden({ ...error, status: 401 })).toBe(false);
    });

    it('isNotFound checks for 404', () => {
      const error = { message: 'Test', status: 404, isNetworkError: false, isTimeout: false };
      expect(isNotFound(error)).toBe(true);
      expect(isNotFound({ ...error, status: 500 })).toBe(false);
    });

    it('isServiceUnavailable checks for 503', () => {
      const error = { message: 'Test', status: 503, isNetworkError: false, isTimeout: false };
      expect(isServiceUnavailable(error)).toBe(true);
      expect(isServiceUnavailable({ ...error, status: 500 })).toBe(false);
    });

    it('isHttpStatus checks for arbitrary status', () => {
      const error = { message: 'Test', status: 429, isNetworkError: false, isTimeout: false };
      expect(isHttpStatus(error, 429)).toBe(true);
      expect(isHttpStatus(error, 500)).toBe(false);
    });

    it('isHttpStatus works with axios errors', () => {
      const axiosError = {
        isAxiosError: true,
        response: { status: 404 },
      };
      expect(isHttpStatus(axiosError, 404)).toBe(true);
      expect(isHttpStatus(axiosError, 500)).toBe(false);
    });
  });
});
