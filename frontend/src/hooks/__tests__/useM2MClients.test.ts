import { renderHook, waitFor, act } from '@testing-library/react';
import axios from 'axios';
import {
  useM2MClients,
  registerM2MClient,
  patchM2MClient,
  deleteM2MClient,
} from '../useIAM';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const mockListResponse = {
  data: {
    total: 1,
    limit: 1000,
    skip: 0,
    items: [
      {
        client_id: 'my-pipeline',
        name: 'My Pipeline',
        description: 'CI service account',
        groups: ['pipeline-operators'],
        enabled: true,
        provider: 'manual',
        created_at: '2026-05-06T18:15:00Z',
        updated_at: '2026-05-06T18:15:00Z',
        idp_app_id: null,
        created_by: 'admin',
      },
    ],
  },
};

describe('useM2MClients', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('fetches clients on mount with limit=1000', async () => {
    mockedAxios.get.mockResolvedValueOnce(mockListResponse);

    const { result } = renderHook(() => useM2MClients());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockedAxios.get).toHaveBeenCalledWith(
      '/api/iam/m2m-clients',
      { params: { limit: 1000 } }
    );
    expect(result.current.clients).toHaveLength(1);
    expect(result.current.clients[0].client_id).toBe('my-pipeline');
    expect(result.current.error).toBeNull();
  });

  test('populates error state when fetch fails', async () => {
    mockedAxios.get.mockRejectedValueOnce({
      response: { data: { detail: 'Failed to load' } },
    });

    const { result } = renderHook(() => useM2MClients());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.clients).toEqual([]);
    expect(result.current.error).toBe('Failed to load');
  });

  test('uses default error message when detail is absent', async () => {
    mockedAxios.get.mockRejectedValueOnce({});

    const { result } = renderHook(() => useM2MClients());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.error).toBe('Failed to load M2M clients');
  });

  test('refetch re-requests the list', async () => {
    mockedAxios.get.mockResolvedValue(mockListResponse);

    const { result } = renderHook(() => useM2MClients());

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(mockedAxios.get).toHaveBeenCalledTimes(1);

    await act(async () => {
      await result.current.refetch();
    });

    expect(mockedAxios.get).toHaveBeenCalledTimes(2);
  });
});

describe('registerM2MClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('POSTs the expected payload shape', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: mockListResponse.data.items[0] });

    const payload = {
      client_id: 'my-pipeline',
      client_name: 'My Pipeline',
      groups: ['pipeline-operators'],
      description: 'CI service account',
    };
    const result = await registerM2MClient(payload);

    expect(mockedAxios.post).toHaveBeenCalledWith('/api/iam/m2m-clients', payload);
    expect(result.client_id).toBe('my-pipeline');
  });
});

describe('patchM2MClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('PATCHes with url-encoded client_id for special chars', async () => {
    mockedAxios.patch.mockResolvedValueOnce({ data: mockListResponse.data.items[0] });

    await patchM2MClient('svc:pipeline.v2', { groups: ['g1'] });

    expect(mockedAxios.patch).toHaveBeenCalledWith(
      '/api/iam/m2m-clients/svc%3Apipeline.v2',
      { groups: ['g1'] }
    );
  });
});

describe('deleteM2MClient', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('issues DELETE with url-encoded client_id', async () => {
    mockedAxios.delete.mockResolvedValueOnce({});

    await deleteM2MClient('svc:pipeline');

    expect(mockedAxios.delete).toHaveBeenCalledWith(
      '/api/iam/m2m-clients/svc%3Apipeline'
    );
  });
});
