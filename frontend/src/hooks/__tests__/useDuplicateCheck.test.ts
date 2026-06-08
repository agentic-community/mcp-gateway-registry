import { renderHook, act, waitFor } from '@testing-library/react';
import axios from 'axios';
import { useDuplicateCheck } from '../useDuplicateCheck';
import type { DuplicateCheckResult } from '../../types/duplicateCheck';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

let mockHintEnabled = true;
jest.mock('../useRegistryConfig', () => ({
  useRegistryConfig: () => ({
    config: { dedup_registration_hint_enabled: mockHintEnabled },
    loading: false,
    error: null,
  }),
}));

const emptyResult: DuplicateCheckResult = {
  collision_with: [],
  advisory_matches: [],
  threshold: 0.8,
  similarity_search_available: true,
  has_collision: false,
};

const collisionResult: DuplicateCheckResult = {
  collision_with: [
    {
      entity_type: 'mcp_server',
      path: '/servers/foo',
      name: 'Foo',
      owner: 'alice',
      registered_at: '2026-05-01T00:00:00Z',
      relevance_score: null,
      match_reason: 'URL match',
    },
  ],
  advisory_matches: [],
  threshold: 0.8,
  similarity_search_available: true,
  has_collision: true,
};

const advisoryResult: DuplicateCheckResult = {
  collision_with: [],
  advisory_matches: [
    {
      entity_type: 'mcp_server',
      path: '/servers/bar',
      name: 'Bar',
      owner: 'bob',
      registered_at: '2026-05-01T00:00:00Z',
      relevance_score: 0.92,
      match_reason: 'Similar name',
    },
  ],
  threshold: 0.8,
  similarity_search_available: true,
  has_collision: false,
};

(axios as unknown as { isCancel: (e: unknown) => boolean }).isCancel = (
  e: unknown,
) => (e as { __CANCEL__?: boolean })?.__CANCEL__ === true;

describe('useDuplicateCheck', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockHintEnabled = true;
  });

  test('short-circuits to proceed when hint flag is off', async () => {
    mockHintEnabled = false;
    const { result } = renderHook(() => useDuplicateCheck());

    let outcome;
    await act(async () => {
      outcome = await result.current.runCheck({
        entityType: 'mcp_server',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          proxy_pass_url: 'http://x/y',
        },
      });
    });

    expect(outcome).toEqual({ kind: 'proceed' });
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  test('proceeds when no collisions or advisory matches', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: emptyResult });
    const { result } = renderHook(() => useDuplicateCheck());

    let outcome;
    await act(async () => {
      outcome = await result.current.runCheck({
        entityType: 'mcp_server',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          proxy_pass_url: 'http://x/y',
        },
      });
    });

    expect(outcome).toEqual({ kind: 'proceed' });
    expect(mockedAxios.post).toHaveBeenCalledWith(
      '/api/servers/check-duplicates',
      expect.objectContaining({ proxy_pass_url: 'http://x/y' }),
      expect.objectContaining({ signal: expect.any(Object) }),
    );
    expect(result.current.showModal).toBe(false);
  });

  test('shows modal on collision_with', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: collisionResult });
    const { result } = renderHook(() => useDuplicateCheck());

    let outcome;
    await act(async () => {
      outcome = await result.current.runCheck({
        entityType: 'mcp_server',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          proxy_pass_url: 'http://x/y',
        },
      });
    });

    expect(outcome).toEqual({ kind: 'show-modal' });
    expect(result.current.showModal).toBe(true);
    expect(result.current.collisionWith).toHaveLength(1);
    expect(result.current.advisoryMatches).toHaveLength(0);
  });

  test('shows modal on advisory_matches alone', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: advisoryResult });
    const { result } = renderHook(() => useDuplicateCheck());

    let outcome;
    await act(async () => {
      outcome = await result.current.runCheck({
        entityType: 'mcp_server',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          proxy_pass_url: 'http://x/y',
        },
      });
    });

    expect(outcome).toEqual({ kind: 'show-modal' });
    expect(result.current.showModal).toBe(true);
    expect(result.current.advisoryMatches).toHaveLength(1);
  });

  test('proceeds with notice when similarity backend unavailable and no hits', async () => {
    mockedAxios.post.mockResolvedValueOnce({
      data: { ...emptyResult, similarity_search_available: false },
    });
    const { result } = renderHook(() => useDuplicateCheck());

    let outcome;
    await act(async () => {
      outcome = await result.current.runCheck({
        entityType: 'a2a_agent',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          url: 'http://x/y',
        },
      });
    });

    expect(outcome).toMatchObject({ kind: 'proceed', notice: expect.any(String) });
  });

  test('proceeds (does not block) when network error occurs', async () => {
    mockedAxios.post.mockRejectedValueOnce(new Error('boom'));
    const { result } = renderHook(() => useDuplicateCheck());

    let outcome;
    await act(async () => {
      outcome = await result.current.runCheck({
        entityType: 'skill',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          skill_md_url: 'https://github.com/x/y',
        },
      });
    });

    expect(outcome).toEqual({ kind: 'proceed' });
  });

  test('routes to correct endpoint per entity type', async () => {
    mockedAxios.post.mockResolvedValue({ data: emptyResult });
    const { result } = renderHook(() => useDuplicateCheck());

    await act(async () => {
      await result.current.runCheck({
        entityType: 'a2a_agent',
        payload: {
          name: 'A',
          description: null,
          self_path: null,
          url: 'http://x',
        },
      });
    });
    expect(mockedAxios.post).toHaveBeenLastCalledWith(
      '/api/agents/check-duplicates',
      expect.anything(),
      expect.anything(),
    );

    await act(async () => {
      await result.current.runCheck({
        entityType: 'skill',
        payload: {
          name: 'S',
          description: null,
          self_path: null,
          skill_md_url: 'https://github.com/a/b',
        },
      });
    });
    expect(mockedAxios.post).toHaveBeenLastCalledWith(
      '/api/skills/check-duplicates',
      expect.anything(),
      expect.anything(),
    );
  });

  test('closeModal clears state', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: collisionResult });
    const { result } = renderHook(() => useDuplicateCheck());

    await act(async () => {
      await result.current.runCheck({
        entityType: 'mcp_server',
        payload: {
          name: 'Foo',
          description: null,
          self_path: null,
          proxy_pass_url: 'http://x/y',
        },
      });
    });

    expect(result.current.showModal).toBe(true);

    act(() => {
      result.current.closeModal();
    });

    await waitFor(() => {
      expect(result.current.showModal).toBe(false);
    });
    expect(result.current.collisionWith).toHaveLength(0);
    expect(result.current.advisoryMatches).toHaveLength(0);
  });
});
