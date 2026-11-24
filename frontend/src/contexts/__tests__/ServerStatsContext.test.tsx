import React from 'react';
import { renderHook, waitFor, act } from '@testing-library/react';
import axios, { AxiosRequestConfig } from 'axios';
import { ServerStatsProvider, useServerStats } from '../ServerStatsContext';
import { useRegistryConfig } from '../../hooks/useRegistryConfig';

jest.mock('axios');
jest.mock('../../hooks/useRegistryConfig');

const mockedAxios = axios as jest.Mocked<typeof axios>;
const mockedUseRegistryConfig = useRegistryConfig as jest.MockedFunction<
  typeof useRegistryConfig
>;

const FEATURES = {
  mcp_servers: true,
  agents: true,
  skills: true,
  virtual_servers: true,
  federation: true,
  gateway_proxy: true,
  custom_types: true,
};

function setConfig(customTypes: { name: string; display_name: string }[]) {
  mockedUseRegistryConfig.mockReturnValue({
    config: {
      deployment_mode: 'with-gateway',
      registry_mode: 'full',
      auth_provider: 'cognito',
      nginx_updates_enabled: true,
      coding_assistants: [],
      dedup_registration_hint_enabled: false,
      features: FEATURES,
      custom_types: customTypes,
    },
    loading: false,
    error: null,
  } as ReturnType<typeof useRegistryConfig>);
}

// Route axios.get by URL so the order of the parallel fetches doesn't matter.
// fetchAllPages uses axios.get(url, { params }) — match on the path only.
function mockGetByUrl(handlers: Record<string, any>) {
  mockedAxios.get.mockImplementation((url: string) => {
    for (const [needle, response] of Object.entries(handlers)) {
      if (url === needle || url.startsWith(needle)) {
        return response instanceof Error
          ? Promise.reject(response)
          : Promise.resolve({ data: response });
      }
    }
    return Promise.resolve({ data: {} });
  });
}

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ServerStatsProvider>{children}</ServerStatsProvider>
);

type ServerListResponse = { data: { servers: Array<{ path: string }> } };

describe('ServerStatsContext custom entities', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('pages through servers when total_count exceeds one page', async () => {
    setConfig([]);
    let serverCalls = 0;
    mockedAxios.get.mockImplementation((url: string, config?: any) => {
      if (url === '/api/servers') {
        serverCalls += 1;
        const offset = config?.params?.offset ?? 0;
        if (offset === 0) {
          return Promise.resolve({
            data: {
              servers: Array.from({ length: 2000 }, (_, i) => ({
                path: `/s${i}`,
                display_name: `S${i}`,
                is_enabled: true,
                health_status: 'healthy',
              })),
              total_count: 2001,
            },
          });
        }
        return Promise.resolve({
          data: {
            servers: [
              {
                path: '/s2000',
                display_name: 'S2000',
                is_enabled: true,
                health_status: 'healthy',
              },
            ],
            total_count: 2001,
          },
        });
      }
      if (url === '/api/agents') {
        return Promise.resolve({ data: { agents: [], total_count: 0 } });
      }
      if (url === '/api/skills') {
        return Promise.resolve({ data: { skills: [], total_count: 0 } });
      }
      return Promise.resolve({ data: {} });
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(serverCalls).toBe(2);
    expect(result.current.servers).toHaveLength(2001);
    expect(result.current.stats.total).toBe(2001);
  });

  test('folds custom records into stats and exposes them by type', async () => {
    setConfig([
      { name: 'prompt_template', display_name: 'Prompt Templates' },
      { name: 'dataset', display_name: 'Datasets' },
    ]);
    mockGetByUrl({
      '/api/servers': { servers: [], total_count: 0 },
      '/api/agents': { agents: [], total_count: 0 },
      '/api/skills': { skills: [], total_count: 0 },
      '/api/custom/prompt_template': {
        records: [
          { path: '/prompt_template/a', name: 'A', is_enabled: true },
          { path: '/prompt_template/b', name: 'B', is_enabled: true },
        ],
        total_count: 2,
      },
      '/api/custom/dataset': {
        records: [{ path: '/dataset/c', name: 'C', is_enabled: false }],
        total_count: 1,
      },
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });

    await waitFor(() => expect(result.current.loading).toBe(false));

    // 3 total records: 2 enabled, 1 disabled.
    expect(result.current.stats.total).toBe(3);
    expect(result.current.stats.enabled).toBe(2);
    expect(result.current.stats.disabled).toBe(1);

    expect(result.current.customRecordsByType).toEqual([
      {
        name: 'prompt_template',
        displayName: 'Prompt Templates',
        descriptor: null,
        records: [
          { path: '/prompt_template/a', name: 'A', is_enabled: true },
          { path: '/prompt_template/b', name: 'B', is_enabled: true },
        ],
      },
      {
        name: 'dataset',
        displayName: 'Datasets',
        descriptor: null,
        records: [{ path: '/dataset/c', name: 'C', is_enabled: false }],
      },
    ]);
  });

  test('attaches matching descriptors from /api/custom-types', async () => {
    setConfig([{ name: 'prompt_template', display_name: 'Prompt Templates' }]);
    const descriptor = {
      name: 'prompt_template',
      display_name: 'Prompt Templates',
      description: null,
      fields: [],
      schema_version: 1,
      created_at: '2026-01-01T00:00:00Z',
    };
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
      '/api/custom-types': { custom_types: [descriptor] },
      '/api/custom/prompt_template': {
        records: [{ path: '/prompt_template/a', name: 'A', is_enabled: true }],
      },
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.customRecordsByType[0].descriptor).toEqual(descriptor);
  });

  test('a failing custom-type fetch is logged and treated as zero records', async () => {
    const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    setConfig([
      { name: 'good', display_name: 'Good' },
      { name: 'broken', display_name: 'Broken' },
    ]);
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
      '/api/custom/good': {
        records: [{ path: '/good/a', name: 'A', is_enabled: true }],
      },
      '/api/custom/broken': new Error('Request failed with status code 422'),
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });

    await waitFor(() => expect(result.current.loading).toBe(false));

    // The broken type yields zero records but does NOT blank the good type.
    expect(result.current.stats.total).toBe(1);
    expect(result.current.customRecordsByType).toEqual([
      {
        name: 'good',
        displayName: 'Good',
        descriptor: null,
        records: [{ path: '/good/a', name: 'A', is_enabled: true }],
      },
      { name: 'broken', displayName: 'Broken', descriptor: null, records: [] },
    ]);

    errorSpy.mockRestore();
  });

  test('does not fetch until registry config is available (avoids count flash)', async () => {
    mockedUseRegistryConfig.mockReturnValue({
      config: null,
      loading: true,
      error: null,
    } as ReturnType<typeof useRegistryConfig>);
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
    });

    renderHook(() => useServerStats(), { wrapper });

    // Wait for configuration before publishing a snapshot with custom counts.
    await waitFor(() => expect(mockedAxios.get).not.toHaveBeenCalled());
  });

  test('skips custom fetches entirely when the feature is disabled', async () => {
    mockedUseRegistryConfig.mockReturnValue({
      config: {
        deployment_mode: 'with-gateway',
        registry_mode: 'full',
        auth_provider: 'cognito',
        nginx_updates_enabled: true,
        coding_assistants: [],
        dedup_registration_hint_enabled: false,
        features: { ...FEATURES, custom_types: false },
        custom_types: [],
      },
      loading: false,
      error: null,
    } as ReturnType<typeof useRegistryConfig>);
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.customRecordsByType).toEqual([]);
    const customCalls = mockedAxios.get.mock.calls.filter(([url]) =>
      String(url).startsWith('/api/custom/'),
    );
    expect(customCalls).toHaveLength(0);
  });
});

describe('ServerStatsContext refresh-on-event', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("refetches servers/agents when a 'registry-data-changed' event fires", async () => {
    // Arrange: a loaded provider with a known number of /api/servers calls.
    setConfig([]);
    mockGetByUrl({
      '/api/servers': { servers: [], total_count: 0 },
      '/api/agents': { agents: [], total_count: 0 },
      '/api/skills': { skills: [], total_count: 0 },
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });
    await waitFor(() => expect(result.current.loading).toBe(false));

    const serversCallsAfterMount = mockedAxios.get.mock.calls.filter(
      ([url]) => url === '/api/servers',
    ).length;
    expect(serversCallsAfterMount).toBe(1);

    // Act: a sibling mutation (e.g. a skill create) only refreshes its own
    // source but dispatches this event to keep the shared stats consistent.
    await act(async () => {
      window.dispatchEvent(new Event('registry-data-changed'));
    });

    // Assert: the shared context re-fetched its lists without a page reload.
    await waitFor(() => {
      const serversCalls = mockedAxios.get.mock.calls.filter(
        ([url]) => url === '/api/servers',
      ).length;
      expect(serversCalls).toBe(serversCallsAfterMount + 1);
    });
  });

  test('does not restore deleted records from an invalidated in-flight snapshot', async () => {
    setConfig([]);
    const pending: Array<(value: ServerListResponse) => void> = [];
    mockedAxios.get.mockImplementation((url: string) => {
      if (url === '/api/servers') {
        // The frontend targets ES2020, before Promise.withResolvers.
        return new Promise<ServerListResponse>((resolve) => pending.push(resolve));
      }
      return Promise.resolve({ data: {} });
    });
    const rendered: string[][] = [];
    const { result } = renderHook(() => {
      const value = useServerStats();
      rendered.push(value.servers.map((server) => server.path));
      return value;
    }, { wrapper });
    await waitFor(() => expect(pending).toHaveLength(1));
    await act(async () => {
      pending.shift()!({ data: { servers: [{ path: '/a' }, { path: '/b' }] } });
    });
    expect(result.current.servers.map((server) => server.path)).toEqual(['/a', '/b']);

    let first: Promise<void>;
    act(() => {
      result.current.setServers((servers) => servers.filter((server) => server.path !== '/a'));
      first = result.current.refreshData();
    });
    await waitFor(() => expect(pending).toHaveLength(1));
    let finished = false;
    let second: Promise<void>;
    act(() => {
      result.current.setServers([]);
      second = result.current.refreshData().then(() => { finished = true; });
      window.dispatchEvent(new Event('registry-data-changed'));
    });
    rendered.length = 0;
    await act(async () => {
      pending.shift()!({ data: { servers: [{ path: '/b' }] } });
    });
    await waitFor(() => expect(pending).toHaveLength(1));
    expect(finished).toBe(false);
    expect(result.current.loading).toBe(true);
    expect(result.current.servers).toEqual([]);
    expect(rendered.every((paths) => !paths.includes('/b'))).toBe(true);
    await act(async () => {
      pending.shift()!({ data: { servers: [] } });
      await Promise.all([first!, second!]);
    });
    expect(finished).toBe(true);
    expect(result.current.servers).toEqual([]);
    expect(result.current.stats.total).toBe(0);
    expect(result.current.loading).toBe(false);
    expect(mockedAxios.get.mock.calls.filter(([url]) => url === '/api/servers')).toHaveLength(3);
  });

  test('ignores an old config completion and waits for the new custom type list', async () => {
    setConfig([]);
    const pending: Array<(value: ServerListResponse) => void> = [];
    mockedAxios.get.mockImplementation((url: string) => {
      if (url === '/api/servers') {
        return new Promise<ServerListResponse>((resolve) => pending.push(resolve));
      }
      if (url === '/api/custom/widgets') {
        return Promise.resolve({ data: { records: [{ name: 'Widget', path: '/widget' }] } });
      }
      return Promise.resolve({ data: {} });
    });
    const { result, rerender } = renderHook(() => useServerStats(), { wrapper });
    await waitFor(() => expect(pending).toHaveLength(1));
    setConfig([{ name: 'widgets', display_name: 'Widgets' }]);
    rerender();
    await act(async () => {
      pending.shift()!({ data: { servers: [{ path: '/stale' }] } });
    });
    await waitFor(() => expect(pending).toHaveLength(1));
    expect(result.current.servers).toEqual([]);
    expect(result.current.loading).toBe(true);
    await act(async () => {
      pending.shift()!({ data: { servers: [] } });
    });
    expect(result.current.customRecordsByType[0].records[0].name).toBe('Widget');
    expect(result.current.stats.total).toBe(1);
    expect(result.current.loading).toBe(false);
  });

  test('settles a queued refresh on unmount without waiting for its HTTP response', async () => {
    setConfig([]);
    let aborted = false;
    mockedAxios.get.mockImplementation((url: string, config?: AxiosRequestConfig) => {
      if (url === '/api/servers') {
        return new Promise<ServerListResponse>((_resolve, reject) => {
          config?.signal?.addEventListener?.('abort', () => {
            aborted = true;
            reject(new Error('request cancelled'));
          }, { once: true });
        });
      }
      return Promise.resolve({ data: {} });
    });
    const { result, unmount } = renderHook(() => useServerStats(), { wrapper });
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalled());
    let refresh: Promise<void>;
    act(() => { refresh = result.current.refreshData(); });
    unmount();
    await act(async () => { await refresh!; });
    expect(aborted).toBe(true);
    expect(mockedAxios.get.mock.calls.filter(([url]) => url === '/api/servers')).toHaveLength(1);
  });

  test('keeps an early refresh pending until config and its complete list arrive', async () => {
    mockedUseRegistryConfig.mockReturnValue({
      config: null,
      loading: true,
      error: null,
    });
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
      '/api/custom/widgets': { records: [{ name: 'Widget', path: '/widget' }] },
    });
    const { result, rerender } = renderHook(() => useServerStats(), { wrapper });
    let complete = false;
    let refresh: Promise<void>;
    await act(async () => {
      refresh = result.current.refreshData().then(() => { complete = true; });
    });
    expect(complete).toBe(false);
    expect(result.current.loading).toBe(true);
    expect(mockedAxios.get).not.toHaveBeenCalled();
    setConfig([{ name: 'widgets', display_name: 'Widgets' }]);
    await act(async () => { rerender(); });
    await act(async () => { await refresh!; });
    expect(complete).toBe(true);
    expect(result.current.stats.total).toBe(1);
    expect(result.current.customRecordsByType[0].records[0].name).toBe('Widget');
    expect(result.current.loading).toBe(false);
  });

  test('cancels all old config requests so the new list can finish independently', async () => {
    setConfig([{ name: 'widgets', display_name: 'Widgets' }]);
    const aborted: string[] = [];
    let replaced = false;
    mockedAxios.get.mockImplementation((url: string, config?: AxiosRequestConfig) => {
      if (!replaced) {
        return new Promise<ServerListResponse>((_resolve, reject) => {
          config?.signal?.addEventListener?.('abort', () => {
            aborted.push(url);
            reject(new Error('request cancelled'));
          }, { once: true });
        });
      }
      return Promise.resolve({
        data: url === '/api/servers' ? { servers: [{ path: '/current' }] } : {},
      });
    });
    const { result, rerender } = renderHook(() => useServerStats(), { wrapper });
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledTimes(5));
    let refresh: Promise<void>;
    act(() => { refresh = result.current.refreshData(); });
    replaced = true;
    setConfig([]);
    await act(async () => { rerender(); });
    await act(async () => { await refresh!; });
    expect(aborted.sort()).toEqual([
      '/api/agents', '/api/custom-types', '/api/custom/widgets', '/api/servers', '/api/skills',
    ]);
    expect(result.current.servers.map((server) => server.path)).toEqual(['/current']);
    expect(result.current.customRecordsByType).toEqual([]);
    expect(result.current.loading).toBe(false);
  });
});
