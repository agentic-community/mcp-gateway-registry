import React from 'react';
import { renderHook, waitFor } from '@testing-library/react';
import axios from 'axios';
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
function mockGetByUrl(handlers: Record<string, any>) {
  mockedAxios.get.mockImplementation((url: string) => {
    for (const [needle, response] of Object.entries(handlers)) {
      if (url.startsWith(needle)) {
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

describe('ServerStatsContext custom entities', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('requests each custom type with limit=1000 (the endpoint max)', async () => {
    setConfig([{ name: 'prompt_template', display_name: 'Prompt Templates' }]);
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
      '/api/custom/prompt_template': { records: [] },
    });

    const { result } = renderHook(() => useServerStats(), { wrapper });

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(mockedAxios.get).toHaveBeenCalledWith('/api/custom/prompt_template?limit=1000');
  });

  test('folds custom records into stats and exposes them by type', async () => {
    setConfig([
      { name: 'prompt_template', display_name: 'Prompt Templates' },
      { name: 'dataset', display_name: 'Datasets' },
    ]);
    mockGetByUrl({
      '/api/servers': { servers: [] },
      '/api/agents': { agents: [] },
      '/api/skills': { skills: [] },
      '/api/custom/prompt_template': {
        records: [
          { path: '/prompt_template/a', name: 'A', is_enabled: true },
          { path: '/prompt_template/b', name: 'B', is_enabled: true },
        ],
      },
      '/api/custom/dataset': {
        records: [{ path: '/dataset/c', name: 'C', is_enabled: false }],
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
    expect(errorSpy).toHaveBeenCalledWith(
      'Failed to fetch custom records for "broken":',
      expect.any(Error),
    );

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

    // With no config yet, fetchData bails before issuing any request, so the
    // first stats update will include custom counts rather than core-only.
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
