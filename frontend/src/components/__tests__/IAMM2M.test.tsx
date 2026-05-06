import React from 'react';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import IAMM2M from '../IAMM2M';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

Object.assign(navigator, {
  clipboard: { writeText: jest.fn().mockResolvedValue(undefined) },
});

const mockGroups = [
  { name: 'pipeline-operators' },
  { name: 'registry-readonly' },
];

const manualClient = {
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
};

const oktaClient = {
  client_id: 'okta-synced',
  name: 'Okta Synced',
  description: null,
  groups: ['registry-readonly'],
  enabled: true,
  provider: 'okta',
  created_at: '2026-05-01T10:00:00Z',
  updated_at: '2026-05-01T10:00:00Z',
  idp_app_id: 'okta-app-123',
  created_by: null,
};

const keycloakClient = {
  client_id: 'kc-legacy',
  name: 'Keycloak Legacy',
  description: null,
  groups: [],
  enabled: true,
  provider: 'keycloak',
  created_at: '2026-04-01T10:00:00Z',
  updated_at: '2026-04-01T10:00:00Z',
  idp_app_id: null,
  created_by: null,
};

const mockListResponse = {
  data: {
    total: 3,
    limit: 1000,
    skip: 0,
    items: [manualClient, oktaClient, keycloakClient],
  },
};

const mockGroupsResponse = { data: { groups: mockGroups } };

function setupDefaultMocks() {
  mockedAxios.get.mockImplementation((url: string) => {
    if (url === '/api/iam/m2m-clients') {
      return Promise.resolve(mockListResponse);
    }
    if (url === '/api/management/iam/groups') {
      return Promise.resolve(mockGroupsResponse);
    }
    return Promise.reject(new Error(`Unexpected GET ${url}`));
  });
}

const toastSpy = jest.fn();

describe('IAMM2M list view', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setupDefaultMocks();
  });

  test('renders a row per client with provider badge', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);

    await waitFor(() => {
      expect(screen.getByText('My Pipeline')).toBeInTheDocument();
    });

    expect(screen.getByText('Okta Synced')).toBeInTheDocument();
    expect(screen.getByText('Keycloak Legacy')).toBeInTheDocument();
    expect(screen.getByText('manual')).toBeInTheDocument();
    expect(screen.getByText('okta')).toBeInTheDocument();
    expect(screen.getByText('keycloak')).toBeInTheDocument();
  });

  test('shows created_by for manual records and em-dash otherwise', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);

    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    const manualRow = screen.getByText('My Pipeline').closest('tr')!;
    expect(within(manualRow).getByText('admin')).toBeInTheDocument();

    const oktaRow = screen.getByText('Okta Synced').closest('tr')!;
    expect(within(oktaRow).getAllByText('—').length).toBeGreaterThanOrEqual(1);
  });

  test('disables edit and delete buttons on non-manual rows', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);

    await waitFor(() => expect(screen.getByText('Okta Synced')).toBeInTheDocument());

    const oktaRow = screen.getByText('Okta Synced').closest('tr')!;
    const editBtn = within(oktaRow).getByTitle(/Managed by IdP sync; cannot edit here/);
    const deleteBtn = within(oktaRow).getByTitle(/Managed by IdP sync; cannot delete here/);
    expect(editBtn).toBeDisabled();
    expect(deleteBtn).toBeDisabled();
  });

  test('enables edit and delete buttons on manual rows', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);

    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    const manualRow = screen.getByText('My Pipeline').closest('tr')!;
    const editBtn = within(manualRow).getByTitle('Edit groups');
    const deleteBtn = within(manualRow).getByTitle('Delete account');
    expect(editBtn).not.toBeDisabled();
    expect(deleteBtn).not.toBeDisabled();
  });

  test('search filters by name or client_id', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);

    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    const input = screen.getByPlaceholderText('Search M2M accounts...');
    await userEvent.type(input, 'okta');

    expect(screen.queryByText('My Pipeline')).not.toBeInTheDocument();
    expect(screen.getByText('Okta Synced')).toBeInTheDocument();

    await userEvent.clear(input);
    await userEvent.type(input, 'kc-legacy');
    expect(screen.getByText('Keycloak Legacy')).toBeInTheDocument();
    expect(screen.queryByText('Okta Synced')).not.toBeInTheDocument();
  });

  test('help popover toggles on icon click', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);

    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    expect(screen.queryByText(/records a/)).not.toBeInTheDocument();

    const helpBtn = screen.getByRole('button', { name: 'Help' });
    await userEvent.click(helpBtn);
    expect(screen.getByText(/records a/)).toBeInTheDocument();

    await userEvent.click(helpBtn);
    expect(screen.queryByText(/records a/)).not.toBeInTheDocument();
  });
});

describe('IAMM2M register flow', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setupDefaultMocks();
  });

  test('validates invalid client_id before submitting', async () => {
    render(<IAMM2M onShowToast={toastSpy} />);
    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    await userEvent.click(screen.getByRole('button', { name: /Register existing client/i }));
    expect(
      screen.getByText(/Register Existing Client/i)
    ).toBeInTheDocument();

    await userEvent.type(screen.getByPlaceholderText('e.g. my-pipeline-client'), 'bad id!');
    await userEvent.type(screen.getByPlaceholderText('Human-readable name'), 'Some Name');
    await userEvent.click(screen.getByRole('button', { name: /^Register$/ }));

    expect(screen.getByText(/Allowed characters/)).toBeInTheDocument();
    expect(mockedAxios.post).not.toHaveBeenCalled();
  });

  test('submits valid registration via POST /api/iam/m2m-clients', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: manualClient });

    render(<IAMM2M onShowToast={toastSpy} />);
    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    await userEvent.click(screen.getByRole('button', { name: /Register existing client/i }));
    await userEvent.type(screen.getByPlaceholderText('e.g. my-pipeline-client'), 'new-svc');
    await userEvent.type(screen.getByPlaceholderText('Human-readable name'), 'New Service');

    const groupCheckbox = screen.getByLabelText('pipeline-operators');
    await userEvent.click(groupCheckbox);

    await userEvent.click(screen.getByRole('button', { name: /^Register$/ }));

    await waitFor(() => {
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/iam/m2m-clients', {
        client_id: 'new-svc',
        client_name: 'New Service',
        description: undefined,
        groups: ['pipeline-operators'],
      });
    });
    expect(toastSpy).toHaveBeenCalledWith(
      'Registered M2M client "New Service"',
      'success'
    );
  });

  test('surfaces 409 conflict detail as error toast', async () => {
    mockedAxios.post.mockRejectedValueOnce({
      response: { data: { detail: 'M2M client new-svc already exists' } },
    });

    render(<IAMM2M onShowToast={toastSpy} />);
    await waitFor(() => expect(screen.getByText('My Pipeline')).toBeInTheDocument());

    await userEvent.click(screen.getByRole('button', { name: /Register existing client/i }));
    await userEvent.type(screen.getByPlaceholderText('e.g. my-pipeline-client'), 'new-svc');
    await userEvent.type(screen.getByPlaceholderText('Human-readable name'), 'New Service');
    await userEvent.click(screen.getByLabelText('pipeline-operators'));
    await userEvent.click(screen.getByRole('button', { name: /^Register$/ }));

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        'M2M client new-svc already exists',
        'error'
      );
    });
  });
});
