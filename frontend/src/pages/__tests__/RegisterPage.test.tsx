import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import RegisterPage from '../RegisterPage';

/**
 * Render-level safety net for RegisterPage, written before adopting the shared
 * form-field primitives. Pins the tab/mode structure and key fields so the
 * refactor can't silently drop a field or break the server/agent switch.
 */

jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: {
      username: 'admin',
      is_admin: true,
      can_modify_servers: true,
      ui_permissions: {
        register_service: ['all'],
        publish_agent: ['all'],
      },
    },
  }),
}));
jest.mock('../../hooks/useDuplicateCheck', () => ({
  useDuplicateCheck: () => ({
    runCheck: jest.fn(),
    collisionWith: [],
    advisoryMatches: [],
    showModal: false,
    closeModal: jest.fn(),
    reset: jest.fn(),
  }),
}));
jest.mock('../../components/DuplicateCheckModal', () => {
  const M = () => null;
  M.displayName = 'DuplicateCheckModal';
  return M;
});

// Discovery Identity is gated on the egress-auth feature + a gateway, exactly as
// in the edit modal, so both signals are controllable per test.
const mockIsEgressAuthEnabled = jest.fn<Promise<boolean>, []>();
let mockDeploymentMode: 'with-gateway' | 'registry-only' = 'with-gateway';
jest.mock('../../utils/egressAuth', () => ({
  isEgressAuthEnabled: () => mockIsEgressAuthEnabled(),
}));
jest.mock('../../hooks/useRegistryConfig', () => ({
  useRegistryConfig: () => ({
    config: { deployment_mode: mockDeploymentMode },
    loading: false,
    error: null,
  }),
}));

beforeEach(() => {
  mockDeploymentMode = 'with-gateway';
  mockIsEgressAuthEnabled.mockReset();
  mockIsEgressAuthEnabled.mockResolvedValue(true);
});

async function renderPage() {
  const result = render(
    <MemoryRouter>
      <RegisterPage />
    </MemoryRouter>,
  );
  // The egress-feature probe resolves after mount; flush it so the gated
  // discovery controls have settled before any assertion.
  await waitFor(() => expect(mockIsEgressAuthEnabled).toHaveBeenCalled());
  return result;
}

describe('RegisterPage', () => {
  it('defaults to the server registration form with its core fields', async () => {
    await renderPage();
    expect(screen.getByText('Server Name *')).toBeInTheDocument();
    expect(screen.getByText('Path *')).toBeInTheDocument();
    expect(screen.getByText('Deployment Type *')).toBeInTheDocument();
  });

  it('switches to the agent form when the Agent tab is selected', async () => {
    await renderPage();
    // Click the Agent registration-type tab.
    fireEvent.click(screen.getByText('A2A Agent'));
    expect(screen.getByText('Agent Name *')).toBeInTheDocument();
  });

  it('auto-generates the server path from the name', async () => {
    await renderPage();
    const nameInput = screen.getByPlaceholderText(/My Custom Server|server name/i);
    fireEvent.change(nameInput, { target: { value: 'My Cool Server' } });
    // Path is auto-generated (slugified) when left untouched.
    expect(screen.getByDisplayValue(/my-cool-server/)).toBeInTheDocument();
  });

  it('offers Quick Form and JSON Upload registration modes', async () => {
    await renderPage();
    expect(screen.getByText('Quick Form')).toBeInTheDocument();
    expect(screen.getByText('JSON Upload')).toBeInTheDocument();
  });

  it('offers OAuth 2.0 but no OAuth 2.1 scheme (discovery is not a scheme)', async () => {
    await renderPage();
    expect(screen.getByText('OAuth 2.0 (client credentials)')).toBeInTheDocument();
    expect(screen.queryByText(/OAuth 2\.1 \(delegated/)).not.toBeInTheDocument();
  });

  it('reveals the OAuth 2.1 discovery config when the discovery toggle is checked', async () => {
    await renderPage();
    const toggle = screen.getByRole('checkbox', {
      name: /Discovery Identity \(OAuth 2\.1\)/,
    });
    fireEvent.click(toggle);
    expect(
      screen.getByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).toBeInTheDocument();
  });

  it('offers no discovery identity when the egress feature is disabled', async () => {
    mockIsEgressAuthEnabled.mockResolvedValue(false);
    await renderPage();
    expect(
      screen.queryByRole('checkbox', { name: /Discovery Identity \(OAuth 2\.1\)/ }),
    ).not.toBeInTheDocument();
  });

  it('offers no discovery identity in registry-only mode', async () => {
    mockDeploymentMode = 'registry-only';
    await renderPage();
    expect(
      screen.queryByRole('checkbox', { name: /Discovery Identity \(OAuth 2\.1\)/ }),
    ).not.toBeInTheDocument();
  });
});
