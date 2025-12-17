import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import {
  AdminCreateAgentModal,
  AdminRevokeAgentModal,
  AdminRevokeAllTokensModal,
  AdminRevokeApiKeyModal,
  AdminRevokeTokenModal,
} from '../AdminModals';
import type { EnforceAIAgent } from '@/api/types';

describe('AdminCreateAgentModal', () => {
  const mockOnClose = vi.fn();
  const mockOnSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  it('renders modal title and description', async () => {
    render(
      <AdminCreateAgentModal
        open={true}
        onClose={mockOnClose}
        onSubmit={mockOnSubmit}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText('Create Agent for User')).toBeInTheDocument();
    expect(screen.getByText(/Creating agent for test@example.com/)).toBeInTheDocument();
  });

  it('shows admin action warning', async () => {
    render(
      <AdminCreateAgentModal
        open={true}
        onClose={mockOnClose}
        onSubmit={mockOnSubmit}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText(/Admin Action/)).toBeInTheDocument();
    expect(screen.getByText('test|user123')).toBeInTheDocument();
  });

  it('requires at least one scope', async () => {
    render(
      <AdminCreateAgentModal
        open={true}
        onClose={mockOnClose}
        onSubmit={mockOnSubmit}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText('At least one scope is required')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create Agent' })).toBeDisabled();
  });

  it('calls onClose when cancel is clicked', async () => {
    const { user } = render(
      <AdminCreateAgentModal
        open={true}
        onClose={mockOnClose}
        onSubmit={mockOnSubmit}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(mockOnClose).toHaveBeenCalled();
  });
});

describe('AdminRevokeAgentModal', () => {
  const mockOnClose = vi.fn();
  const mockOnConfirm = vi.fn();
  const mockAgent: EnforceAIAgent = {
    user_id: 'test|user123',
    agent_id: 'test-agent-id',
    scopes: ['sqlite.manage'],
    alias: 'test-agent',
    allowed_tools: null,
    metadata: null,
    revoked_at: null,
    tokens_valid_after: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders revoke confirmation message', () => {
    render(
      <AdminRevokeAgentModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        agent={mockAgent}
        targetUserEmail="test@example.com"
      />
    );

    // Title and button have same text - use role
    expect(screen.getByRole('heading', { name: 'Revoke Agent' })).toBeInTheDocument();
    // Agent alias and id appear in multiple places
    expect(screen.getAllByText(/test-agent/).length).toBeGreaterThan(0);
    expect(screen.getByText(/test@example.com/)).toBeInTheDocument();
  });

  it('requires typing agent ID to confirm', async () => {
    render(
      <AdminRevokeAgentModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        agent={mockAgent}
        targetUserEmail="test@example.com"
      />
    );

    expect(screen.getByRole('button', { name: 'Revoke Agent' })).toBeDisabled();
    expect(screen.getByPlaceholderText('test-agent-id')).toBeInTheDocument();
  });

  it('requires reason for revocation', async () => {
    render(
      <AdminRevokeAgentModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        agent={mockAgent}
        targetUserEmail="test@example.com"
      />
    );

    expect(screen.getByText('Reason for revocation (required)')).toBeInTheDocument();
  });

  it('does not render when agent is null', () => {
    render(
      <AdminRevokeAgentModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        agent={null}
        targetUserEmail="test@example.com"
      />
    );

    expect(screen.queryByText('Revoke Agent')).not.toBeInTheDocument();
  });
});

describe('AdminRevokeAllTokensModal', () => {
  const mockOnClose = vi.fn();
  const mockOnConfirm = vi.fn();
  const mockAgent: EnforceAIAgent = {
    user_id: 'test|user123',
    agent_id: 'test-agent-id',
    scopes: ['sqlite.manage'],
    alias: 'test-agent',
    allowed_tools: null,
    metadata: null,
    revoked_at: null,
    tokens_valid_after: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders revoke all tokens confirmation', () => {
    render(
      <AdminRevokeAllTokensModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        agent={mockAgent}
        targetUserEmail="test@example.com"
      />
    );

    // Title and button have same text - use role
    expect(screen.getByRole('heading', { name: 'Revoke All Tokens' })).toBeInTheDocument();
    expect(screen.getByText(/invalidate all active tokens/)).toBeInTheDocument();
  });

  it('requires typing agent ID to confirm', async () => {
    render(
      <AdminRevokeAllTokensModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        agent={mockAgent}
        targetUserEmail="test@example.com"
      />
    );

    expect(screen.getByRole('button', { name: 'Revoke All Tokens' })).toBeDisabled();
    expect(screen.getByPlaceholderText('test-agent-id')).toBeInTheDocument();
  });
});

describe('AdminRevokeApiKeyModal', () => {
  const mockOnClose = vi.fn();
  const mockOnConfirm = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders revoke API key confirmation', () => {
    render(
      <AdminRevokeApiKeyModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        keyId="eak_test123"
        targetUserEmail="test@example.com"
      />
    );

    // Title appears as h3 and button - check for presence of title
    expect(screen.getByRole('heading', { name: 'Revoke API Key' })).toBeInTheDocument();
    // Key ID appears in multiple places (code, placeholder) - just verify it's present
    expect(screen.getAllByText(/eak_test123/).length).toBeGreaterThan(0);
  });

  it('requires typing key ID to confirm', () => {
    render(
      <AdminRevokeApiKeyModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        keyId="eak_test123"
        targetUserEmail="test@example.com"
      />
    );

    expect(screen.getByRole('button', { name: 'Revoke API Key' })).toBeDisabled();
    expect(screen.getByPlaceholderText('eak_test123')).toBeInTheDocument();
  });

  it('does not render when keyId is null', () => {
    render(
      <AdminRevokeApiKeyModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        keyId={null}
        targetUserEmail="test@example.com"
      />
    );

    expect(screen.queryByRole('heading', { name: 'Revoke API Key' })).not.toBeInTheDocument();
  });
});

describe('AdminRevokeTokenModal', () => {
  const mockOnClose = vi.fn();
  const mockOnConfirm = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders revoke token form', () => {
    render(
      <AdminRevokeTokenModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText('Revoke Gateway Token')).toBeInTheDocument();
    expect(screen.getByText('Agent ID')).toBeInTheDocument();
    expect(screen.getByText('JTI (JWT ID)')).toBeInTheDocument();
  });

  it('shows admin action warning', () => {
    render(
      <AdminRevokeTokenModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByText(/Admin Action/)).toBeInTheDocument();
    expect(screen.getByText('test|user123')).toBeInTheDocument();
  });

  it('requires agent ID and JTI', async () => {
    render(
      <AdminRevokeTokenModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    expect(screen.getByRole('button', { name: 'Revoke Token' })).toBeDisabled();
  });

  it('enables submit when agent ID and JTI are provided', async () => {
    const { user } = render(
      <AdminRevokeTokenModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    const agentIdInput = screen.getByPlaceholderText('UUID of the agent');
    const jtiInput = screen.getByPlaceholderText('Token JTI to revoke');

    await user.type(agentIdInput, 'test-agent-id');
    await user.type(jtiInput, 'test-jti-123');

    expect(screen.getByRole('button', { name: 'Revoke Token' })).not.toBeDisabled();
  });

  it('calls onConfirm with correct data', async () => {
    const { user } = render(
      <AdminRevokeTokenModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    const agentIdInput = screen.getByPlaceholderText('UUID of the agent');
    const jtiInput = screen.getByPlaceholderText('Token JTI to revoke');
    const reasonInput = screen.getByPlaceholderText('Reason for revocation...');

    await user.type(agentIdInput, 'test-agent-id');
    await user.type(jtiInput, 'test-jti-123');
    await user.type(reasonInput, 'Security concern');

    await user.click(screen.getByRole('button', { name: 'Revoke Token' }));

    await waitFor(() => {
      expect(mockOnConfirm).toHaveBeenCalledWith('test-agent-id', 'test-jti-123', 'Security concern');
    });
  });

  it('calls onClose when cancel is clicked', async () => {
    const { user } = render(
      <AdminRevokeTokenModal
        open={true}
        onClose={mockOnClose}
        onConfirm={mockOnConfirm}
        targetUserEmail="test@example.com"
        targetUserId="test|user123"
      />
    );

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(mockOnClose).toHaveBeenCalled();
  });
});
