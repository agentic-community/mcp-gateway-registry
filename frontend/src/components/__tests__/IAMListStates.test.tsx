import React from 'react';
import { render, screen } from '@testing-library/react';
import IAMUsers from '../IAMUsers';
import IAMGroups from '../IAMGroups';
import IAMUserGroups from '../IAMUserGroups';

/**
 * Smoke tests for the IAM list views' loading / empty / populated states.
 * These components had no test coverage; this net guards the ListStateBoundary
 * adoption (a broken JSX fragment would compile but fail to render).
 */

// Mutable hook return values, reset per test.
const usersState = { users: [] as any[], isLoading: false, error: null as string | null, refetch: jest.fn() };
const groupsState = { groups: [] as any[], isLoading: false, error: null as string | null, refetch: jest.fn() };
const userGroupsState = {
  data: { total: 0, limit: 25, skip: 0, items: [] as any[] },
  items: [] as any[],
  total: 0,
  isLoading: false,
  error: null as string | null,
  refetch: jest.fn(),
};

jest.mock('../../hooks/useIAM', () => ({
  useIAMUsers: () => usersState,
  useIAMGroups: () => groupsState,
  useUserGroups: () => userGroupsState,
  createHumanUser: jest.fn(),
  deleteUser: jest.fn(),
  updateUserGroups: jest.fn(),
  createUserGroup: jest.fn(),
  updateUserGroup: jest.fn(),
  deleteUserGroup: jest.fn(),
  getUserGroup: jest.fn(),
  createPingFederateUser: jest.fn(),
}));
jest.mock('../../hooks/useRegistryConfig', () => ({
  useRegistryConfig: () => ({ config: { auth_provider: 'keycloak' } }),
}));
// The list views read the signed-in user for admin-only actions.
jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ user: null }),
}));

const toast = jest.fn();

beforeEach(() => {
  usersState.users = [];
  usersState.isLoading = false;
  usersState.error = null;
  groupsState.groups = [];
  groupsState.isLoading = false;
  groupsState.error = null;
  userGroupsState.items = [];
  userGroupsState.total = 0;
  userGroupsState.isLoading = false;
  userGroupsState.error = null;
});

describe('IAMUsers list states', () => {
  it('shows the empty message when there are no users', () => {
    render(<IAMUsers onShowToast={toast} />);
    expect(screen.getByText(/No users yet/)).toBeInTheDocument();
  });

  it('renders a user row when populated', () => {
    usersState.users = [{ username: 'alice', email: 'a@x.com', groups: [] }];
    render(<IAMUsers onShowToast={toast} />);
    expect(screen.getByText('alice')).toBeInTheDocument();
  });

  it('shows the error state', () => {
    usersState.error = 'boom';
    render(<IAMUsers onShowToast={toast} />);
    expect(screen.getByText('boom')).toBeInTheDocument();
  });
});

describe('IAMGroups list states', () => {
  it('shows the empty message when there are no groups', () => {
    render(<IAMGroups onShowToast={toast} />);
    expect(screen.getByText(/No groups yet/)).toBeInTheDocument();
  });

  it('renders a group row when populated', () => {
    groupsState.groups = [{ name: 'admins', description: 'Admin group' }];
    render(<IAMGroups onShowToast={toast} />);
    expect(screen.getByText('admins')).toBeInTheDocument();
  });
});

describe('IAMUserGroups list states', () => {
  it('shows the empty message when there are no mappings', () => {
    render(<IAMUserGroups onShowToast={toast} />);
    expect(screen.getByText(/No user-to-group mappings yet/)).toBeInTheDocument();
  });

  it('renders a mapping row when populated', () => {
    userGroupsState.items = [
      {
        username: 'bob',
        groups: ['team-a'],
        enabled: true,
        provider: 'manual',
        created_at: '',
        updated_at: '',
      },
    ];
    userGroupsState.total = 1;
    render(<IAMUserGroups onShowToast={toast} />);
    expect(screen.getByText('bob')).toBeInTheDocument();
  });
});
