/**
 * "Manage APIs" on the ordinary server card (issue #2022): shown only to an
 * admin, only with the feature on, and only on an OpenAPI MCP server.
 */
import React from 'react';
import { render, screen } from '@testing-library/react';

import ServerCard, { Server } from '../ServerCard';

let mockIsAdmin = true;

jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
});
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));
jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { username: 'u', is_admin: mockIsAdmin, can_modify_servers: true, ui_permissions: {} },
  }),
}));
jest.mock('axios');
jest.mock('../ServerConfigModal', () => () => null);
jest.mock('../SecurityScanModal', () => () => null);
jest.mock('../VersionSelectorModal', () => () => null);
jest.mock('../ServerDetailsModal', () => () => null);
jest.mock('../PullCardPreviewModal', () => () => null);
jest.mock('../ResourceBoundTokenButton', () => () => null);
jest.mock('../OpenApiServerDetailsModal', () => () => null);
jest.mock('../StarRatingWidget', () => () => <div data-testid="stars" />);

const noop = () => {};


function _server(
  overrides: Partial<Server> = {},
): Server {
  return {
    name: 'Weather APIs',
    path: '/weather-apis',
    enabled: true,
    openapi_server_slug: 'weather-apis',
    ...overrides,
  };
}


beforeEach(() => {
  mockIsAdmin = true;
});


describe('ServerCard Manage APIs link', () => {
  it('shows Manage APIs to an admin when the feature is on', () => {
    render(<ServerCard server={_server()} onToggle={noop} openApiEnabled />);

    expect(screen.getByRole('button', { name: /Manage APIs/ })).toBeInTheDocument();
  });

  it('hides Manage APIs when the feature is off', () => {
    render(<ServerCard server={_server()} onToggle={noop} openApiEnabled={false} />);

    expect(screen.queryByRole('button', { name: /Manage APIs/ })).not.toBeInTheDocument();
  });

  it('hides Manage APIs from a non-admin', () => {
    mockIsAdmin = false;
    render(<ServerCard server={_server()} onToggle={noop} openApiEnabled />);

    expect(screen.queryByRole('button', { name: /Manage APIs/ })).not.toBeInTheDocument();
  });

  it('hides Manage APIs on an ordinary server', () => {
    render(
      <ServerCard server={_server({ openapi_server_slug: undefined })} onToggle={noop} openApiEnabled />,
    );

    expect(screen.queryByRole('button', { name: /Manage APIs/ })).not.toBeInTheDocument();
  });
});
