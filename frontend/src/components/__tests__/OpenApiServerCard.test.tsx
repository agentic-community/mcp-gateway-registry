/**
 * OpenApiServerCard (issue #2022): the Connect button opens the standard
 * connection dialog, and the bare MCP URL is no longer printed on the card.
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';

import OpenApiServerCard from '../OpenApiServerCard';
import { OpenApiServer } from '../../types/openApiServer';

jest.mock('../ServerConfigModal', () => ({
  __esModule: true,
  default: (props: { isOpen: boolean; server: { name: string; path: string }; resourceType: string }) =>
    props.isOpen ? (
      <div data-testid="config-modal">
        {props.server.name} {props.server.path} {props.resourceType}
      </div>
    ) : null,
}));


function _server(): OpenApiServer {
  return {
    slug: 'weather-and-space',
    name: 'Weather and Space',
    description: 'Forecasts and NASA data',
    path: '/weather-and-space',
    mcp_url: 'https://gw.example.com/weather-and-space/mcp',
    sync_status: 'applied',
    targets: [],
  } as OpenApiServer;
}


function _render() {
  render(
    <OpenApiServerCard server={_server()} onManage={jest.fn()} onEdit={jest.fn()} onDelete={jest.fn()} />,
  );
}


describe('OpenApiServerCard', () => {
  it('does not print the MCP URL on the card', () => {
    _render();

    expect(screen.queryByText('https://gw.example.com/weather-and-space/mcp')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /copy/i })).not.toBeInTheDocument();
  });

  it('opens the connection dialog for this server from Connect', () => {
    _render();
    expect(screen.queryByTestId('config-modal')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Connect to Weather and Space' }));

    expect(screen.getByTestId('config-modal')).toHaveTextContent('Weather and Space /weather-and-space server');
  });
});
