/**
 * OpenApiServerForm (issue #2022): path preview, edit-mode name lock, and
 * where API errors appear.
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

import OpenApiServerForm from '../OpenApiServerForm';
import { OpenApiServer } from '../../types/openApiServer';


function _apiError(
  status: number,
  body: Record<string, unknown>,
) {
  return { response: { status, data: body } };
}


function _server(): OpenApiServer {
  return {
    slug: 'weather-apis',
    name: 'Weather APIs',
    description: 'Forecasts',
    path: '/weather-apis',
    tags: ['openapi', 'weather'],
    visibility: 'private',
    targets: [],
  };
}


describe('OpenApiServerForm', () => {
  it('previews the path as the admin types the name', () => {
    render(<OpenApiServerForm onSave={jest.fn()} onCancel={jest.fn()} />);

    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'Weather APIs' } });

    expect(screen.getByTestId('openapi-path-preview')).toHaveTextContent('/weather-apis');
  });

  it('defaults visibility to private', () => {
    render(<OpenApiServerForm onSave={jest.fn()} onCancel={jest.fn()} />);

    expect(screen.getByLabelText('Visibility')).toHaveValue('private');
  });

  it('disables the name on edit and leaves the registry openapi tag out of the input', () => {
    render(<OpenApiServerForm server={_server()} onSave={jest.fn()} onCancel={jest.fn()} />);

    expect(screen.getByLabelText(/Name/)).toBeDisabled();
    expect(screen.getByLabelText('Tags')).toHaveValue('weather');
  });

  it('sends the name and fields on create', async () => {
    const onSave = jest.fn().mockResolvedValue(undefined);
    render(<OpenApiServerForm onSave={onSave} onCancel={jest.fn()} />);

    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'Weather APIs' } });
    fireEvent.change(screen.getByLabelText('Tags'), { target: { value: 'weather, forecast' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create Server' }));

    await waitFor(() => expect(onSave).toHaveBeenCalledWith({
      name: 'Weather APIs',
      description: '',
      tags: ['weather', 'forecast'],
      visibility: 'private',
      allowed_groups: [],
    }));
  });

  it('shows duplicate_name beside the name field', async () => {
    const onSave = jest.fn().mockRejectedValue(
      _apiError(409, { error: 'duplicate_name', detail: 'A server already uses /weather-apis.' }),
    );
    render(<OpenApiServerForm onSave={onSave} onCancel={jest.fn()} />);

    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'Weather APIs' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create Server' }));

    expect(await screen.findByTestId('openapi-name-error')).toHaveTextContent(
      'A server already uses /weather-apis.',
    );
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows limit_reached as a banner that names the limit', async () => {
    const onSave = jest.fn().mockRejectedValue(
      _apiError(409, { detail: { error: 'limit_reached', detail: 'Limit reached.', limit: 10 } }),
    );
    render(<OpenApiServerForm onSave={onSave} onCancel={jest.fn()} />);

    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'Weather APIs' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create Server' }));

    const banner = await screen.findByRole('alert');
    expect(banner).toHaveTextContent('maximum of 10');
    expect(banner).toHaveTextContent('OPENAPI_MCP_MAX_SERVERS');
  });
});
