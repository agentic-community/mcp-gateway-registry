/**
 * OpenApiMyKeyControl and OpenApiMyKeysModal (issue #2022, per-user keys):
 * the key field is write-only, and add, replace and remove call the
 * my-credential routes with the CSRF token.
 */
import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import axios from 'axios';

import OpenApiMyKeyControl from '../OpenApiMyKeyControl';
import OpenApiMyKeysModal from '../OpenApiMyKeysModal';
import { MyCredentialStatus } from '../../types/openApiServer';

jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;

const KEY_URL = '/api/openapi-servers/weather-apis/targets/petstore/my-credential';
const CSRF = { headers: { 'X-CSRF-Token': 'csrf-123' } };


function _status(
  overrides: Partial<MyCredentialStatus> = {},
): MyCredentialStatus {
  return { target: 'petstore', configured: false, expires_at: null, ...overrides };
}


function _renderControl(
  status: MyCredentialStatus | null,
  onChanged = jest.fn(),
) {
  render(
    <OpenApiMyKeyControl slug="weather-apis" targetName="petstore" status={status}
      onChanged={onChanged} />,
  );
  return onChanged;
}


beforeEach(() => {
  jest.resetAllMocks();
  mockedAxios.get.mockResolvedValue({ data: { csrf_token: 'csrf-123' } });
});


describe('OpenApiMyKeyControl', () => {
  it('shows Add key when no key is set', () => {
    _renderControl(_status());

    expect(screen.getByTestId('openapi-my-key-status-petstore')).toHaveTextContent('Not set');
    expect(screen.getByRole('button', { name: 'Add key' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Remove key' })).not.toBeInTheDocument();
  });

  it('shows the expiry with Replace key and Remove key when a key is set', () => {
    const future = new Date(Date.now() + 86400000).toISOString();
    _renderControl(_status({ configured: true, expires_at: future }));

    expect(screen.getByTestId('openapi-my-key-status-petstore')).toHaveTextContent(/^Set, expires/);
    expect(screen.getByRole('button', { name: 'Replace key' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove key' })).toBeInTheDocument();
  });

  it('never pre-fills the key and renders it as a new-password field', () => {
    _renderControl(_status({ configured: true, expires_at: null }));

    fireEvent.click(screen.getByRole('button', { name: 'Replace key' }));

    const field = screen.getByLabelText('Your key for petstore');
    expect(field).toHaveValue('');
    expect(field).toHaveAttribute('type', 'password');
    expect(field).toHaveAttribute('autocomplete', 'new-password');
    expect(screen.getByLabelText('Valid for')).toHaveValue(30);
    expect(screen.getByLabelText('Unit')).toHaveValue('days');
  });

  it('adds a key with PUT and the CSRF token', async () => {
    mockedAxios.put.mockResolvedValue({
      data: { target: 'petstore', configured: true, expires_at: '2026-12-14T00:00:00Z' },
    });
    const onChanged = _renderControl(_status());

    fireEvent.click(screen.getByRole('button', { name: 'Add key' }));
    fireEvent.change(screen.getByLabelText('Your key for petstore'), { target: { value: 'my-secret' } });
    fireEvent.change(screen.getByLabelText('Valid for'), { target: { value: '12' } });
    fireEvent.change(screen.getByLabelText('Unit'), { target: { value: 'hours' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }));

    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(mockedAxios.put).toHaveBeenCalledWith(
      KEY_URL,
      { secret: 'my-secret', ttl_value: 12, ttl_unit: 'hours' },
      CSRF,
    );
  });

  it('replaces a key with the same PUT route', async () => {
    mockedAxios.put.mockResolvedValue({
      data: { target: 'petstore', configured: true, expires_at: null },
    });
    const onChanged = _renderControl(_status({ configured: true }));

    fireEvent.click(screen.getByRole('button', { name: 'Replace key' }));
    fireEvent.change(screen.getByLabelText('Your key for petstore'), { target: { value: 'new-secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }));

    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(mockedAxios.put).toHaveBeenCalledWith(
      KEY_URL,
      { secret: 'new-secret', ttl_value: 30, ttl_unit: 'days' },
      CSRF,
    );
  });

  it('refuses a lifetime over 30 days without calling the API', () => {
    _renderControl(_status());

    fireEvent.click(screen.getByRole('button', { name: 'Add key' }));
    fireEvent.change(screen.getByLabelText('Your key for petstore'), { target: { value: 'k' } });
    fireEvent.change(screen.getByLabelText('Valid for'), { target: { value: '31' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }));

    expect(screen.getByText(/30 days at most/)).toBeInTheDocument();
    expect(mockedAxios.put).not.toHaveBeenCalled();
  });

  it('removes a key with DELETE and the CSRF token', async () => {
    mockedAxios.delete.mockResolvedValue({ status: 204 });
    const onChanged = _renderControl(_status({ configured: true }));

    fireEvent.click(screen.getByRole('button', { name: 'Remove key' }));

    await waitFor(() => expect(onChanged).toHaveBeenCalled());
    expect(mockedAxios.delete).toHaveBeenCalledWith(KEY_URL, CSRF);
  });

  it('shows not_per_user as a sentence and does not echo the key', async () => {
    mockedAxios.put.mockRejectedValue({
      response: { status: 409, data: { error: 'not_per_user', detail: 'Target uses a shared credential' } },
    });
    _renderControl(_status());

    fireEvent.click(screen.getByRole('button', { name: 'Add key' }));
    fireEvent.change(screen.getByLabelText('Your key for petstore'), { target: { value: 'leak-check' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save key' }));

    expect(await screen.findByRole('alert')).toHaveTextContent(/uses a shared credential/);
    expect(screen.queryByDisplayValue('leak-check')).not.toBeInTheDocument();
    expect(document.body.textContent).not.toContain('leak-check');
  });
});


describe('OpenApiMyKeysModal', () => {
  function _mockList(targets: MyCredentialStatus[]) {
    mockedAxios.get.mockImplementation((url: string) => {
      if (url === '/api/openapi-servers/weather-apis/my-credentials') {
        return Promise.resolve({ data: { slug: 'weather-apis', targets } });
      }
      return Promise.resolve({ data: { csrf_token: 'csrf-123' } });
    });
  }

  async function _renderModal() {
    await act(async () => {
      render(
        <OpenApiMyKeysModal slug="weather-apis" serverName="Weather APIs" isOpen onClose={jest.fn()} />,
      );
    });
  }

  it('lists each per-user target with its own controls', async () => {
    _mockList([_status(), _status({ target: 'forecast', configured: true })]);
    await _renderModal();

    expect(screen.getByTestId('openapi-my-key-petstore')).toBeInTheDocument();
    expect(screen.getByTestId('openapi-my-key-forecast')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add key' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Replace key' })).toBeInTheDocument();
  });

  it('says so when the server has no per-user targets', async () => {
    _mockList([]);
    await _renderModal();

    expect(screen.getByText('This server has no APIs that need your own key.')).toBeInTheDocument();
  });
});
