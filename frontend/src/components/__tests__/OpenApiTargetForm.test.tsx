/**
 * OpenApiTargetForm (issue #2022): scheme-dependent auth fields, the
 * write-only credential field, CSRF on writes, and API error placement.
 */
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';

import OpenApiTargetForm from '../OpenApiTargetForm';
import { OpenApiTarget } from '../../types/openApiServer';

jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;


function _target(
  overrides: Partial<OpenApiTarget> = {},
): OpenApiTarget {
  return {
    name: 'petstore',
    description: 'Pets',
    spec_url: 'https://petstore.example.com/openapi.json',
    base_url: 'https://petstore.example.com/api',
    spec_title: 'Petstore',
    spec_version: '1.0.0',
    auth: { scheme: 'api_key_header', header_name: 'api_key', credential_set: true },
    tool_names: ['petstore_getPetById'],
    ...overrides,
  };
}


function _renderCreate(onSaved = jest.fn()) {
  render(
    <OpenApiTargetForm serverSlug="weather-apis" intent="create" onSaved={onSaved} onCancel={jest.fn()} />,
  );
  return onSaved;
}


function _fillCreate() {
  fireEvent.change(screen.getByLabelText(/^Name/), { target: { value: 'petstore' } });
  fireEvent.change(screen.getByLabelText(/Spec URL/), {
    target: { value: 'https://petstore.example.com/openapi.json' },
  });
  fireEvent.change(screen.getByLabelText(/Base URL/), {
    target: { value: 'https://petstore.example.com/api' },
  });
}


function _rejectPost(
  status: number,
  body: Record<string, unknown>,
) {
  mockedAxios.post.mockRejectedValue({ response: { status, data: body } });
}


function _submit() {
  fireEvent.click(screen.getByRole('button', { name: 'Add API' }));
}


beforeEach(() => {
  jest.resetAllMocks();
  mockedAxios.get.mockResolvedValue({ data: { csrf_token: 'csrf-123' } });
});


describe('OpenApiTargetForm auth fields', () => {
  it('shows no auth fields for scheme none', () => {
    _renderCreate();

    expect(screen.queryByLabelText('Header name')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Query parameter')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Username')).not.toBeInTheDocument();
    expect(document.querySelector('input[type="password"]')).toBeNull();
  });

  it.each([
    ['api_key_header', 'Header name', 'API key'],
    ['api_key_query', 'Query parameter', 'API key'],
    ['basic', 'Username', 'Password'],
  ])('shows only the fields scheme %s needs', (scheme, extraLabel, credentialLabel) => {
    _renderCreate();

    fireEvent.change(screen.getByLabelText('Auth scheme'), { target: { value: scheme } });

    expect(screen.getByLabelText(extraLabel)).toBeInTheDocument();
    expect(screen.getByLabelText(credentialLabel)).toBeInTheDocument();
    const others = ['Header name', 'Query parameter', 'Username'].filter((l) => l !== extraLabel);
    for (const label of others) {
      expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
    }
  });

  it('shows only the token field for bearer', () => {
    _renderCreate();

    fireEvent.change(screen.getByLabelText('Auth scheme'), { target: { value: 'bearer' } });

    expect(screen.getByLabelText('Bearer token')).toBeInTheDocument();
    expect(screen.queryByLabelText('Header name')).not.toBeInTheDocument();
  });

  it('renders the credential as a new-password field', () => {
    _renderCreate();

    fireEvent.change(screen.getByLabelText('Auth scheme'), { target: { value: 'bearer' } });

    const field = screen.getByLabelText('Bearer token');
    expect(field).toHaveAttribute('type', 'password');
    expect(field).toHaveAttribute('autocomplete', 'new-password');
  });

  it('never pre-fills the credential on edit, even when one is stored', () => {
    render(
      <OpenApiTargetForm serverSlug="weather-apis" intent="edit" target={_target()}
        onSaved={jest.fn()} onCancel={jest.fn()} />,
    );

    const field = screen.getByLabelText('API key');
    expect(field).toHaveValue('');
    expect(field).toHaveAttribute('placeholder', 'Leave empty to keep the stored credential');
  });

  it('leaves auth out of the PATCH when the credential stays empty on edit', async () => {
    mockedAxios.patch.mockResolvedValue({ data: { name: 'petstore', sync_status: 'applied' } });
    const onSaved = jest.fn();
    render(
      <OpenApiTargetForm serverSlug="weather-apis" intent="edit" target={_target()}
        onSaved={onSaved} onCancel={jest.fn()} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(onSaved).toHaveBeenCalled());
    const body = mockedAxios.patch.mock.calls[0][1] as Record<string, unknown>;
    expect(body).not.toHaveProperty('auth');
    expect(body).not.toHaveProperty('base_url');
  });
});


describe('OpenApiTargetForm writes', () => {
  it('sends X-CSRF-Token on create', async () => {
    mockedAxios.post.mockResolvedValue({ data: { name: 'petstore', sync_status: 'applied' } });
    const onSaved = _renderCreate();

    _fillCreate();
    _submit();

    await waitFor(() => expect(onSaved).toHaveBeenCalled());
    expect(mockedAxios.post).toHaveBeenCalledWith(
      '/api/openapi-servers/weather-apis/targets',
      expect.objectContaining({ name: 'petstore', auth: { scheme: 'none' } }),
      { headers: { 'X-CSRF-Token': 'csrf-123' } },
    );
  });
});


describe('OpenApiTargetForm error placement', () => {
  it('shows url_blocked beside the base URL', async () => {
    _rejectPost(422, { error: 'url_blocked', detail: 'Private address.' });
    _renderCreate();

    _fillCreate();
    _submit();

    expect(await screen.findByTestId('openapi-target-error-base_url')).toHaveTextContent('Private address.');
    expect(screen.queryByTestId('openapi-target-error-spec_url')).not.toBeInTheDocument();
  });

  it('shows url_blocked beside the spec URL when the body names it', async () => {
    _rejectPost(422, { error: 'url_blocked', detail: 'Private address.', field: 'spec_url' });
    _renderCreate();

    _fillCreate();
    _submit();

    expect(await screen.findByTestId('openapi-target-error-spec_url')).toBeInTheDocument();
  });

  it.each(['spec_invalid', 'spec_too_large'])('shows %s beside the spec URL', async (code) => {
    _rejectPost(422, { error: code, detail: 'Bad spec.' });
    _renderCreate();

    _fillCreate();
    _submit();

    expect(await screen.findByTestId('openapi-target-error-spec_url')).toHaveTextContent('Bad spec.');
  });

  it('shows too_many_tools beside Operations and lists the tags with counts', async () => {
    _rejectPost(422, {
      error: 'too_many_tools',
      detail: '240 tools, limit 100.',
      count: 240,
      limit: 100,
      tags: { pet: 80, store: 160 },
    });
    _renderCreate();

    _fillCreate();
    _submit();

    expect(await screen.findByTestId('openapi-target-error-operations')).toHaveTextContent('240 tools');
    const tagList = screen.getByTestId('openapi-tag-list');
    expect(tagList).toHaveTextContent('pet');
    expect(tagList).toHaveTextContent('(80)');
    expect(tagList).toHaveTextContent('(160)');
    expect(screen.getByRole('radio', { name: 'Only these' })).toBeChecked();
  });

  it('shows tool_name_invalid with the names and a link that opens the overrides', async () => {
    _rejectPost(422, {
      error: 'tool_name_invalid',
      detail: 'Names too long.',
      names: ['petstore_a_very_long_generated_name'],
    });
    _renderCreate();

    _fillCreate();
    _submit();

    const block = await screen.findByTestId('openapi-tool-name-error');
    expect(block).toHaveTextContent('petstore_a_very_long_generated_name');
    expect(screen.queryByTestId('openapi-advanced-section')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open tool name overrides' }));
    expect(screen.getByTestId('openapi-advanced-section')).toBeInTheDocument();
  });

  it('shows limit_reached as a banner', async () => {
    _rejectPost(409, { error: 'limit_reached', detail: 'Full.', limit: 20 });
    _renderCreate();

    _fillCreate();
    _submit();

    const banner = await screen.findByRole('alert');
    expect(banner).toHaveTextContent('maximum of 20');
    expect(banner).toHaveTextContent('OPENAPI_MCP_MAX_TARGETS_PER_SERVER');
  });

  it('shows duplicate_name beside the name', async () => {
    _rejectPost(409, { error: 'duplicate_name', detail: 'Taken.' });
    _renderCreate();

    _fillCreate();
    _submit();

    expect(await screen.findByTestId('openapi-target-error-name')).toHaveTextContent('Taken.');
  });
});
