import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import AuditFilterBar, { AuditFilters } from '../AuditFilterBar';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Only registry_api and mcp_access identities are enumerated; the token_mint
// stream's identity filter is deliberately a free-text search.
const filterOptions = {
  data: { usernames: ['alice@example.com', 'bob@example.com'], server_names: ['currenttime'] },
};

const renderBar = (stream: AuditFilters['stream'], onFilterChange = jest.fn()) => {
  render(
    <AuditFilterBar filters={{ stream }} onFilterChange={onFilterChange} />,
  );
  return onFilterChange;
};

describe('AuditFilterBar identity search', () => {
  beforeEach(() => {
    mockedAxios.get.mockReset();
    mockedAxios.get.mockResolvedValue(filterOptions);
  });

  it('tells the operator how the token_mint identity search works', async () => {
    renderBar('token_mint');
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledTimes(2));

    // The hint is visible without opening the dropdown.
    expect(
      screen.getByText('Free-text search: press Enter to search for the identity you typed.'),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByPlaceholderText('Search identity or paste an IdP id...'));

    // token_mint options are never fetched, so the generic empty state would
    // read as a broken control. It must explain the free-text path instead.
    expect(screen.queryByText('No options available')).not.toBeInTheDocument();
    expect(
      screen.getByText(/Token mint identities are not listed\..*press Enter to search\./),
    ).toBeInTheDocument();
  });

  it('offers the typed identifier as an explicit choice for token_mint', async () => {
    const onFilterChange = renderBar('token_mint');
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledTimes(2));

    const input = screen.getByPlaceholderText('Search identity or paste an IdP id...');
    await userEvent.type(input, '00000000-0000-4000-8000-0000000000ab');

    // Nothing matches an empty option list, but the value is still committable:
    // the dropdown must say so rather than dead-end on "No matches found".
    expect(screen.queryByText('No matches found')).not.toBeInTheDocument();
    const commit = screen.getByRole('button', {
      name: /Search for "00000000-0000-4000-8000-0000000000ab"/,
    });

    await userEvent.click(commit);
    expect(onFilterChange).toHaveBeenLastCalledWith({
      stream: 'token_mint',
      username: '00000000-0000-4000-8000-0000000000ab',
    });
  });

  it('keeps the enumerated username list for registry_api', async () => {
    renderBar('registry_api');
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledTimes(2));

    // The free-text hint is specific to token_mint.
    expect(
      screen.queryByText('Free-text search: press Enter to search for the identity you typed.'),
    ).not.toBeInTheDocument();

    await userEvent.click(screen.getByPlaceholderText('Search username...'));

    expect(screen.getByText('alice@example.com')).toBeInTheDocument();
    expect(screen.queryByText(/Token mint identities are not listed/)).not.toBeInTheDocument();
  });
});
