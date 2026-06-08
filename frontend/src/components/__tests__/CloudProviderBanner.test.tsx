import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import axios from 'axios';
import { CloudProviderBanner } from '../CloudProviderBanner';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const hiddenState = {
  should_show: false,
  last_cloud: 'aws',
  last_detection_method: 'imds',
  hint_set: false,
};

const visibleState = {
  should_show: true,
  last_cloud: 'unknown',
  last_detection_method: 'unknown',
  hint_set: false,
};

describe('CloudProviderBanner', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders nothing when should_show is false', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: hiddenState });
    const { container } = render(<CloudProviderBanner />);
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledTimes(1));
    expect(container.firstChild).toBeNull();
  });

  test('renders three buttons when should_show is true', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    render(<CloudProviderBanner />);

    await waitFor(() => {
      expect(screen.getByRole('region')).toBeInTheDocument();
    });

    expect(screen.getByText('On-premises')).toBeInTheDocument();
    expect(screen.getByText('Other / not sure')).toBeInTheDocument();
    expect(screen.getByText("Don't show this again")).toBeInTheDocument();
  });

  test('clicking On-premises sends hint=on_premises and hides banner', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    mockedAxios.post.mockResolvedValueOnce({ status: 204 });
    render(<CloudProviderBanner />);

    await waitFor(() => screen.getByText('On-premises'));
    fireEvent.click(screen.getByText('On-premises'));

    await waitFor(() =>
      expect(mockedAxios.post).toHaveBeenCalledWith(
        '/api/registry/v0.1/cloud-provider-hint',
        { hint: 'on_premises' }
      )
    );
    expect(screen.queryByRole('region')).toBeNull();
  });

  test('clicking Other sends hint=other', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    mockedAxios.post.mockResolvedValueOnce({ status: 204 });
    render(<CloudProviderBanner />);

    await waitFor(() => screen.getByText('Other / not sure'));
    fireEvent.click(screen.getByText('Other / not sure'));

    await waitFor(() =>
      expect(mockedAxios.post).toHaveBeenCalledWith(
        '/api/registry/v0.1/cloud-provider-hint',
        { hint: 'other' }
      )
    );
  });

  test('clicking decline sends hint=declined', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    mockedAxios.post.mockResolvedValueOnce({ status: 204 });
    render(<CloudProviderBanner />);

    await waitFor(() => screen.getByText("Don't show this again"));
    fireEvent.click(screen.getByText("Don't show this again"));

    await waitFor(() =>
      expect(mockedAxios.post).toHaveBeenCalledWith(
        '/api/registry/v0.1/cloud-provider-hint',
        { hint: 'declined' }
      )
    );
  });

  test('409 on POST is treated as success and banner stays hidden', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    const conflictError = Object.assign(new Error('Conflict'), {
      isAxiosError: true,
      response: { status: 409 },
    });
    mockedAxios.isAxiosError.mockReturnValue(true);
    mockedAxios.post.mockRejectedValueOnce(conflictError);
    render(<CloudProviderBanner />);

    await waitFor(() => screen.getByText('On-premises'));
    fireEvent.click(screen.getByText('On-premises'));

    await waitFor(() => expect(mockedAxios.post).toHaveBeenCalledTimes(1));
    // No re-fetch; banner stays hidden after optimistic hide
    expect(screen.queryByRole('region')).toBeNull();
  });

  test('non-409 POST failure rolls back and re-shows banner', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    const networkError = Object.assign(new Error('Network Error'), {
      isAxiosError: true,
      response: { status: 500 },
    });
    mockedAxios.isAxiosError.mockReturnValue(true);
    mockedAxios.post.mockRejectedValueOnce(networkError);
    mockedAxios.get.mockResolvedValueOnce({ data: visibleState });
    render(<CloudProviderBanner />);

    await waitFor(() => screen.getByText('On-premises'));
    fireEvent.click(screen.getByText('On-premises'));

    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledTimes(2));
    expect(screen.getByRole('region')).toBeInTheDocument();
  });
});
