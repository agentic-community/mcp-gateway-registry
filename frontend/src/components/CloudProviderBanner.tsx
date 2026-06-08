import React, { useEffect, useState } from 'react';
import axios from 'axios';

type Hint = 'aws' | 'azure' | 'gcp' | 'on_premises' | 'other' | 'declined';

interface BannerState {
  should_show: boolean;
  last_cloud: string;
  last_detection_method: string;
  hint_set: boolean;
}

export const CloudProviderBanner: React.FC = () => {
  const [shouldShow, setShouldShow] = useState(false);

  useEffect(() => {
    axios
      .get<BannerState>('/api/registry/v0.1/banner-state')
      .then((res) => setShouldShow(res.data.should_show))
      .catch((err) => console.warn('Failed to fetch banner state:', err));
  }, []);

  const submit = async (hint: Hint) => {
    setShouldShow(false);
    try {
      await axios.post('/api/registry/v0.1/cloud-provider-hint', { hint });
    } catch (err: unknown) {
      const status = axios.isAxiosError(err) ? err.response?.status : undefined;
      if (status === 409) {
        console.warn('cloud-provider-hint already set (409); treating as success');
      } else {
        console.warn('cloud-provider-hint POST failed; rolling back banner', err);
        try {
          const res = await axios.get<BannerState>('/api/registry/v0.1/banner-state');
          setShouldShow(res.data.should_show);
        } catch {
          // ignore rollback fetch failure; banner stays hidden
        }
      }
    }
  };

  if (!shouldShow) return null;

  return (
    <div
      role="region"
      aria-label="Deployment environment"
      className="bg-indigo-900/40 border border-indigo-700/50 rounded-lg p-3 my-3 mx-6"
    >
      <p className="text-sm text-indigo-200 mb-2">
        Help us improve: where is this registry deployed?
      </p>
      <div className="flex flex-wrap gap-2 items-center">
        <button
          type="button"
          onClick={() => submit('aws')}
          className="px-3 py-1 bg-gray-700 text-gray-200 text-xs font-medium rounded hover:bg-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500"
        >
          AWS
        </button>
        <button
          type="button"
          onClick={() => submit('azure')}
          className="px-3 py-1 bg-gray-700 text-gray-200 text-xs font-medium rounded hover:bg-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500"
        >
          Azure
        </button>
        <button
          type="button"
          onClick={() => submit('gcp')}
          className="px-3 py-1 bg-gray-700 text-gray-200 text-xs font-medium rounded hover:bg-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500"
        >
          GCP
        </button>
        <button
          type="button"
          onClick={() => submit('on_premises')}
          className="px-3 py-1 bg-gray-700 text-gray-200 text-xs font-medium rounded hover:bg-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500"
        >
          On-premises
        </button>
        <button
          type="button"
          onClick={() => submit('other')}
          className="px-3 py-1 bg-gray-700 text-gray-200 text-xs font-medium rounded hover:bg-gray-600 focus:outline-none focus:ring-1 focus:ring-gray-500"
        >
          Other
        </button>
        <button
          type="button"
          onClick={() => submit('declined')}
          className="text-xs text-gray-500 hover:text-gray-300 focus:outline-none ml-2"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

export default CloudProviderBanner;
