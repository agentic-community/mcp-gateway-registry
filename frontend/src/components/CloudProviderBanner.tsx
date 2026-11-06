import React, { useEffect, useState } from 'react';
import axios from 'axios';

type Hint = 'aws' | 'azure' | 'gcp' | 'on_premises' | 'other' | 'declined';

interface BannerState {
  should_show: boolean;
  last_cloud: string;
  last_detection_method: string;
  hint_set: boolean;
}

const PROVIDER_CHOICES: { value: Hint; label: string }[] = [
  { value: 'aws', label: 'AWS' },
  { value: 'azure', label: 'Azure' },
  { value: 'gcp', label: 'GCP' },
  { value: 'on_premises', label: 'On-premises' },
  { value: 'other', label: 'Other' },
];

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
      className="mx-4 mt-4 rounded-xl border border-indigo-200 bg-indigo-50 p-4 sm:mx-6 lg:mx-8 dark:border-indigo-500/30 dark:bg-indigo-500/10"
    >
      <p className="mb-1 text-sm font-medium text-indigo-900 dark:text-indigo-100">
        Help us improve: where is this registry deployed?
      </p>
      <p className="mb-3 text-xs text-indigo-700 dark:text-indigo-300">
        Automatic detection could not identify this environment. Your answer is saved for this
        registry. To set it at deploy time instead, use <code className="font-mono">MCP_CLOUD_PROVIDER</code> (Docker or
        ECS) or <code className="font-mono">app.mcpCloudProvider</code> (Helm).
      </p>
      <div className="flex flex-wrap items-center gap-2">
        {PROVIDER_CHOICES.map(({ value, label }) => (
          <button
            key={value}
            type="button"
            onClick={() => submit(value)}
            className="btn-secondary px-3 py-1 text-xs"
          >
            {label}
          </button>
        ))}
        <button
          type="button"
          onClick={() => submit('declined')}
          className="ml-1 rounded px-2 py-1 text-xs text-indigo-700 hover:text-indigo-900 dark:text-indigo-300 dark:hover:text-indigo-100"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

export default CloudProviderBanner;
