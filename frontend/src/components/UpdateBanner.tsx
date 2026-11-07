import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { isSafeUrl } from '../utils/safeUrl';

interface UpdateCheckResponse {
  current: string;
  latest: string | null;
  update_available: boolean;
  release_notes_url: string | null;
  checked_at: string | null;
  check_enabled: boolean;
}

const dismissedKey = (latest: string) => `mcp-update-banner-dismissed:${latest}`;

export const UpdateBanner: React.FC = () => {
  const [data, setData] = useState<UpdateCheckResponse | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    axios
      .get<UpdateCheckResponse>('/api/system/update-check')
      .then((res) => {
        setData(res.data);
        if (res.data.latest && localStorage.getItem(dismissedKey(res.data.latest))) {
          setDismissed(true);
        }
      })
      .catch((err) => console.warn('Failed to fetch update-check:', err));
  }, []);

  if (!data || !data.update_available || !data.latest || dismissed) return null;

  const handleDismiss = () => {
    if (data.latest) {
      localStorage.setItem(dismissedKey(data.latest), '1');
    }
    setDismissed(true);
  };

  const showLink = data.release_notes_url && isSafeUrl(data.release_notes_url);

  return (
    <div
      role="region"
      aria-label="Update available"
      className="mx-4 mt-4 rounded-xl border border-success-200 bg-success-50 p-3 sm:mx-6 lg:mx-8 dark:border-success-500/30 dark:bg-success-500/10"
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-success-900 dark:text-success-100">
          A newer registry release is available:{' '}
          <span className="font-semibold">{data.latest}</span>{' '}
          <span className="text-success-700 dark:text-success-300">(running {data.current})</span>
          {showLink && (
            <>
              {' · '}
              <a
                href={data.release_notes_url!}
                target="_blank"
                rel="noopener noreferrer"
                className="font-medium underline underline-offset-2 hover:text-success-700 dark:hover:text-success-50"
              >
                Release notes
              </a>
            </>
          )}
        </p>
        <button
          type="button"
          onClick={handleDismiss}
          className="shrink-0 rounded px-2 py-1 text-xs text-success-700 hover:text-success-900 dark:text-success-300 dark:hover:text-success-100"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

export default UpdateBanner;
