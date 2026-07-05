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
      className="bg-emerald-900/40 border border-emerald-700/50 rounded-lg p-3 my-3 mx-6"
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm text-emerald-100">
          A newer registry release is available:{' '}
          <span className="font-semibold">{data.latest}</span>{' '}
          <span className="text-emerald-300">(running {data.current})</span>
          {showLink && (
            <>
              {' · '}
              <a
                href={data.release_notes_url!}
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:text-emerald-50"
              >
                Release notes
              </a>
            </>
          )}
        </p>
        <button
          type="button"
          onClick={handleDismiss}
          className="text-xs text-emerald-300 hover:text-emerald-100 focus:outline-none"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

export default UpdateBanner;
