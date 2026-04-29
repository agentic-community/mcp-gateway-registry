import React, { useState, useEffect, useCallback } from 'react';
import {
  ArrowDownTrayIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import axios from 'axios';
import JSZip from 'jszip';


interface ExportableCollection {
  id: string;
  label: string;
  description: string;
  endpoint: string;
  queryParams: Record<string, string>;
  dataKey: string | null;
  countKey: string | null;
  filename: string;
  isPaginated: boolean;
  paginationLimit: number;
  paginationOffsetKey?: string;
}

const EXPORTABLE_COLLECTIONS: ExportableCollection[] = [
  {
    id: 'servers',
    label: 'Servers',
    description: 'All registered MCP servers',
    endpoint: '/api/servers',
    queryParams: {},
    dataKey: 'servers',
    countKey: 'total_count',
    filename: 'servers',
    isPaginated: true,
    paginationLimit: 500,
  },
  {
    id: 'agents',
    label: 'Agents',
    description: 'All registered AI agents',
    endpoint: '/api/agents',
    queryParams: {},
    dataKey: 'agents',
    countKey: 'total_count',
    filename: 'agents',
    isPaginated: true,
    paginationLimit: 500,
  },
  {
    id: 'skills',
    label: 'Skills',
    description: 'All registered skills (including disabled)',
    endpoint: '/api/skills',
    queryParams: { include_disabled: 'true' },
    dataKey: 'skills',
    countKey: 'total_count',
    filename: 'skills',
    isPaginated: true,
    paginationLimit: 500,
  },
  {
    id: 'virtual-servers',
    label: 'Virtual Servers',
    description: 'All virtual server configurations',
    endpoint: '/api/virtual-servers',
    queryParams: {},
    dataKey: null,
    countKey: null,
    filename: 'virtual-servers',
    isPaginated: false,
    paginationLimit: 500,
  },
  {
    id: 'federation-peers',
    label: 'Federation Peers',
    description: 'All configured federation peers',
    endpoint: '/api/peers',
    queryParams: {},
    dataKey: null,
    countKey: null,
    filename: 'federation-peers',
    isPaginated: false,
    paginationLimit: 500,
  },
  {
    id: 'federation-configs',
    label: 'Federation Configs',
    description: 'Federation configuration settings',
    endpoint: '/api/federation/configs',
    queryParams: {},
    dataKey: 'configs',
    countKey: null,
    filename: 'federation-configs',
    isPaginated: false,
    paginationLimit: 500,
  },
  {
    id: 'registry-card',
    label: 'Registry Card',
    description: 'Registry metadata and card information',
    endpoint: '/api/registry/v0.1/card',
    queryParams: {},
    dataKey: null,
    countKey: null,
    filename: 'registry-card',
    isPaginated: false,
    paginationLimit: 1,
  },
  {
    id: 'iam-users',
    label: 'IAM Users',
    description: 'All users and service accounts',
    endpoint: '/api/management/iam/users',
    queryParams: {},
    dataKey: 'users',
    countKey: null,
    filename: 'iam-users',
    isPaginated: false,
    paginationLimit: 500,
  },
  {
    id: 'iam-groups',
    label: 'IAM Groups',
    description: 'All IAM groups and scopes',
    endpoint: '/api/management/iam/groups',
    queryParams: {},
    dataKey: 'groups',
    countKey: null,
    filename: 'iam-groups',
    isPaginated: false,
    paginationLimit: 500,
  },
  {
    id: 'iam-m2m-clients',
    label: 'IAM M2M Clients',
    description: 'All machine-to-machine service accounts',
    endpoint: '/api/iam/m2m-clients',
    queryParams: {},
    dataKey: 'items',
    countKey: 'total',
    filename: 'iam-m2m-clients',
    isPaginated: true,
    paginationLimit: 500,
    paginationOffsetKey: 'skip',
  },
  {
    id: 'scopes',
    label: 'Scopes',
    description: 'Authorization scopes, server access rules, and group permissions',
    endpoint: '/api/export/scopes',
    queryParams: {},
    dataKey: 'scopes',
    countKey: 'total_count',
    filename: 'scopes',
    isPaginated: false,
    paginationLimit: 500,
  },
];


function _buildDateSuffix(): string {
  return new Date().toISOString().slice(0, 10);
}


async function _fetchAllPages(
  collection: ExportableCollection,
): Promise<any[]> {
  const { endpoint, queryParams, dataKey, isPaginated, paginationLimit } = collection;
  const offsetKey = collection.paginationOffsetKey || 'offset';

  if (!isPaginated) {
    const response = await axios.get(endpoint, { params: queryParams });
    const json = response.data;
    if (dataKey) {
      return json[dataKey] || [];
    }
    return Array.isArray(json) ? json : [json];
  }

  const allRecords: any[] = [];
  let offset = 0;
  while (true) {
    const params = {
      ...queryParams,
      limit: String(paginationLimit),
      [offsetKey]: String(offset),
    };
    const response = await axios.get(endpoint, { params });
    const json = response.data;
    const page = dataKey ? (json[dataKey] || []) : json;
    allRecords.push(...page);
    if (page.length < paginationLimit) {
      break;
    }
    offset += paginationLimit;
  }
  return allRecords;
}


function _triggerBlobDownload(
  blob: Blob,
  filename: string,
): void {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}


async function _recordAuditEvent(
  exportType: string,
  collections: string[],
): Promise<void> {
  try {
    await axios.post('/api/export/audit-event', {
      export_type: exportType,
      collections,
    });
  } catch {
    // Audit event recording is best-effort; do not block the export
  }
}


async function _fetchCount(
  collection: ExportableCollection,
): Promise<number> {
  const { endpoint, queryParams, dataKey, countKey, isPaginated, paginationLimit } = collection;
  const offsetKey = collection.paginationOffsetKey || 'offset';

  try {
    // Fast path: API returns a count field (servers, agents, skills, m2m-clients)
    if (countKey && isPaginated) {
      const response = await axios.get(endpoint, {
        params: { ...queryParams, limit: '1', [offsetKey]: '0' },
      });
      return response.data[countKey] ?? 0;
    }

    // Fallback: fetch data and count the array length
    const params: Record<string, string> = { ...queryParams };
    if (isPaginated) {
      params.limit = String(paginationLimit);
      params[offsetKey] = '0';
    }
    const response = await axios.get(endpoint, { params });
    const json = response.data;

    if (dataKey) {
      return Array.isArray(json[dataKey]) ? json[dataKey].length : 0;
    }
    return Array.isArray(json) ? json.length : 1;
  } catch {
    return 0;
  }
}


interface DataExportProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


const DataExport: React.FC<DataExportProps> = ({ onShowToast }) => {
  const [counts, setCounts] = useState<Record<string, number | null>>({});
  const [downloading, setDownloading] = useState<Record<string, boolean>>({});
  const [downloadingAll, setDownloadingAll] = useState(false);
  const [completedInZip, setCompletedInZip] = useState<Set<string>>(new Set());
  const [loadingCounts, setLoadingCounts] = useState(true);

  const fetchCounts = useCallback(async () => {
    setLoadingCounts(true);
    const results = await Promise.allSettled(
      EXPORTABLE_COLLECTIONS.map(async (col) => {
        const count = await _fetchCount(col);
        return { id: col.id, count };
      })
    );

    const newCounts: Record<string, number | null> = {};
    // Promise.allSettled preserves input order, so index maps to collection
    for (let i = 0; i < results.length; i++) {
      const result = results[i];
      const collectionId = EXPORTABLE_COLLECTIONS[i].id;
      if (result.status === 'fulfilled') {
        newCounts[collectionId] = result.value.count;
      } else {
        newCounts[collectionId] = null;
      }
    }
    setCounts(newCounts);
    setLoadingCounts(false);
  }, []);

  useEffect(() => {
    fetchCounts();
  }, [fetchCounts]);

  const handleDownload = useCallback(async (collection: ExportableCollection) => {
    setDownloading((prev) => ({ ...prev, [collection.id]: true }));
    try {
      const data = await _fetchAllPages(collection);
      const dateSuffix = _buildDateSuffix();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      _triggerBlobDownload(blob, `${collection.filename}-export-${dateSuffix}.json`);
      await _recordAuditEvent('single', [collection.id]);
      onShowToast(`Downloaded ${collection.label} (${data.length} records)`, 'success');
    } catch (err: any) {
      onShowToast(`Failed to download ${collection.label}: ${err.message}`, 'error');
    } finally {
      setDownloading((prev) => ({ ...prev, [collection.id]: false }));
    }
  }, [onShowToast]);

  const handleDownloadAll = useCallback(async () => {
    setDownloadingAll(true);
    setCompletedInZip(new Set());
    const zip = new JSZip();
    const dateSuffix = _buildDateSuffix();
    const failedIds: string[] = [];

    for (const collection of EXPORTABLE_COLLECTIONS) {
      try {
        const data = await _fetchAllPages(collection);
        const jsonStr = JSON.stringify(data, null, 2);
        zip.file(`${collection.filename}-export-${dateSuffix}.json`, jsonStr);
        setCompletedInZip((prev) => new Set(prev).add(collection.id));
      } catch (err: any) {
        failedIds.push(collection.id);
      }
    }

    try {
      const blob = await zip.generateAsync({ type: 'blob' });
      _triggerBlobDownload(blob, `registry-export-${dateSuffix}.zip`);
      const exportedIds = EXPORTABLE_COLLECTIONS
        .filter((c) => !failedIds.includes(c.id))
        .map((c) => c.id);
      await _recordAuditEvent('all', exportedIds);

      if (failedIds.length > 0) {
        const failedLabels = EXPORTABLE_COLLECTIONS
          .filter((c) => failedIds.includes(c.id))
          .map((c) => c.label);
        onShowToast(
          `ZIP downloaded with errors. Failed: ${failedLabels.join(', ')}`,
          'error',
        );
      } else {
        onShowToast('All collections downloaded as ZIP', 'success');
      }
    } catch (err: any) {
      onShowToast(`Failed to create ZIP: ${err.message}`, 'error');
    } finally {
      setDownloadingAll(false);
    }
  }, [onShowToast]);

  const isAnyDownloading = downloadingAll || Object.values(downloading).some(Boolean);

  return (
    <div>
      {/* Page header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          Data Export
        </h2>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Download registry data as JSON for debugging and auditing purposes.
        </p>
      </div>

      {/* Sensitive data warning banner */}
      <div className="mb-6 flex items-start gap-3 rounded-lg border border-amber-300 dark:border-amber-700
                      bg-amber-50 dark:bg-amber-900/20 px-4 py-3">
        <ExclamationTriangleIcon className="h-5 w-5 text-amber-500 dark:text-amber-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-amber-800 dark:text-amber-300">
          Exported data may contain sensitive information such as email addresses,
          client IDs, and configuration details. Handle exported files with care.
        </p>
      </div>

      {/* Collection table */}
      <div className="overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead className="bg-gray-50 dark:bg-gray-900/50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Collection
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider hidden sm:table-cell">
                Description
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Records
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            {EXPORTABLE_COLLECTIONS.map((collection) => (
              <tr key={collection.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900 dark:text-gray-100">
                  {collection.label}
                </td>
                <td className="px-6 py-4 text-sm text-gray-500 dark:text-gray-400 hidden sm:table-cell">
                  {collection.description}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700 dark:text-gray-300">
                  {loadingCounts ? (
                    <span className="inline-block w-8 h-4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
                  ) : (
                    counts[collection.id] ?? '—'
                  )}
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right">
                  <div className="flex items-center justify-end gap-2">
                    {downloadingAll && completedInZip.has(collection.id) && (
                      <CheckCircleIcon className="h-5 w-5 text-green-500" />
                    )}
                    <button
                      onClick={() => handleDownload(collection)}
                      disabled={isAnyDownloading}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-md
                                 text-purple-700 dark:text-purple-300
                                 bg-purple-50 dark:bg-purple-900/30
                                 hover:bg-purple-100 dark:hover:bg-purple-900/50
                                 disabled:opacity-50 disabled:cursor-not-allowed
                                 transition-colors"
                    >
                      {downloading[collection.id] ? (
                        <ArrowPathIcon className="h-4 w-4 animate-spin" />
                      ) : (
                        <ArrowDownTrayIcon className="h-4 w-4" />
                      )}
                      Download
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Download All button */}
      <div className="mt-6 flex justify-end">
        <button
          onClick={handleDownloadAll}
          disabled={isAnyDownloading}
          className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-medium rounded-lg
                     text-white bg-purple-600 hover:bg-purple-700
                     disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors"
        >
          {downloadingAll ? (
            <>
              <ArrowPathIcon className="h-4 w-4 animate-spin" />
              Downloading...
            </>
          ) : (
            <>
              <ArrowDownTrayIcon className="h-4 w-4" />
              Download All as ZIP
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default DataExport;
