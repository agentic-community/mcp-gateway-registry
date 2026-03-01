import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import {
  ChartBarIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

interface UsageSummaryItem {
  name: string;
  count: number;
}

interface TimeSeriesBucket {
  period: string;
  count: number;
}

interface StatusDistribution {
  status_2xx: number;
  status_4xx: number;
  status_5xx: number;
}

interface UserActivityItem {
  username: string;
  total: number;
  operations: UsageSummaryItem[];
}

interface AuditStatisticsData {
  total_events: number;
  top_users: UsageSummaryItem[];
  top_servers: UsageSummaryItem[];
  top_operations: UsageSummaryItem[];
  activity_timeline: TimeSeriesBucket[];
  status_distribution: StatusDistribution;
  user_activity: UserActivityItem[];
}

interface AuditStatisticsProps {
  stream: 'registry_api' | 'mcp_access';
  days?: number;
  username?: string;
}

const STORAGE_KEY = 'audit-statistics-collapsed';

const BarChart: React.FC<{
  items: UsageSummaryItem[];
  color: string;
  emptyMessage?: string;
}> = ({ items, color, emptyMessage = 'No data available' }) => {
  if (!items.length) {
    return <p className="text-sm text-gray-400 italic py-2">{emptyMessage}</p>;
  }

  const maxCount = Math.max(...items.map((i) => i.count));

  return (
    <div className="space-y-1.5">
      {items.map((item) => (
        <div key={item.name} className="flex items-center gap-2">
          <span className="text-xs text-gray-700 dark:text-gray-300 w-28 truncate" title={item.name}>
            {item.name}
          </span>
          <div className="flex-1 bg-gray-100 dark:bg-gray-700 rounded-full h-3.5">
            <div
              className={`${color} h-3.5 rounded-full transition-all duration-300`}
              style={{ width: `${Math.max((item.count / maxCount) * 100, 2)}%` }}
            />
          </div>
          <span className="text-xs text-gray-500 dark:text-gray-400 w-10 text-right tabular-nums">
            {item.count.toLocaleString()}
          </span>
        </div>
      ))}
    </div>
  );
};

const StatusBar: React.FC<{ distribution: StatusDistribution }> = ({ distribution }) => {
  const total = distribution.status_2xx + distribution.status_4xx + distribution.status_5xx;
  if (total === 0) {
    return <p className="text-sm text-gray-400 italic py-2">No data available</p>;
  }

  const segments = [
    { label: '2xx', count: distribution.status_2xx, color: 'bg-green-500', textColor: 'text-green-600 dark:text-green-400' },
    { label: '4xx', count: distribution.status_4xx, color: 'bg-yellow-500', textColor: 'text-yellow-600 dark:text-yellow-400' },
    { label: '5xx', count: distribution.status_5xx, color: 'bg-red-500', textColor: 'text-red-600 dark:text-red-400' },
  ];

  return (
    <div>
      {/* Stacked bar */}
      <div className="flex h-5 rounded-full overflow-hidden bg-gray-100 dark:bg-gray-700 mb-2">
        {segments.map((seg) =>
          seg.count > 0 ? (
            <div
              key={seg.label}
              className={`${seg.color} transition-all duration-300`}
              style={{ width: `${(seg.count / total) * 100}%` }}
              title={`${seg.label}: ${seg.count.toLocaleString()} (${((seg.count / total) * 100).toFixed(1)}%)`}
            />
          ) : null
        )}
      </div>
      {/* Legend */}
      <div className="flex gap-4 text-xs">
        {segments.map((seg) => (
          <div key={seg.label} className="flex items-center gap-1">
            <div className={`w-2.5 h-2.5 rounded-full ${seg.color}`} />
            <span className={seg.textColor}>
              {seg.label}: {seg.count.toLocaleString()} ({total > 0 ? ((seg.count / total) * 100).toFixed(1) : 0}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

const WEEKDAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

/**
 * Fill in missing days so every day in the range has an entry.
 * The API only returns days with events, so days with 0 events are missing.
 */
function _fillTimelineDays(timeline: TimeSeriesBucket[], days: number): TimeSeriesBucket[] {
  const countByDate = new Map(timeline.map((t) => [t.period, t.count]));
  const filled: TimeSeriesBucket[] = [];
  const now = new Date();

  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    const key = d.toISOString().slice(0, 10);
    filled.push({ period: key, count: countByDate.get(key) || 0 });
  }

  return filled;
}

function _formatDateLabel(period: string): string {
  const d = new Date(period + 'T00:00:00');
  const weekday = WEEKDAY_NAMES[d.getDay()];
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${weekday} ${month}/${day}`;
}

const VB_W = 600;
const VB_H = 180;
const PAD = { top: 20, right: 50, bottom: 32, left: 45 };

const TimelineChart: React.FC<{ timeline: TimeSeriesBucket[]; days: number }> = ({ timeline, days }) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const filled = _fillTimelineDays(timeline, days);
  const maxCount = Math.max(...filled.map((t) => t.count), 1);

  if (!filled.length) {
    return <p className="text-sm text-gray-400 italic py-2">No data available</p>;
  }

  const plotW = VB_W - PAD.left - PAD.right;
  const plotH = VB_H - PAD.top - PAD.bottom;

  const points = filled.map((b, i) => {
    const x = PAD.left + (filled.length > 1 ? (i / (filled.length - 1)) * plotW : plotW / 2);
    const y = PAD.top + plotH - (b.count / maxCount) * plotH;
    return { x, y, ...b };
  });

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
  const areaPath = `${linePath} L${points[points.length - 1].x},${PAD.top + plotH} L${points[0].x},${PAD.top + plotH} Z`;

  const gridValues = [0, Math.round(maxCount / 2), maxCount];
  const gridLines = gridValues.map((v) => ({
    y: PAD.top + plotH - (v / maxCount) * plotH,
    label: v >= 1000 ? `${(v / 1000).toFixed(1)}k` : String(v),
  }));

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full"
      >
        {/* Horizontal grid lines + Y-axis labels */}
        {gridLines.map((g, i) => (
          <g key={i}>
            <line
              x1={PAD.left}
              y1={g.y}
              x2={VB_W - PAD.right}
              y2={g.y}
              className="stroke-gray-300 dark:stroke-gray-600"
              strokeWidth="1"
              strokeDasharray={i === 0 ? undefined : '4,3'}
            />
            <text
              x={PAD.left - 6}
              y={g.y}
              className="fill-gray-400 dark:fill-gray-500"
              fontSize="11"
              dominantBaseline="middle"
              textAnchor="end"
            >
              {g.label}
            </text>
          </g>
        ))}

        {/* Area fill */}
        <path d={areaPath} className="fill-blue-500/15 dark:fill-blue-400/15" />

        {/* Line */}
        <path
          d={linePath}
          fill="none"
          className="stroke-blue-500 dark:stroke-blue-400"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Data points */}
        {points.map((p, i) => (
          <circle
            key={p.period}
            cx={p.x}
            cy={p.y}
            r={hoverIndex === i ? 5 : (p.count > 0 ? 3.5 : 2)}
            className={
              p.count > 0
                ? 'fill-blue-500 dark:fill-blue-400'
                : 'fill-gray-300 dark:fill-gray-600'
            }
            stroke="white"
            strokeWidth="1.5"
          />
        ))}

        {/* Hover tooltip */}
        {hoverIndex !== null && points[hoverIndex] && (() => {
          const hp = points[hoverIndex];
          const label = `${hp.count.toLocaleString()} events`;
          const boxW = label.length * 7 + 16;
          const boxH = 22;
          const boxX = Math.max(4, Math.min(hp.x - boxW / 2, VB_W - boxW - 4));
          const boxY = Math.max(2, hp.y - boxH - 10);
          return (
            <g>
              <line
                x1={hp.x} y1={PAD.top} x2={hp.x} y2={PAD.top + plotH}
                className="stroke-blue-400/50"
                strokeWidth="1"
                strokeDasharray="4,3"
              />
              <rect x={boxX} y={boxY} width={boxW} height={boxH} rx="4"
                className="fill-gray-800 dark:fill-gray-200" opacity="0.92"
              />
              <text x={boxX + boxW / 2} y={boxY + boxH / 2 + 1}
                className="fill-white dark:fill-gray-800"
                fontSize="11" fontWeight="600" textAnchor="middle" dominantBaseline="middle"
              >
                {label}
              </text>
            </g>
          );
        })()}

        {/* Invisible hit areas for hover */}
        {points.map((p, i) => (
          <rect
            key={`hit-${p.period}`}
            x={p.x - (plotW / filled.length) / 2}
            y={0}
            width={plotW / filled.length}
            height={VB_H}
            fill="transparent"
            onMouseEnter={() => setHoverIndex(i)}
            onMouseLeave={() => setHoverIndex(null)}
          />
        ))}

        {/* X-axis labels */}
        {points.map((p) => (
          <text
            key={`label-${p.period}`}
            x={p.x}
            y={VB_H - 6}
            className="fill-gray-400 dark:fill-gray-500"
            fontSize="10"
            textAnchor="middle"
          >
            {_formatDateLabel(p.period)}
          </text>
        ))}
      </svg>
    </div>
  );
};

const UserActivityTable: React.FC<{ items: UserActivityItem[] }> = ({ items }) => {
  if (!items.length) {
    return <p className="text-sm text-gray-400 italic py-2">No user activity data</p>;
  }

  return (
    <div className="overflow-auto max-h-[160px]">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-white dark:bg-gray-800">
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="text-left py-1 pr-2 font-medium text-gray-500 dark:text-gray-400">User</th>
            <th className="text-right py-1 px-2 font-medium text-gray-500 dark:text-gray-400">Total</th>
            <th className="text-left py-1 pl-2 font-medium text-gray-500 dark:text-gray-400">Top Operations</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.username} className="border-b border-gray-100 dark:border-gray-700/50">
              <td className="py-1.5 pr-2 text-gray-700 dark:text-gray-300 font-medium truncate max-w-[100px]" title={item.username}>
                {item.username}
              </td>
              <td className="py-1.5 px-2 text-right text-gray-500 dark:text-gray-400 tabular-nums">
                {item.total.toLocaleString()}
              </td>
              <td className="py-1.5 pl-2">
                <div className="flex flex-wrap gap-1">
                  {item.operations.slice(0, 3).map((op) => (
                    <span
                      key={op.name}
                      className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
                      title={`${op.name}: ${op.count}`}
                    >
                      {op.name}
                      <span className="text-gray-400 dark:text-gray-500">({op.count})</span>
                    </span>
                  ))}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const AuditStatistics: React.FC<AuditStatisticsProps> = ({ stream, days = 7, username }) => {
  const [data, setData] = useState<AuditStatisticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored === null ? true : stored === 'true';
    } catch {
      return true;
    }
  });

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchStatistics = useCallback(async (currentStream: string, currentDays: number, currentUsername?: string) => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, string | number> = { stream: currentStream, days: currentDays };
      if (currentUsername) {
        params.username = currentUsername;
      }
      const res = await axios.get('/api/audit/statistics', { params });
      setData(res.data);
    } catch (err) {
      console.error('Failed to fetch audit statistics:', err);
      setError('Failed to load statistics');
    } finally {
      setLoading(false);
    }
  }, []);

  // Debounced fetch when stream, days, or username change
  useEffect(() => {
    if (collapsed) return;

    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }

    debounceRef.current = setTimeout(() => {
      fetchStatistics(stream, days, username);
    }, 300);

    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, [stream, days, username, collapsed, fetchStatistics]);

  const toggleCollapsed = () => {
    const next = !collapsed;
    setCollapsed(next);
    try {
      localStorage.setItem(STORAGE_KEY, String(next));
    } catch {
      // Ignore localStorage errors
    }
  };

  const handleRefresh = () => {
    fetchStatistics(stream, days, username);
  };

  const isMcpStream = stream === 'mcp_access';

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 mb-6">
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 cursor-pointer select-none"
        onClick={toggleCollapsed}
      >
        <div className="flex items-center gap-2">
          {collapsed ? (
            <ChevronRightIcon className="h-4 w-4 text-gray-500" />
          ) : (
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          )}
          <ChartBarIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Statistics
          </h3>
          {data && !collapsed && (
            <span className="text-xs text-gray-400 ml-2">
              {data.total_events.toLocaleString()} events (last {days} days){username ? ` - filtered by "${username}"` : ''}
            </span>
          )}
        </div>
        {!collapsed && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              handleRefresh();
            }}
            disabled={loading}
            className="p-1.5 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors disabled:opacity-50"
            title="Refresh statistics"
          >
            <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        )}
      </div>

      {/* Content */}
      {!collapsed && (
        <div className="px-4 pb-4">
          {loading && !data ? (
            <div className="flex items-center justify-center py-8">
              <ArrowPathIcon className="h-6 w-6 text-gray-400 animate-spin" />
              <span className="ml-2 text-sm text-gray-400">Loading statistics...</span>
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <p className="text-sm text-red-500">{error}</p>
              <button
                onClick={handleRefresh}
                className="mt-2 text-sm text-blue-500 hover:text-blue-600"
              >
                Retry
              </button>
            </div>
          ) : data ? (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Top Users */}
              <div className="border border-gray-100 dark:border-gray-700 rounded-lg p-3">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Top Users
                </h4>
                <BarChart
                  items={data.top_users.filter((u) => u.name !== 'anonymous')}
                  color="bg-blue-500"
                  emptyMessage="No user data"
                />
                {(() => {
                  const anon = data.top_users.find((u) => u.name === 'anonymous');
                  return anon ? (
                    <p className="text-xs text-gray-400 dark:text-gray-500 italic mt-2">
                      + {anon.count.toLocaleString()} anonymous events (unauthenticated API calls, health checks, login attempts)
                    </p>
                  ) : null;
                })()}
              </div>

              {/* Top Operations */}
              <div className="border border-gray-100 dark:border-gray-700 rounded-lg p-3">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  {isMcpStream ? 'Top MCP Methods' : 'Top Operations'}
                </h4>
                <BarChart items={data.top_operations} color="bg-purple-500" emptyMessage="No operation data" />
              </div>

              {/* Top MCP Servers (MCP stream only) */}
              {isMcpStream && (
                <div className="border border-gray-100 dark:border-gray-700 rounded-lg p-3">
                  <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                    Top MCP Servers
                  </h4>
                  <BarChart items={data.top_servers} color="bg-indigo-500" emptyMessage="No server data" />
                </div>
              )}

              {/* Status Distribution */}
              <div className="border border-gray-100 dark:border-gray-700 rounded-lg p-3">
                <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                  Status Distribution
                </h4>
                <StatusBar distribution={data.status_distribution} />
              </div>

              {/* User Activity + Activity Timeline - split panel */}
              <div className={`border border-gray-100 dark:border-gray-700 rounded-lg p-3 lg:col-span-2`}>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {/* Left: User Activity Table */}
                  <div>
                    <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                      User Activity Breakdown
                    </h4>
                    <UserActivityTable items={data.user_activity} />
                  </div>
                  {/* Right: Activity Timeline */}
                  <div>
                    <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
                      Activity Timeline (Last {days} Days)
                    </h4>
                    <TimelineChart timeline={data.activity_timeline} days={days} />
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
};

export default AuditStatistics;
