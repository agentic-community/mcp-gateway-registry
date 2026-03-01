import React, {
  useState,
  useEffect,
} from 'react';
import {
  SystemStats,
} from '../types/stats';


/**
 * UptimeDisplay component shows system uptime with a hover tooltip containing detailed stats.
 *
 * Features:
 * - Fetches /api/stats every 60 seconds
 * - Displays human-readable uptime (e.g., "2 days 5 hours")
 * - Shows detailed system info on hover
 * - Handles loading and error states gracefully
 * - Hidden on mobile screens (<768px)
 */
const UptimeDisplay: React.FC = () => {
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [error, setError] = useState<boolean>(false);


  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/stats');
        if (!response.ok) {
          throw new Error('Failed to fetch stats');
        }
        const data = await response.json();
        setStats(data);
        setError(false);
      } catch (err) {
        console.error('Error fetching stats:', err);
        setError(true);
      }
    };

    // Initial fetch
    fetchStats();

    // Poll every 60 seconds
    const interval = setInterval(fetchStats, 60000);

    return () => clearInterval(interval);
  }, []);


  const formatUptime = (
    seconds: number,
  ): string => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);

    const parts: string[] = [];
    if (days > 0) {
      parts.push(`${days} day${days > 1 ? 's' : ''}`);
    }
    if (hours > 0) {
      parts.push(`${hours} hour${hours > 1 ? 's' : ''}`);
    }
    if (parts.length === 0 && minutes > 0) {
      parts.push(`${minutes} minute${minutes > 1 ? 's' : ''}`);
    }
    if (parts.length === 0) {
      return 'less than a minute';
    }

    return parts.join(' ');
  };


  if (error) {
    return (
      <div className="hidden md:flex items-center px-2.5 py-1 bg-gray-50 dark:bg-gray-900/20 rounded-md">
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
          Uptime: unavailable
        </span>
      </div>
    );
  }


  if (!stats) {
    return null;
  }


  const uptimeText = formatUptime(stats.uptime_seconds);
  const dbStatusColor = stats.database_status.status.toLowerCase() === 'healthy'
    ? 'text-green-600 dark:text-green-400'
    : 'text-red-600 dark:text-red-400';
  const authStatusColor = stats.auth_status.status.toLowerCase() === 'healthy'
    ? 'text-green-600 dark:text-green-400'
    : 'text-red-600 dark:text-red-400';


  return (
    <div className="hidden md:flex items-center px-2.5 py-1 bg-green-50 dark:bg-green-900/20 rounded-md group relative">
      <span className="text-xs font-medium text-green-700 dark:text-green-300">
        Uptime: {uptimeText}
      </span>

      {/* Tooltip on hover */}
      <div className="absolute right-0 top-full mt-2 w-80 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg ring-1 ring-black ring-opacity-5 p-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
            AI Gateway and Registry
          </h3>

          {/* Version & Start Time */}
          <div className="space-y-1 text-xs mb-3">
            <div className="flex justify-between gap-2">
              <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Version:</span>
              <span
                className="text-gray-900 dark:text-gray-100 font-mono truncate text-right"
                title={stats.version}
              >
                {stats.version}
              </span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Started:</span>
              <span className="text-gray-900 dark:text-gray-100 truncate text-right">
                {new Date(stats.started_at).toLocaleString()}
              </span>
            </div>
          </div>

          {/* Deployment */}
          <div className="mb-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Deployment
            </h4>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Type:</span>
                <span className="text-gray-900 dark:text-gray-100 truncate text-right">
                  {stats.deployment_type}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Mode:</span>
                <span className="text-gray-900 dark:text-gray-100 truncate text-right">
                  {stats.deployment_mode}
                </span>
              </div>
            </div>
          </div>

          {/* Registry Stats */}
          <div className="mb-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Registry Stats
            </h4>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Servers:</span>
                <span className="text-gray-900 dark:text-gray-100 text-right">
                  {stats.registry_stats.servers}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Agents:</span>
                <span className="text-gray-900 dark:text-gray-100 text-right">
                  {stats.registry_stats.agents}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Skills:</span>
                <span className="text-gray-900 dark:text-gray-100 text-right">
                  {stats.registry_stats.skills}
                </span>
              </div>
            </div>
          </div>

          {/* Database Status */}
          <div className="mb-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Database
            </h4>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Backend:</span>
                <span className="text-gray-900 dark:text-gray-100 truncate text-right">
                  {stats.database_status.backend}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Status:</span>
                <span className={`font-medium ${dbStatusColor} truncate text-right`}>
                  {stats.database_status.status}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Host:</span>
                <span
                  className="text-gray-900 dark:text-gray-100 font-mono text-xs truncate text-right"
                  title={stats.database_status.host}
                >
                  {stats.database_status.host}
                </span>
              </div>
            </div>
          </div>

          {/* Auth Server Status */}
          <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
            <h4 className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Auth Server
            </h4>
            <div className="space-y-1 text-xs">
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Provider:</span>
                <span className="text-gray-900 dark:text-gray-100 truncate text-right">
                  {stats.auth_status.provider}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">Status:</span>
                <span className={`font-medium ${authStatusColor} truncate text-right`}>
                  {stats.auth_status.status}
                </span>
              </div>
              <div className="flex justify-between gap-2">
                <span className="text-gray-500 dark:text-gray-400 flex-shrink-0">URL:</span>
                <span
                  className="text-gray-900 dark:text-gray-100 font-mono text-xs truncate text-right"
                  title={stats.auth_status.url}
                >
                  {stats.auth_status.url}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};


export default UptimeDisplay;
