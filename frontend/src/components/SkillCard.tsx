import React, { useState, useCallback, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  SparklesIcon,
  PencilIcon,
  TrashIcon,
  GlobeAltIcon,
  LockClosedIcon,
  UserGroupIcon,
  InformationCircleIcon,
  ArrowTopRightOnSquareIcon,
  WrenchScrewdriverIcon,
  CheckCircleIcon,
  XCircleIcon,
  ArrowPathIcon,
  ClockIcon,
  ClipboardIcon,
  ArrowDownTrayIcon,
  ShieldCheckIcon,
  ShieldExclamationIcon,
} from '@heroicons/react/24/outline';
import { Skill } from '../types/skill';
import StatusBadge from './StatusBadge';
import StarRatingWidget from './StarRatingWidget';
import SecurityScanModal from './SecurityScanModal';
import useEscapeKey from '../hooks/useEscapeKey';

/**
 * Props for the SkillCard component.
 */
interface SkillCardProps {
  skill: Skill & { [key: string]: any };
  onToggle: (path: string, enabled: boolean) => void;
  onEdit?: (skill: Skill) => void;
  onDelete?: (path: string) => void;
  canModify?: boolean;
  canToggle?: boolean;
  canHealthCheck?: boolean;
  onRefreshSuccess?: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  onSkillUpdate?: (path: string, updates: Partial<Skill>) => void;
  authToken?: string | null;
}

// Helper function to parse YAML frontmatter from markdown
const parseYamlFrontmatter = (content: string): { frontmatter: Record<string, string> | null; body: string } => {
  // Check if content starts with --- (YAML frontmatter delimiter)
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = content.match(frontmatterRegex);

  if (match) {
    const yamlContent = match[1];
    const body = match[2];

    // Simple YAML parsing for key: value pairs
    const frontmatter: Record<string, string> = {};
    const lines = yamlContent.split('\n');
    for (const line of lines) {
      const colonIndex = line.indexOf(':');
      if (colonIndex > 0) {
        const key = line.substring(0, colonIndex).trim();
        const value = line.substring(colonIndex + 1).trim();
        if (key && value) {
          frontmatter[key] = value;
        }
      }
    }

    return { frontmatter: Object.keys(frontmatter).length > 0 ? frontmatter : null, body };
  }

  return { frontmatter: null, body: content };
};


// Helper function to format time since last checked
const formatTimeSince = (timestamp: string | null | undefined): string | null => {
  if (!timestamp) {
    return null;
  }

  try {
    const now = new Date();
    const lastChecked = new Date(timestamp);

    if (isNaN(lastChecked.getTime())) {
      return null;
    }

    const diffMs = now.getTime() - lastChecked.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);
    const diffMinutes = Math.floor(diffSeconds / 60);
    const diffHours = Math.floor(diffMinutes / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) {
      return `${diffDays}d ago`;
    } else if (diffHours > 0) {
      return `${diffHours}h ago`;
    } else if (diffMinutes > 0) {
      return `${diffMinutes}m ago`;
    } else {
      return `${diffSeconds}s ago`;
    }
  } catch (error) {
    console.error('formatTimeSince error:', error, 'for timestamp:', timestamp);
    return null;
  }
};

/**
 * SkillCard component for displaying Agent Skills.
 *
 * Uses amber/orange tones to distinguish from servers (purple) and agents (cyan).
 */
const SkillCard: React.FC<SkillCardProps> = React.memo(({
  skill,
  onToggle,
  onEdit,
  onDelete,
  canModify,
  canToggle = true,
  canHealthCheck = true,
  onShowToast,
  onSkillUpdate,
  authToken
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [skillMdContent, setSkillMdContent] = useState<string | null>(null);

  useEscapeKey(() => setShowDetails(false), showDetails);
  const [loadingToolCheck, setLoadingToolCheck] = useState(false);
  const [toolCheckResult, setToolCheckResult] = useState<any>(null);
  const [loadingHealthCheck, setLoadingHealthCheck] = useState(false);
  const [healthStatus, setHealthStatus] = useState<'healthy' | 'unhealthy' | 'unknown'>(
    skill.health_status || 'unknown'
  );
  const [lastCheckedTime, setLastCheckedTime] = useState<string | null>(
    skill.last_checked_time || null
  );
  const [showSecurityScan, setShowSecurityScan] = useState(false);
  const [securityScanResult, setSecurityScanResult] = useState<any>(null);
  const [loadingSecurityScan, setLoadingSecurityScan] = useState(false);

  // Sync health status from props when skill changes
  useEffect(() => {
    setHealthStatus(skill.health_status || 'unknown');
    setLastCheckedTime(skill.last_checked_time || null);
  }, [skill.health_status, skill.last_checked_time]);

  // Extract skill name from path for API calls
  // skill.path is like "/skills/doc-coauthoring", API routes already have /skills prefix
  // so we need just "/doc-coauthoring" for the path parameter
  const getSkillApiPath = (path: string) => {
    if (path.startsWith('/skills/')) {
      return path.replace('/skills/', '/');
    }
    return path;
  };
  const skillApiPath = getSkillApiPath(skill.path);

  // Fetch security scan status on mount to show correct icon color
  useEffect(() => {
    const fetchSecurityScan = async () => {
      try {
        const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
        const response = await axios.get(
          `/api/skills${skillApiPath}/security-scan`,
          headers ? { headers } : undefined
        );
        setSecurityScanResult(response.data);
      } catch {
        // Silently ignore - no scan result available
      }
    };
    fetchSecurityScan();
  }, [skillApiPath, authToken]);

  const getVisibilityIcon = () => {
    switch (skill.visibility) {
      case 'public':
        return <GlobeAltIcon className="h-3 w-3" />;
      case 'group':
        return <UserGroupIcon className="h-3 w-3" />;
      default:
        return <LockClosedIcon className="h-3 w-3" />;
    }
  };

  const getVisibilityColor = () => {
    switch (skill.visibility) {
      case 'public':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border border-green-200 dark:border-green-700';
      case 'group':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border border-blue-200 dark:border-blue-700';
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600';
    }
  };

  const handleViewDetails = useCallback(async () => {
    setShowDetails(true);
    setLoadingDetails(true);
    setSkillMdContent(null);

    try {
      // Fetch SKILL.md content via backend proxy to avoid CORS issues
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/skills${skillApiPath}/content`,
        headers ? { headers } : undefined
      );
      setSkillMdContent(response.data.content);
    } catch (error: any) {
      console.error('Failed to fetch SKILL.md content:', error);
      const detail = error.response?.data?.detail;
      if (error.response?.status === 409 && detail) {
        setSkillMdContent(`> **Content Drift Detected**\n>\n> ${detail}\n>\n> Re-register this skill to update the integrity baseline and re-enable it.`);
      } else if (onShowToast) {
        onShowToast(detail || 'Failed to load SKILL.md content', 'error');
      }
    } finally {
      setLoadingDetails(false);
    }
  }, [skillApiPath, authToken, onShowToast]);

  const handleCheckTools = useCallback(async () => {
    if (loadingToolCheck) return;

    setLoadingToolCheck(true);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/skills${skillApiPath}/tools`,
        headers ? { headers } : undefined
      );
      setToolCheckResult(response.data);
      if (onShowToast) {
        const result = response.data;
        if (result.all_available) {
          onShowToast('All required tools are available', 'success');
        } else {
          onShowToast(`Missing tools: ${result.missing_tools?.join(', ') || 'Unknown'}`, 'error');
        }
      }
    } catch (error: any) {
      console.error('Failed to check tool availability:', error);
      if (onShowToast) {
        onShowToast('Failed to check tool availability', 'error');
      }
    } finally {
      setLoadingToolCheck(false);
    }
  }, [skill.path, authToken, loadingToolCheck, onShowToast]);

  const handleRefreshHealth = useCallback(async () => {
    if (loadingHealthCheck) return;

    setLoadingHealthCheck(true);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/skills${skillApiPath}/health`,
        headers ? { headers } : undefined
      );

      const newStatus = response.data.healthy ? 'healthy' : 'unhealthy';
      setHealthStatus(newStatus);
      setLastCheckedTime(new Date().toISOString());

      // Update parent if callback provided
      if (onSkillUpdate) {
        onSkillUpdate(skill.path, {
          health_status: newStatus,
          last_checked_time: new Date().toISOString()
        } as any);
      }

      if (onShowToast) {
        onShowToast(
          response.data.healthy
            ? 'SKILL.md is accessible'
            : `SKILL.md check failed: ${response.data.error || 'Unknown error'}`,
          response.data.healthy ? 'success' : 'error'
        );
      }
    } catch (error: any) {
      console.error('Failed to check skill health:', error);
      setHealthStatus('unhealthy');
      if (onShowToast) {
        onShowToast('Failed to check skill health', 'error');
      }
    } finally {
      setLoadingHealthCheck(false);
    }
  }, [skill.path, authToken, loadingHealthCheck, onShowToast, onSkillUpdate]);

  const handleViewSecurityScan = useCallback(async () => {
    if (loadingSecurityScan) return;

    setShowSecurityScan(true);
    setLoadingSecurityScan(true);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/skills${skillApiPath}/security-scan`,
        headers ? { headers } : undefined
      );
      setSecurityScanResult(response.data);
    } catch (error: any) {
      if (error.response?.status !== 404) {
        if (onShowToast) {
          onShowToast('Failed to load security scan results', 'error');
        }
      }
      setSecurityScanResult(null);
    } finally {
      setLoadingSecurityScan(false);
    }
  }, [skillApiPath, authToken, loadingSecurityScan, onShowToast]);

  const handleRescan = useCallback(async () => {
    const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
    const response = await axios.post(
      `/api/skills${skillApiPath}/rescan`,
      undefined,
      headers ? { headers } : undefined
    );
    setSecurityScanResult(response.data);
  }, [skillApiPath, authToken]);

  const getSecurityIconState = () => {
    if (!securityScanResult) {
      return { Icon: ShieldCheckIcon, color: 'text-gray-400 dark:text-gray-500', title: 'View security scan results' };
    }
    if (securityScanResult.scan_failed) {
      return { Icon: ShieldExclamationIcon, color: 'text-red-500 dark:text-red-400', title: 'Security scan failed' };
    }
    const hasVulnerabilities = securityScanResult.critical_issues > 0 ||
      securityScanResult.high_severity > 0 ||
      securityScanResult.medium_severity > 0 ||
      securityScanResult.low_severity > 0;
    if (hasVulnerabilities) {
      return { Icon: ShieldExclamationIcon, color: 'text-red-500 dark:text-red-400', title: 'Security issues found' };
    }
    return { Icon: ShieldCheckIcon, color: 'text-green-500 dark:text-green-400', title: 'Security scan passed' };
  };

  return (
    <>
      <div className="group rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 h-full flex flex-col bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border-2 border-amber-200 dark:border-amber-700 hover:border-amber-300 dark:hover:border-amber-600">
        {/* Header */}
        <div className="p-5 pb-4">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-3 flex-wrap">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white truncate">
                  {skill.name}
                </h3>
                <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 dark:from-amber-900/30 dark:to-orange-900/30 dark:text-amber-300 rounded-full flex-shrink-0 border border-amber-200 dark:border-amber-600">
                  SKILL
                </span>
                <span className={`px-2 py-0.5 text-xs font-semibold rounded-full flex-shrink-0 flex items-center gap-1 ${getVisibilityColor()}`}>
                  {getVisibilityIcon()}
                  {skill.visibility.toUpperCase()}
                </span>
                {skill.status && skill.status !== 'active' && (
                  <StatusBadge status={skill.status} />
                )}
              </div>

              <code className="text-xs text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 px-2 py-1 rounded font-mono">
                {skill.path}
              </code>
              {skill.version && (
                <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">
                  v{skill.version}
                </span>
              )}
              {skill.author && (
                <span className="ml-2 text-xs text-gray-500 dark:text-gray-400">
                  by {skill.author}
                </span>
              )}
            </div>

            <div className="flex items-center gap-1">
              {canModify && (
                <>
                  <button
                    className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                    onClick={() => onEdit?.(skill)}
                    title="Edit skill"
                  >
                    <PencilIcon className="h-4 w-4" />
                  </button>
                  <button
                    className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-lg transition-all duration-200 flex-shrink-0"
                    onClick={() => onDelete?.(skillApiPath)}
                    title="Delete skill"
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </>
              )}

              {/* Tool Check Button */}
              {skill.allowed_tools && skill.allowed_tools.length > 0 && (
                <button
                  onClick={handleCheckTools}
                  disabled={loadingToolCheck}
                  className={`p-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0 ${
                    toolCheckResult?.all_available === true
                      ? 'text-green-500 dark:text-green-400'
                      : toolCheckResult?.all_available === false
                      ? 'text-red-500 dark:text-red-400'
                      : 'text-gray-400 dark:text-gray-500'
                  }`}
                  title="Check tool availability"
                >
                  <WrenchScrewdriverIcon className={`h-4 w-4 ${loadingToolCheck ? 'animate-spin' : ''}`} />
                </button>
              )}

              {/* Security Scan Button */}
              <button
                onClick={handleViewSecurityScan}
                className={`p-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0 ${getSecurityIconState().color}`}
                title={getSecurityIconState().title}
                aria-label={getSecurityIconState().title}
              >
                {React.createElement(getSecurityIconState().Icon, { className: `h-4 w-4 ${loadingSecurityScan ? 'animate-pulse' : ''}` })}
              </button>

              {/* Details Button */}
              <button
                onClick={handleViewDetails}
                className="p-2 text-gray-400 hover:text-amber-600 dark:hover:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                title="View SKILL.md content"
              >
                <InformationCircleIcon className="h-4 w-4" />
              </button>
            </div>
          </div>

          {/* Description */}
          <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed line-clamp-2 mb-4">
            {skill.description || 'No description available'}
          </p>

          {/* Tags */}
          {skill.tags && skill.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-4">
              {skill.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className={`px-2 py-1 text-xs font-medium rounded ${
                    tag === 'security-pending' || tag === 'content-drifted'
                      ? 'bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300 border border-red-200 dark:border-red-700'
                      : 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'
                  }`}
                >
                  #{tag}
                </span>
              ))}
              {skill.tags.length > 3 && (
                <span className="px-2 py-1 text-xs font-medium bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300 rounded">
                  +{skill.tags.length - 3}
                </span>
              )}
            </div>
          )}

          {/* Target Agents */}
          {skill.target_agents && skill.target_agents.length > 0 && (
            <div className="mb-4">
              <span className="text-xs text-gray-500 dark:text-gray-400">Target agents: </span>
              <span className="text-xs text-amber-700 dark:text-amber-300">
                {skill.target_agents.join(', ')}
              </span>
            </div>
          )}

          {/* Tools Count */}
          {skill.allowed_tools && skill.allowed_tools.length > 0 && (
            <div className="flex items-center gap-2 mb-4">
              <WrenchScrewdriverIcon className="h-4 w-4 text-amber-600 dark:text-amber-400" />
              <span className="text-xs text-gray-600 dark:text-gray-300">
                {skill.allowed_tools.length} tool{skill.allowed_tools.length !== 1 ? 's' : ''} required
              </span>
              {toolCheckResult && (
                toolCheckResult.all_available ? (
                  <CheckCircleIcon className="h-4 w-4 text-green-500" title="All tools available" />
                ) : (
                  <XCircleIcon className="h-4 w-4 text-red-500" title="Some tools missing" />
                )
              )}
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="px-5 pb-4">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="p-1.5 bg-amber-50 dark:bg-amber-900/30 rounded">
                <SparklesIcon className="h-4 w-4 text-amber-600 dark:text-amber-400" />
              </div>
              <div>
                <div className="text-xs text-gray-500 dark:text-gray-400">Registry</div>
                <div className="text-sm font-semibold text-gray-900 dark:text-white">
                  {skill.registry_name || 'local'}
                </div>
              </div>
            </div>

            {/* Rating Widget */}
            <StarRatingWidget
              resourceType="skills"
              path={skillApiPath}
              initialRating={skill.num_stars || 0}
              authToken={authToken}
              onShowToast={onShowToast}
            />

            {/* SKILL.md Link */}
            {skill.skill_md_url && (
              <a
                href={skill.skill_md_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1 text-xs text-amber-700 dark:text-amber-300 hover:underline"
              >
                <ArrowTopRightOnSquareIcon className="h-3 w-3" />
                SKILL.md
              </a>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-auto px-5 py-4 border-t border-amber-100 dark:border-amber-700 bg-amber-50/50 dark:bg-amber-900/30 rounded-b-2xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Status Indicator */}
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  skill.is_enabled
                    ? 'bg-green-400 shadow-lg shadow-green-400/30'
                    : 'bg-gray-300 dark:bg-gray-600'
                }`} />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {skill.is_enabled ? 'Enabled' : 'Disabled'}
                </span>
                {!skill.is_enabled && skill.tags?.includes('content-drifted') && (
                  <span className="text-xs text-red-600 dark:text-red-400 font-medium" title="Skill content changed since registration. Re-register to update the baseline.">
                    — content drifted
                  </span>
                )}
                {!skill.is_enabled && skill.tags?.includes('security-pending') && !skill.tags?.includes('content-drifted') && (
                  <span className="text-xs text-red-600 dark:text-red-400 font-medium">
                    — security review pending
                  </span>
                )}
              </div>

              <div className="w-px h-4 bg-amber-200 dark:bg-amber-600" />

              {/* Health Status */}
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  healthStatus === 'healthy'
                    ? 'bg-emerald-400 shadow-lg shadow-emerald-400/30'
                    : healthStatus === 'unhealthy'
                    ? 'bg-red-400 shadow-lg shadow-red-400/30'
                    : 'bg-amber-400 shadow-lg shadow-amber-400/30'
                }`} />
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  {healthStatus === 'healthy' ? 'Healthy' :
                   healthStatus === 'unhealthy' ? 'Unhealthy' : 'Unknown'}
                </span>
              </div>
            </div>

            {/* Controls */}
            <div className="flex items-center gap-3">
              {/* Last Checked */}
              {(() => {
                const timeText = formatTimeSince(lastCheckedTime);
                return lastCheckedTime && timeText ? (
                  <div className="text-xs text-gray-500 dark:text-gray-300 flex items-center gap-1.5">
                    <ClockIcon className="h-3.5 w-3.5" />
                    <span>{timeText}</span>
                  </div>
                ) : null;
              })()}

              {/* Refresh Health Button */}
              {canHealthCheck && (
                <button
                  onClick={handleRefreshHealth}
                  disabled={loadingHealthCheck}
                  className="p-2.5 text-gray-500 hover:text-amber-600 dark:hover:text-amber-400 hover:bg-amber-50 dark:hover:bg-amber-900/20 rounded-lg transition-all duration-200 disabled:opacity-50"
                  title="Check SKILL.md accessibility"
                  aria-label={`Check health for ${skill.name}`}
                >
                  <ArrowPathIcon className={`h-4 w-4 ${loadingHealthCheck ? 'animate-spin' : ''}`} />
                </button>
              )}

              {/* Toggle Switch */}
              {canToggle && (
                <label className="relative inline-flex items-center cursor-pointer" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={skill.is_enabled}
                    onChange={(e) => {
                      e.stopPropagation();
                      onToggle(skill.path, e.target.checked);
                    }}
                    className="sr-only peer"
                  />
                  <div className={`relative w-12 h-6 rounded-full transition-colors duration-200 ease-in-out ${
                    skill.is_enabled
                      ? 'bg-amber-600'
                      : 'bg-gray-300 dark:bg-gray-600'
                  }`}>
                    <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-200 ease-in-out ${
                      skill.is_enabled ? 'translate-x-6' : 'translate-x-0'
                    }`} />
                  </div>
                </label>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Skill Details Modal */}
      {showDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {skill.name}
              </h3>
              <button
                onClick={() => setShowDetails(false)}
                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-4 mb-4 pb-4 border-b border-gray-200 dark:border-gray-700">
              {skill.skill_md_url && (
                <a
                  href={skill.skill_md_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-sm text-amber-700 dark:text-amber-300 hover:underline"
                >
                  <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                  View Skill
                </a>
              )}
              {skill.repository_url && (
                <a
                  href={skill.repository_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-sm text-amber-700 dark:text-amber-300 hover:underline"
                >
                  <ArrowTopRightOnSquareIcon className="h-4 w-4" />
                  View Repo
                </a>
              )}
              {skillMdContent && (
                <>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(skillMdContent);
                      if (onShowToast) {
                        onShowToast('SKILL.md copied to clipboard', 'success');
                      }
                    }}
                    className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400 hover:text-amber-700 dark:hover:text-amber-300 transition-colors"
                    title="Copy to clipboard"
                  >
                    <ClipboardIcon className="h-4 w-4" />
                    Copy
                  </button>
                  <button
                    onClick={() => {
                      const blob = new Blob([skillMdContent], { type: 'text/markdown' });
                      const url = URL.createObjectURL(blob);
                      const a = document.createElement('a');
                      a.href = url;
                      a.download = `${skill.name || 'skill'}.md`;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                      URL.revokeObjectURL(url);
                    }}
                    className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400 hover:text-amber-700 dark:hover:text-amber-300 transition-colors"
                    title="Download SKILL.md"
                  >
                    <ArrowDownTrayIcon className="h-4 w-4" />
                    Download
                  </button>
                </>
              )}
            </div>

            {loadingDetails ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600"></div>
              </div>
            ) : skillMdContent ? (
              (() => {
                const { frontmatter, body } = parseYamlFrontmatter(skillMdContent);
                return (
                  <>
                    {/* YAML Frontmatter Table */}
                    {frontmatter && (
                      <div className="mb-6 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                        <table className="w-full text-sm">
                          <tbody>
                            {Object.entries(frontmatter).map(([key, value]) => (
                              <tr key={key} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                                <td className="px-4 py-2 bg-gray-50 dark:bg-gray-900/50 font-medium text-gray-700 dark:text-gray-300 w-1/4">
                                  {key}
                                </td>
                                <td className="px-4 py-2 text-gray-900 dark:text-white">
                                  {value}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {/* Markdown Body */}
                    <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-amber-800 dark:prose-headings:text-amber-200 prose-a:text-amber-600 dark:prose-a:text-amber-400 prose-code:bg-gray-100 dark:prose-code:bg-gray-900 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-100 dark:prose-pre:bg-gray-900">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
                    </div>
                  </>
                );
              })()
            ) : (
              <div className="text-center py-12 text-gray-500">
                <p>Could not load SKILL.md content.</p>
                <p className="mt-2 text-sm">
                  Try visiting the{' '}
                  <a
                    href={skill.skill_md_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-amber-600 hover:underline"
                  >
                    source URL
                  </a>{' '}
                  directly.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      {/* Security Scan Modal */}
      <SecurityScanModal
        resourceName={skill.name}
        resourceType="skill"
        isOpen={showSecurityScan}
        onClose={() => setShowSecurityScan(false)}
        loading={loadingSecurityScan}
        scanResult={securityScanResult}
        onRescan={canModify ? handleRescan : undefined}
        canRescan={canModify}
        onShowToast={onShowToast}
      />
    </>
  );
});

SkillCard.displayName = 'SkillCard';

export default SkillCard;
