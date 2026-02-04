import React, { useState, useCallback } from 'react';
import axios from 'axios';
import {
  SparklesIcon,
  PencilIcon,
  GlobeAltIcon,
  LockClosedIcon,
  UserGroupIcon,
  InformationCircleIcon,
  ArrowTopRightOnSquareIcon,
  WrenchScrewdriverIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';

/**
 * Skill interface representing an Agent Skill.
 */
export interface Skill {
  name: string;
  path: string;
  description?: string;
  skill_md_url: string;
  version?: string;
  author?: string;
  visibility: 'public' | 'private' | 'group';
  is_enabled: boolean;
  tags?: string[];
  owner?: string;
  registry_name?: string;
  target_agents?: string[];
  allowed_tools?: Array<{
    tool_name: string;
    server_path?: string;
    capabilities?: string[];
  }>;
  requirements?: Array<{
    type: string;
    target: string;
    min_version?: string;
    required?: boolean;
  }>;
}

/**
 * Props for the SkillCard component.
 */
interface SkillCardProps {
  skill: Skill & { [key: string]: any };
  onToggle: (path: string, enabled: boolean) => void;
  onEdit?: (skill: Skill) => void;
  canModify?: boolean;
  canToggle?: boolean;
  onRefreshSuccess?: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  onSkillUpdate?: (path: string, updates: Partial<Skill>) => void;
  authToken?: string | null;
}

/**
 * SkillCard component for displaying Agent Skills.
 *
 * Uses amber/orange tones to distinguish from servers (purple) and agents (cyan).
 */
const SkillCard: React.FC<SkillCardProps> = React.memo(({
  skill,
  onToggle,
  onEdit,
  canModify,
  canToggle = true,
  onRefreshSuccess,
  onShowToast,
  onSkillUpdate,
  authToken
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const [loadingDetails, setLoadingDetails] = useState(false);
  const [fullSkillDetails, setFullSkillDetails] = useState<any>(null);
  const [loadingToolCheck, setLoadingToolCheck] = useState(false);
  const [toolCheckResult, setToolCheckResult] = useState<any>(null);

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
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/skills${skill.path}`,
        headers ? { headers } : undefined
      );
      setFullSkillDetails(response.data);
    } catch (error) {
      console.error('Failed to fetch skill details:', error);
      if (onShowToast) {
        onShowToast('Failed to load skill details', 'error');
      }
    } finally {
      setLoadingDetails(false);
    }
  }, [skill.path, authToken, onShowToast]);

  const handleCheckTools = useCallback(async () => {
    if (loadingToolCheck) return;

    setLoadingToolCheck(true);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/skills${skill.path}/tools`,
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

  const handleCopyDetails = useCallback(
    async (data: any) => {
      try {
        await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
        onShowToast?.('Skill JSON copied to clipboard!', 'success');
      } catch (error) {
        console.error('Failed to copy JSON:', error);
        onShowToast?.('Failed to copy JSON', 'error');
      }
    },
    [onShowToast]
  );

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
                <button
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                  onClick={() => onEdit?.(skill)}
                  title="Edit skill"
                >
                  <PencilIcon className="h-4 w-4" />
                </button>
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

              {/* Details Button */}
              <button
                onClick={handleViewDetails}
                className="p-2 text-gray-400 hover:text-amber-600 dark:hover:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                title="View skill details (JSON)"
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
                  className="px-2 py-1 text-xs font-medium bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded"
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
              </div>
            </div>

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

      {/* Skill Details Modal */}
      {showDetails && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Skill Details: {skill.name}
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

            {loadingDetails ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600"></div>
              </div>
            ) : fullSkillDetails ? (
              <div className="space-y-4">
                <pre className="bg-gray-50 dark:bg-gray-900 p-4 rounded-lg overflow-x-auto text-xs font-mono text-gray-800 dark:text-gray-200">
                  {JSON.stringify(fullSkillDetails, null, 2)}
                </pre>
                <button
                  onClick={() => handleCopyDetails(fullSkillDetails)}
                  className="w-full px-4 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 rounded-md transition-colors"
                >
                  Copy JSON
                </button>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                No details available
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
});

SkillCard.displayName = 'SkillCard';

export default SkillCard;
