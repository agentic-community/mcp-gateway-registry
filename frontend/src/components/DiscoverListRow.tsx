import React, { useState } from 'react';
import {
  StarIcon,
  WrenchScrewdriverIcon,
  ChevronDownIcon,
  ChevronUpIcon,
} from '@heroicons/react/24/solid';
import {
  ServerIcon,
  CpuChipIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import ServerCard from './ServerCard';
import type { Server } from './ServerCard';
import AgentCard from './AgentCard';
import SkillCard from './SkillCard';
import type { Skill } from '../types/skill';


interface DiscoverListRowProps {
  type: 'server' | 'agent' | 'skill';
  item: Server | Skill;
  onToggle: (path: string, enabled: boolean) => void;
  onEdit?: (item: any) => void;
  onDelete?: (path: string) => any;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  authToken?: string | null;
}


/**
 * Get average rating from rating_details array.
 */
function _getAverageRating(
  ratingDetails: Array<{ user: string; rating: number }> | undefined
): number {
  if (!ratingDetails || ratingDetails.length === 0) {
    return 0;
  }
  const sum = ratingDetails.reduce((acc, r) => acc + r.rating, 0);
  return sum / ratingDetails.length;
}


/**
 * Get type badge styling by item type.
 */
function _getTypeBadge(type: 'server' | 'agent' | 'skill') {
  if (type === 'server') {
    return {
      bg: 'bg-indigo-500/15 text-indigo-300',
      icon: ServerIcon,
      label: 'Server',
    };
  }
  if (type === 'agent') {
    return {
      bg: 'bg-cyan-500/15 text-cyan-300',
      icon: CpuChipIcon,
      label: 'Agent',
    };
  }
  return {
    bg: 'bg-amber-500/15 text-amber-300',
    icon: SparklesIcon,
    label: 'Skill',
  };
}


const DiscoverListRow: React.FC<DiscoverListRowProps> = ({
  type,
  item,
  onToggle,
  onEdit,
  onDelete,
  onShowToast,
  authToken,
}) => {
  const [expanded, setExpanded] = useState(false);

  const badge = _getTypeBadge(type);
  const TypeIcon = badge.icon;

  const name = item.name;
  const description = (item as any).description || '';
  const tags: string[] = (item as any).tags || [];

  // Rating
  let rating = 0;
  let ratingCount = 0;
  if (type === 'skill') {
    rating = (item as Skill).num_stars || 0;
  } else {
    const server = item as Server;
    rating = _getAverageRating(server.rating_details);
    ratingCount = server.rating_details?.length || 0;
  }

  // Tool count for servers/agents
  const toolCount = type !== 'skill' ? (item as any).num_tools || 0 : 0;

  return (
    <div className="mb-1.5">
      {/* Compact row */}
      <div
        className={`flex items-center gap-3 px-4 py-2.5 rounded-lg cursor-pointer
          transition-colors duration-150
          border border-gray-700/50
          ${expanded
            ? 'bg-gray-800/90 border-gray-600'
            : 'bg-gray-800/40 hover:bg-gray-800/70 hover:border-gray-600/50'
          }`}
        onClick={() => setExpanded(!expanded)}
        data-testid={`list-row-${type}-${item.path}`}
      >
        {/* Type badge */}
        <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded
          text-xs font-semibold flex-shrink-0 ${badge.bg}`}>
          <TypeIcon className="h-3 w-3" />
          {badge.label}
        </span>

        {/* Name */}
        <span className="text-sm font-semibold text-gray-100 whitespace-nowrap flex-shrink-0">
          {name}
        </span>

        {/* Separator */}
        {description && (
          <span className="text-gray-600 flex-shrink-0">&middot;</span>
        )}

        {/* Description */}
        <span className="text-sm text-gray-400 whitespace-nowrap overflow-hidden text-ellipsis flex-1 min-w-0">
          {description}
        </span>

        {/* Tags (up to 2) */}
        {tags.length > 0 && (
          <div className="hidden sm:flex items-center gap-1 flex-shrink-0">
            {tags.slice(0, 2).map((tag: string) => (
              <span
                key={tag}
                className="px-1.5 py-0.5 rounded text-[11px] bg-gray-700/60 text-gray-400"
              >
                #{tag}
              </span>
            ))}
            {tags.length > 2 && (
              <span className="text-[11px] text-gray-500">+{tags.length - 2}</span>
            )}
          </div>
        )}

        {/* Tool count */}
        {toolCount > 0 && (
          <span className="hidden md:inline-flex items-center gap-1 text-xs text-blue-400 flex-shrink-0">
            <WrenchScrewdriverIcon className="h-3 w-3" />
            {toolCount}
          </span>
        )}

        {/* Rating */}
        {rating > 0 && (
          <span className="inline-flex items-center gap-1 text-xs text-yellow-400 flex-shrink-0">
            <StarIcon className="h-3 w-3" />
            {rating.toFixed(1)}
            {ratingCount > 0 && (
              <span className="text-gray-500">({ratingCount})</span>
            )}
          </span>
        )}

        {/* Expand chevron */}
        {expanded ? (
          <ChevronUpIcon className="h-4 w-4 text-gray-400 flex-shrink-0" />
        ) : (
          <ChevronDownIcon className="h-4 w-4 text-gray-500 flex-shrink-0" />
        )}
      </div>

      {/* Expanded detail: full card */}
      {expanded && (
        <div className="mt-1 ml-4 mr-4" data-testid={`expanded-${type}-${item.path}`}>
          {type === 'server' && (
            <ServerCard
              server={item as Server}
              onToggle={onToggle}
              onEdit={onEdit}
              onDelete={onDelete}
              onShowToast={onShowToast}
              authToken={authToken}
            />
          )}
          {type === 'agent' && (
            <AgentCard
              agent={item as any}
              onToggle={onToggle}
              onEdit={onEdit}
              onDelete={onDelete}
              onShowToast={onShowToast}
              authToken={authToken}
            />
          )}
          {type === 'skill' && (
            <SkillCard
              skill={item as Skill}
              onToggle={onToggle}
              onEdit={onEdit}
              onDelete={onDelete}
              onShowToast={onShowToast}
              authToken={authToken}
            />
          )}
        </div>
      )}
    </div>
  );
};

export default DiscoverListRow;
