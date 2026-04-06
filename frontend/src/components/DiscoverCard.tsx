import React from 'react';
import { StarIcon, WrenchScrewdriverIcon } from '@heroicons/react/24/solid';
import {
  ServerIcon,
  CpuChipIcon,
  SparklesIcon,
} from '@heroicons/react/24/outline';
import type { Server } from './ServerCard';
import type { Skill } from '../types/skill';


interface DiscoverCardProps {
  type: 'server' | 'agent' | 'skill';
  item: Server | Skill;
  onClick?: () => void;
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
 * Get the accent color classes based on item type.
 */
function _getTypeStyles(type: 'server' | 'agent' | 'skill') {
  if (type === 'server') {
    return {
      border: 'border-indigo-500/30 hover:border-indigo-400/60',
      badge: 'bg-indigo-500/20 text-indigo-300',
      icon: ServerIcon,
      label: 'Server',
    };
  }
  if (type === 'agent') {
    return {
      border: 'border-cyan-500/30 hover:border-cyan-400/60',
      badge: 'bg-cyan-500/20 text-cyan-300',
      icon: CpuChipIcon,
      label: 'Agent',
    };
  }
  return {
    border: 'border-amber-500/30 hover:border-amber-400/60',
    badge: 'bg-amber-500/20 text-amber-300',
    icon: SparklesIcon,
    label: 'Skill',
  };
}


const DiscoverCard: React.FC<DiscoverCardProps> = ({ type, item, onClick }) => {
  const styles = _getTypeStyles(type);
  const TypeIcon = styles.icon;

  // Extract common fields
  const name = item.name;
  const description = (item as any).description || '';
  const tags = (item as any).tags || [];

  // Rating: servers/agents use rating_details, skills use num_stars
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
    <div
      className={`rounded-lg border ${styles.border} bg-gray-800/60
        p-3 cursor-pointer transition-all duration-200 hover:bg-gray-800/90
        hover:shadow-lg hover:shadow-black/20`}
      onClick={onClick}
    >
      {/* Top row: type badge + name */}
      <div className="flex items-center gap-2 mb-1.5">
        <span className={`inline-flex items-center gap-1 px-1.5 py-0.5
          rounded text-xs font-medium ${styles.badge}`}>
          <TypeIcon className="h-3 w-3" />
          {styles.label}
        </span>
        <h3 className="text-sm font-semibold text-gray-100 truncate flex-1">
          {name}
        </h3>
      </div>

      {/* Description - 1 line */}
      {description && (
        <p className="text-xs text-gray-400 line-clamp-1 mb-1.5">
          {description}
        </p>
      )}

      {/* Bottom row: tags + rating + tools */}
      <div className="flex items-center justify-between gap-2">
        {/* Tags (show up to 2) */}
        <div className="flex items-center gap-1 min-w-0 flex-1">
          {tags.slice(0, 2).map((tag: string) => (
            <span
              key={tag}
              className="px-1.5 py-0.5 rounded text-[10px] bg-gray-700 text-gray-400 truncate"
            >
              #{tag}
            </span>
          ))}
          {tags.length > 2 && (
            <span className="text-[10px] text-gray-500">
              +{tags.length - 2}
            </span>
          )}
        </div>

        {/* Rating + tools */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {rating > 0 && (
            <span className="inline-flex items-center gap-0.5 text-xs text-yellow-400">
              <StarIcon className="h-3 w-3" />
              {rating.toFixed(1)}
              {ratingCount > 0 && (
                <span className="text-gray-500">({ratingCount})</span>
              )}
            </span>
          )}
          {toolCount > 0 && (
            <span className="inline-flex items-center gap-0.5 text-xs text-blue-400">
              <WrenchScrewdriverIcon className="h-3 w-3" />
              {toolCount}
            </span>
          )}
        </div>
      </div>
    </div>
  );
};

export default DiscoverCard;
