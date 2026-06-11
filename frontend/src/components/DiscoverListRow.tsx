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
  Square3Stack3DIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline';
import ServerCard from './ServerCard';
import type { Server } from './ServerCard';
import AgentCard from './AgentCard';
import SkillCard from './SkillCard';
import type { Skill } from '../types/skill';
import VirtualServerCard from './VirtualServerCard';
import type { VirtualServerInfo } from '../types/virtualServer';
import CustomEntityCard from './CustomEntityCard';
import type {
  CustomEntityRecord,
  CustomTypeDescriptor,
} from '../types/customEntity';


type ItemType = 'server' | 'agent' | 'skill' | 'virtual' | 'custom';


interface DiscoverListRowProps {
  type: ItemType;
  item: Server | Skill | VirtualServerInfo | CustomEntityRecord;
  onToggle: (path: string, enabled: boolean) => void;
  onEdit?: (item: any) => void;
  onDelete?: (path: string) => any;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  authToken?: string | null;
  // Custom-entity rows render a CustomEntityCard when expanded, which is
  // descriptor-driven and uses record-based (not path-based) handlers.
  customDescriptor?: CustomTypeDescriptor;
  customCanModify?: boolean;
  onCustomView?: (record: CustomEntityRecord) => void;
  onCustomEdit?: (record: CustomEntityRecord) => void;
  onCustomDelete?: (record: CustomEntityRecord) => void;
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
function _getTypeBadge(type: ItemType) {
  if (type === 'server') {
    return {
      bg: 'bg-indigo-500/15 text-indigo-300',
      icon: ServerIcon,
      label: 'Server',
    };
  }
  if (type === 'virtual') {
    return {
      bg: 'bg-teal-500/15 text-teal-300',
      icon: Square3Stack3DIcon,
      label: 'Virtual',
    };
  }
  if (type === 'agent') {
    return {
      bg: 'bg-cyan-500/15 text-cyan-300',
      icon: CpuChipIcon,
      label: 'Agent',
    };
  }
  if (type === 'custom') {
    return {
      bg: 'bg-teal-500/15 text-teal-300',
      icon: Square3Stack3DIcon,
      label: 'Custom',
    };
  }
  return {
    bg: 'bg-amber-500/15 text-amber-300',
    icon: SparklesIcon,
    label: 'Skill',
  };
}


/**
 * Get the source registry name for a server or agent, if it comes from
 * a federated peer or an external registry.
 */
function _getServerRegistrySource(server: Server): string | null {
  // Federated peer registry
  if (server.sync_metadata?.is_federated && server.sync_metadata?.source_peer_id) {
    return server.sync_metadata.source_peer_id;
  }
  // External registry identified by tags
  const tags = server.tags || [];
  const externalTags = ['anthropic-registry', 'workday-asor', 'asor', 'federated'];
  const match = tags.find(t => externalTags.includes(t));
  if (match) {
    return match;
  }
  return null;
}


/**
 * Extract display fields from any item type in a uniform way.
 */
function _extractDisplayFields(
  type: ItemType,
  item: Server | Skill | VirtualServerInfo | CustomEntityRecord
) {
  if (type === 'virtual') {
    const vs = item as VirtualServerInfo;
    return {
      name: vs.server_name,
      description: vs.description || '',
      tags: vs.tags || [],
      rating: _getAverageRating(vs.rating_details),
      ratingCount: vs.rating_details?.length || 0,
      toolCount: vs.tool_count || 0,
      registrySource: null as string | null,
    };
  }
  if (type === 'skill') {
    const skill = item as Skill;
    const source = skill.registry_name && skill.registry_name !== 'local'
      ? skill.registry_name
      : null;
    return {
      name: skill.name,
      description: skill.description || '',
      tags: skill.tags || [],
      rating: skill.num_stars || 0,
      ratingCount: 0,
      toolCount: 0,
      registrySource: source,
    };
  }
  if (type === 'custom') {
    const record = item as CustomEntityRecord;
    return {
      name: record.name,
      description: record.description || '',
      tags: record.tags || [],
      rating: _getAverageRating(record.rating_details),
      ratingCount: record.rating_details?.length || 0,
      toolCount: 0,
      registrySource: null as string | null,
    };
  }
  // server or agent
  const server = item as Server;
  return {
    name: server.name,
    description: (server as any).description || '',
    tags: (server as any).tags || [],
    rating: _getAverageRating(server.rating_details),
    ratingCount: server.rating_details?.length || 0,
    toolCount: (server as any).num_tools || 0,
    registrySource: _getServerRegistrySource(server),
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
  customDescriptor,
  customCanModify = false,
  onCustomView,
  onCustomEdit,
  onCustomDelete,
}) => {
  const [expanded, setExpanded] = useState(false);

  const badge = _getTypeBadge(type);
  const TypeIcon = badge.icon;
  const fields = _extractDisplayFields(type, item);

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

        {/* Registry source label */}
        {fields.registrySource && (
          <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded
            text-[11px] font-medium bg-purple-500/15 text-purple-300 flex-shrink-0">
            <GlobeAltIcon className="h-3 w-3" />
            {fields.registrySource}
          </span>
        )}

        {/* Name */}
        <span className="text-sm font-semibold text-gray-100 whitespace-nowrap flex-shrink-0">
          {fields.name}
        </span>

        {/* Separator */}
        {fields.description && (
          <span className="text-gray-600 flex-shrink-0">&middot;</span>
        )}

        {/* Description */}
        <span className="text-sm text-gray-400 whitespace-nowrap overflow-hidden text-ellipsis flex-1 min-w-0">
          {fields.description}
        </span>

        {/* Tags (up to 2) */}
        {fields.tags.length > 0 && (
          <div className="hidden sm:flex items-center gap-1 flex-shrink-0">
            {fields.tags.slice(0, 2).map((tag: string) => (
              <span
                key={tag}
                className="px-1.5 py-0.5 rounded text-[11px] bg-gray-700/60 text-gray-400"
              >
                #{tag}
              </span>
            ))}
            {fields.tags.length > 2 && (
              <span className="text-[11px] text-gray-500">+{fields.tags.length - 2}</span>
            )}
          </div>
        )}

        {/* Tool count */}
        {fields.toolCount > 0 && (
          <span className="hidden md:inline-flex items-center gap-1 text-xs text-blue-400 flex-shrink-0">
            <WrenchScrewdriverIcon className="h-3 w-3" />
            {fields.toolCount}
          </span>
        )}

        {/* Rating */}
        {fields.rating > 0 && (
          <span className="inline-flex items-center gap-1 text-xs text-yellow-400 flex-shrink-0">
            <StarIcon className="h-3 w-3" />
            {fields.rating.toFixed(1)}
            {fields.ratingCount > 0 && (
              <span className="text-gray-500">({fields.ratingCount})</span>
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
          {type === 'virtual' && (
            <VirtualServerCard
              virtualServer={item as VirtualServerInfo}
              canModify={true}
              onToggle={onToggle}
              onEdit={onEdit as any}
              onDelete={onDelete as any}
              onShowToast={onShowToast as any}
              authToken={authToken}
            />
          )}
          {type === 'custom' && customDescriptor && (
            <CustomEntityCard
              descriptor={customDescriptor}
              record={item as CustomEntityRecord}
              canModify={customCanModify}
              onView={onCustomView ?? (() => {})}
              onEdit={onCustomEdit ?? (() => {})}
              onDelete={onCustomDelete ?? (() => {})}
              authToken={authToken}
              onShowToast={onShowToast}
            />
          )}
          {type === 'custom' && !customDescriptor && (
            // Descriptor failed to load (deleted type, network error): show an
            // inline notice instead of rendering an empty expanded box.
            <div className="px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-900/50 text-sm text-gray-500 dark:text-gray-400">
              Schema unavailable for this entry (type descriptor could not be loaded).
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DiscoverListRow;
