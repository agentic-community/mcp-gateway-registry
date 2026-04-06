import React, { useState, useMemo, useCallback } from 'react';
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useSemanticSearch } from '../hooks/useSemanticSearch';
import SemanticSearchResults from './SemanticSearchResults';
import DiscoverListRow from './DiscoverListRow';
import type { Server } from './ServerCard';
import type { Skill } from '../types/skill';
import type { VirtualServerInfo } from '../types/virtualServer';


// Path for the built-in AI Registry Tools server
const AI_REGISTRY_TOOLS_PATH = '/airegistry-tools/';

// Maximum featured items per category
const MAX_FEATURED = 4;


interface DiscoverTabProps {
  servers: Server[];
  agents: Server[];
  skills: Skill[];
  virtualServers: VirtualServerInfo[];
  externalServers: Server[];
  externalAgents: Server[];
  loading: boolean;
  onServerToggle: (path: string, enabled: boolean) => void;
  onServerEdit?: (server: Server) => void;
  onServerDelete?: (path: string) => Promise<void>;
  onAgentToggle: (path: string, enabled: boolean) => void;
  onAgentEdit?: (agent: Server) => void;
  onAgentDelete?: (path: string) => Promise<void>;
  onSkillToggle: (path: string, enabled: boolean) => void;
  onSkillEdit?: (skill: Skill) => void;
  onSkillDelete?: (path: string) => void;
  onVirtualServerToggle: (path: string, enabled: boolean) => void;
  onVirtualServerEdit?: (vs: VirtualServerInfo) => void;
  onVirtualServerDelete?: (path: string) => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  authToken?: string | null;
}


/**
 * Compute average rating from rating_details array.
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
 * Sort servers by average rating (descending), then alphabetically by name.
 */
function _sortServersByRating(servers: Server[]): Server[] {
  return [...servers].sort((a, b) => {
    const ratingDiff = _getAverageRating(b.rating_details) - _getAverageRating(a.rating_details);
    if (ratingDiff !== 0) return ratingDiff;
    return a.name.localeCompare(b.name);
  });
}


/**
 * Sort skills by num_stars (descending), then alphabetically by name.
 */
function _sortSkillsByStars(skills: Skill[]): Skill[] {
  return [...skills].sort((a, b) => {
    const ratingDiff = (b.num_stars || 0) - (a.num_stars || 0);
    if (ratingDiff !== 0) return ratingDiff;
    return a.name.localeCompare(b.name);
  });
}


/**
 * Sort virtual servers by rating then name.
 */
function _sortVirtualServersByRating(vs: VirtualServerInfo[]): VirtualServerInfo[] {
  return [...vs].sort((a, b) => {
    const ratingDiff = _getAverageRating(b.rating_details) - _getAverageRating(a.rating_details);
    if (ratingDiff !== 0) return ratingDiff;
    return a.server_name.localeCompare(b.server_name);
  });
}


/**
 * Check if an item matches a keyword search query.
 * Searches name, description, path, and tags.
 */
function _matchesKeyword(
  item: { name: string; description?: string; path: string; tags?: string[] },
  query: string
): boolean {
  const q = query.toLowerCase();
  return (
    item.name.toLowerCase().includes(q) ||
    (item.description || '').toLowerCase().includes(q) ||
    item.path.toLowerCase().includes(q) ||
    (item.tags || []).some(tag => tag.toLowerCase().includes(q))
  );
}


/**
 * Build a count fragment like "4 servers".
 */
function _countFragment(
  count: number,
  label: string
): string {
  const plural = count !== 1 ? 's' : '';
  return `${count} ${label}${plural}`;
}


/**
 * Build the summary text showing counts per category.
 * Default: "18 servers, 2 virtual, 8 agents, 4 skills, 3 external"
 * Searching: "3 servers" (only matched counts, no totals)
 */
function _buildSummaryText(
  totals: { servers: number; virtual: number; agents: number; skills: number; external: number },
  matched: { servers: number; virtual: number; agents: number; skills: number; external: number },
  isSearching: boolean
): string {
  const parts: string[] = [];

  // When searching, only show categories that have matches
  // When not searching, show all categories that have items
  const categories = [
    { total: totals.servers, match: matched.servers, label: 'server' },
    { total: totals.virtual, match: matched.virtual, label: 'virtual' },
    { total: totals.agents, match: matched.agents, label: 'agent' },
    { total: totals.skills, match: matched.skills, label: 'skill' },
    { total: totals.external, match: matched.external, label: 'external' },
  ];

  for (const cat of categories) {
    if (isSearching && cat.match > 0) {
      parts.push(_countFragment(cat.match, cat.label));
    } else if (!isSearching && cat.total > 0) {
      parts.push(_countFragment(cat.total, cat.label));
    }
  }

  if (parts.length === 0) {
    return isSearching ? 'No matches' : 'No items registered';
  }

  const prefix = isSearching ? 'Showing ' : '';
  return prefix + parts.join(', ');
}


/**
 * Check if a virtual server matches a keyword search query.
 */
function _virtualServerMatchesKeyword(
  vs: VirtualServerInfo,
  query: string
): boolean {
  const q = query.toLowerCase();
  return (
    vs.server_name.toLowerCase().includes(q) ||
    (vs.description || '').toLowerCase().includes(q) ||
    vs.path.toLowerCase().includes(q) ||
    (vs.tags || []).some(tag => tag.toLowerCase().includes(q))
  );
}


/**
 * Get featured items for the Discover landing page.
 * AI Registry Tools always first among servers if it exists.
 * Returns sorted, enabled items up to the max per category.
 */
function _getFeaturedItems(
  servers: Server[],
  agents: Server[],
  skills: Skill[],
  virtualServers: VirtualServerInfo[],
  externalServers: Server[],
  externalAgents: Server[],
  keywordFilter: string
) {
  // Filter enabled items
  const enabledServers = servers.filter(s => s.enabled);
  const enabledAgents = agents.filter(a => a.enabled);
  const enabledSkills = skills.filter(s => s.is_enabled);
  const enabledVirtual = virtualServers.filter(vs => vs.is_enabled);
  const enabledExtServers = externalServers.filter(s => s.enabled);
  const enabledExtAgents = externalAgents.filter(a => a.enabled);

  // Apply keyword filter if present
  const hasFilter = keywordFilter.length > 0;

  const filteredServers = hasFilter
    ? enabledServers.filter(s => _matchesKeyword(s, keywordFilter))
    : enabledServers;
  const filteredAgents = hasFilter
    ? enabledAgents.filter(a => _matchesKeyword(a, keywordFilter))
    : enabledAgents;
  const filteredSkills = hasFilter
    ? enabledSkills.filter(s => _matchesKeyword({
        name: s.name, description: s.description, path: s.path, tags: s.tags,
      }, keywordFilter))
    : enabledSkills;
  const filteredVirtual = hasFilter
    ? enabledVirtual.filter(vs => _virtualServerMatchesKeyword(vs, keywordFilter))
    : enabledVirtual;
  const filteredExtServers = hasFilter
    ? enabledExtServers.filter(s => _matchesKeyword(s, keywordFilter))
    : enabledExtServers;
  const filteredExtAgents = hasFilter
    ? enabledExtAgents.filter(a => _matchesKeyword(a, keywordFilter))
    : enabledExtAgents;

  // Sort and pick top items
  // AI Registry Tools goes first if it's in the filtered list
  const aiRegistryTools = filteredServers.find(s => s.path === AI_REGISTRY_TOOLS_PATH);
  const otherServers = filteredServers.filter(s => s.path !== AI_REGISTRY_TOOLS_PATH);
  const sortedOther = _sortServersByRating(otherServers);

  const featuredServers: Server[] = [];
  if (aiRegistryTools) {
    featuredServers.push(aiRegistryTools);
  }
  featuredServers.push(...sortedOther.slice(0, MAX_FEATURED - featuredServers.length));

  const featuredAgents = _sortServersByRating(filteredAgents).slice(0, MAX_FEATURED);
  const featuredSkills = _sortSkillsByStars(filteredSkills).slice(0, MAX_FEATURED);
  const featuredVirtual = _sortVirtualServersByRating(filteredVirtual).slice(0, MAX_FEATURED);
  const featuredExtServers = _sortServersByRating(filteredExtServers).slice(0, MAX_FEATURED);
  const featuredExtAgents = _sortServersByRating(filteredExtAgents).slice(0, MAX_FEATURED);

  return {
    featuredServers,
    featuredAgents,
    featuredSkills,
    featuredVirtual,
    featuredExtServers,
    featuredExtAgents,
    // Total enabled counts (before keyword filter + before MAX_FEATURED cap)
    totalServers: enabledServers.length,
    totalVirtual: enabledVirtual.length,
    totalAgents: enabledAgents.length,
    totalSkills: enabledSkills.length,
    totalExternal: enabledExtServers.length + enabledExtAgents.length,
    // Filtered counts (after keyword filter, before MAX_FEATURED cap)
    matchedServers: filteredServers.length,
    matchedVirtual: filteredVirtual.length,
    matchedAgents: filteredAgents.length,
    matchedSkills: filteredSkills.length,
    matchedExternal: filteredExtServers.length + filteredExtAgents.length,
    matchedExtServers: filteredExtServers.length,
    matchedExtAgents: filteredExtAgents.length,
  };
}


const DiscoverTab: React.FC<DiscoverTabProps> = ({
  servers,
  agents,
  skills,
  virtualServers,
  externalServers,
  externalAgents,
  loading,
  onServerToggle,
  onServerEdit,
  onServerDelete,
  onAgentToggle,
  onAgentEdit,
  onAgentDelete,
  onSkillToggle,
  onSkillEdit,
  onSkillDelete,
  onVirtualServerToggle,
  onVirtualServerEdit,
  onVirtualServerDelete,
  onShowToast,
  authToken,
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [committedQuery, setCommittedQuery] = useState('');

  // Semantic search (only fires when committedQuery is set via Enter)
  const {
    results: searchResults,
    loading: searchLoading,
    error: searchError,
  } = useSemanticSearch(committedQuery, {
    enabled: committedQuery.length >= 2,
  });

  const isSemanticActive = committedQuery.length >= 2;

  // Compute featured items with keyword filtering
  const {
    featuredServers,
    featuredAgents,
    featuredSkills,
    featuredVirtual,
    featuredExtServers,
    featuredExtAgents,
    totalServers, totalVirtual, totalAgents, totalSkills, totalExternal,
    matchedServers, matchedVirtual, matchedAgents, matchedSkills, matchedExternal,
    matchedExtServers, matchedExtAgents,
  } = useMemo(
    () => _getFeaturedItems(
      servers, agents, skills, virtualServers,
      externalServers, externalAgents,
      isSemanticActive ? '' : searchTerm
    ),
    [servers, agents, skills, virtualServers, externalServers, externalAgents, searchTerm, isSemanticActive]
  );

  const totalFeatured = featuredServers.length + featuredAgents.length +
    featuredSkills.length + featuredVirtual.length +
    featuredExtServers.length + featuredExtAgents.length;

  const handleSemanticSearch = useCallback(() => {
    if (searchTerm.trim().length >= 2) {
      setCommittedQuery(searchTerm.trim());
    }
  }, [searchTerm]);

  const handleClearSearch = useCallback(() => {
    setSearchTerm('');
    setCommittedQuery('');
  }, []);

  return (
    <div className="flex flex-col h-full">
      {/* Header: title + search bar - always at top */}
      <div className="w-full max-w-3xl mx-auto px-4 pt-4 pb-2">
        <h1 className="text-lg font-bold text-center mb-3 text-gray-800 dark:text-gray-100">
          Discover MCP Servers, Agents & Skills
        </h1>

        {/* Search Input */}
        <div className="relative">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <MagnifyingGlassIcon className="h-4 w-4 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="Search servers, agents, skills, or tools..."
            className="input pl-10 pr-9 w-full py-2 text-sm rounded-lg
              border border-gray-200 dark:border-gray-600
              focus:border-indigo-500 dark:focus:border-indigo-400
              shadow-sm hover:shadow-md transition-shadow"
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              if (committedQuery) {
                setCommittedQuery('');
              }
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleSemanticSearch();
              }
            }}
          />
          {searchTerm && (
            <button
              type="button"
              onClick={handleClearSearch}
              className="absolute inset-y-0 right-0 flex items-center pr-3
                text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            >
              <XMarkIcon className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Summary counts + hint */}
        {!isSemanticActive && (
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-1.5 text-center italic">
            {_buildSummaryText(
              { servers: totalServers, virtual: totalVirtual, agents: totalAgents, skills: totalSkills, external: totalExternal },
              { servers: matchedServers, virtual: matchedVirtual, agents: matchedAgents, skills: matchedSkills, external: matchedExternal },
              searchTerm.length > 0
            )}
            {searchTerm && (
              <span className="text-gray-600 dark:text-gray-600">
                {' '}&middot; press Enter for semantic search
              </span>
            )}
          </p>
        )}
      </div>

      {/* Content Area */}
      {isSemanticActive ? (
        /* Semantic Search Results */
        <div className="px-4 mt-2">
          <SemanticSearchResults
            query={committedQuery}
            loading={searchLoading}
            error={searchError}
            servers={searchResults?.servers || []}
            tools={searchResults?.tools || []}
            agents={searchResults?.agents || []}
            skills={searchResults?.skills || []}
            virtualServers={searchResults?.virtual_servers || []}
          />
        </div>
      ) : (
        /* Featured List Rows */
        <div className="relative flex-1 min-h-0">
        <div className="w-full max-w-5xl mx-auto px-4 mt-2 h-full overflow-y-auto discover-scroll">
          {loading ? (
            <div className="text-center text-gray-500 dark:text-gray-400 py-8">
              Loading featured items...
            </div>
          ) : totalFeatured === 0 ? (
            <div className="text-center text-gray-500 dark:text-gray-400 py-8">
              {searchTerm
                ? `No items matching "${searchTerm}"`
                : 'No items registered yet. Register your first MCP server, agent, or skill!'}
            </div>
          ) : (
            <div className="space-y-4">
              {/* MCP Servers section */}
              {featuredServers.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    MCP Servers
                    {matchedServers > featuredServers.length && (
                      <span className="ml-1.5 font-normal normal-case tracking-normal text-gray-500/70">
                        (showing {featuredServers.length} of {matchedServers})
                      </span>
                    )}
                  </h2>
                  {featuredServers.map(server => (
                    <DiscoverListRow
                      key={server.path}
                      type="server"
                      item={server}
                      onToggle={onServerToggle}
                      onEdit={onServerEdit}
                      onDelete={onServerDelete}
                      onShowToast={onShowToast}
                      authToken={authToken}
                    />
                  ))}
                </div>
              )}

              {/* Virtual MCP Servers section */}
              {featuredVirtual.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    Virtual MCP Servers
                    {matchedVirtual > featuredVirtual.length && (
                      <span className="ml-1.5 font-normal normal-case tracking-normal text-gray-500/70">
                        (showing {featuredVirtual.length} of {matchedVirtual})
                      </span>
                    )}
                  </h2>
                  {featuredVirtual.map(vs => (
                    <DiscoverListRow
                      key={vs.path}
                      type="virtual"
                      item={vs}
                      onToggle={onVirtualServerToggle}
                      onEdit={onVirtualServerEdit}
                      onDelete={onVirtualServerDelete}
                      onShowToast={onShowToast}
                      authToken={authToken}
                    />
                  ))}
                </div>
              )}

              {/* Agents section */}
              {featuredAgents.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    Agents
                    {matchedAgents > featuredAgents.length && (
                      <span className="ml-1.5 font-normal normal-case tracking-normal text-gray-500/70">
                        (showing {featuredAgents.length} of {matchedAgents})
                      </span>
                    )}
                  </h2>
                  {featuredAgents.map(agent => (
                    <DiscoverListRow
                      key={agent.path}
                      type="agent"
                      item={agent}
                      onToggle={onAgentToggle}
                      onEdit={onAgentEdit}
                      onDelete={onAgentDelete}
                      onShowToast={onShowToast}
                      authToken={authToken}
                    />
                  ))}
                </div>
              )}

              {/* Skills section */}
              {featuredSkills.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    Skills
                    {matchedSkills > featuredSkills.length && (
                      <span className="ml-1.5 font-normal normal-case tracking-normal text-gray-500/70">
                        (showing {featuredSkills.length} of {matchedSkills})
                      </span>
                    )}
                  </h2>
                  {featuredSkills.map(skill => (
                    <DiscoverListRow
                      key={skill.path}
                      type="skill"
                      item={skill}
                      onToggle={onSkillToggle}
                      onEdit={onSkillEdit}
                      onDelete={onSkillDelete}
                      onShowToast={onShowToast}
                      authToken={authToken}
                    />
                  ))}
                </div>
              )}

              {/* External Servers section */}
              {featuredExtServers.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    External Registry Servers
                    {matchedExtServers > featuredExtServers.length && (
                      <span className="ml-1.5 font-normal normal-case tracking-normal text-gray-500/70">
                        (showing {featuredExtServers.length} of {matchedExtServers})
                      </span>
                    )}
                  </h2>
                  {featuredExtServers.map(server => (
                    <DiscoverListRow
                      key={server.path}
                      type="server"
                      item={server}
                      onToggle={onServerToggle}
                      onEdit={onServerEdit}
                      onDelete={onServerDelete}
                      onShowToast={onShowToast}
                      authToken={authToken}
                    />
                  ))}
                </div>
              )}

              {/* External Agents section */}
              {featuredExtAgents.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    External Registry Agents
                    {matchedExtAgents > featuredExtAgents.length && (
                      <span className="ml-1.5 font-normal normal-case tracking-normal text-gray-500/70">
                        (showing {featuredExtAgents.length} of {matchedExtAgents})
                      </span>
                    )}
                  </h2>
                  {featuredExtAgents.map(agent => (
                    <DiscoverListRow
                      key={agent.path}
                      type="agent"
                      item={agent}
                      onToggle={onAgentToggle}
                      onEdit={onAgentEdit}
                      onDelete={onAgentDelete}
                      onShowToast={onShowToast}
                      authToken={authToken}
                    />
                  ))}
                </div>
              )}

              {/* Bottom padding so fade gradient doesn't cover last row */}
              <div className="h-8" />
            </div>
          )}
        </div>
        {/* Fade gradient at bottom to hint more content */}
        <div className="absolute bottom-0 left-0 right-0 h-12
          bg-gradient-to-t from-gray-900/80 to-transparent
          pointer-events-none" />
        </div>
      )}
    </div>
  );
};

export default DiscoverTab;
