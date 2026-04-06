import React, { useState, useMemo, useCallback } from 'react';
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useSemanticSearch } from '../hooks/useSemanticSearch';
import SemanticSearchResults from './SemanticSearchResults';
import DiscoverCard from './DiscoverCard';
import type { Server } from './ServerCard';
import type { Skill } from '../types/skill';


// Path for the built-in AI Registry Tools server
const AI_REGISTRY_TOOLS_PATH = '/airegistry-tools/';

// Maximum featured items per category
const MAX_SERVERS = 4;
const MAX_AGENTS = 4;
const MAX_SKILLS = 4;


interface DiscoverTabProps {
  servers: Server[];
  agents: Server[];
  skills: Skill[];
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
 * Get featured items for the Discover landing page.
 * AI Registry Tools always first among servers if it exists.
 * Returns sorted, enabled items up to the max per category.
 */
function _getFeaturedItems(
  servers: Server[],
  agents: Server[],
  skills: Skill[],
  keywordFilter: string
) {
  // Filter enabled items
  const enabledServers = servers.filter(s => s.enabled);
  const enabledAgents = agents.filter(a => a.enabled);
  const enabledSkills = skills.filter(s => s.is_enabled);

  // Apply keyword filter if present
  const filterFn = keywordFilter.length > 0;
  const filteredServers = filterFn
    ? enabledServers.filter(s => _matchesKeyword(s, keywordFilter))
    : enabledServers;
  const filteredAgents = filterFn
    ? enabledAgents.filter(a => _matchesKeyword(a, keywordFilter))
    : enabledAgents;
  const filteredSkills = filterFn
    ? enabledSkills.filter(s => _matchesKeyword({
        name: s.name,
        description: s.description,
        path: s.path,
        tags: s.tags,
      }, keywordFilter))
    : enabledSkills;

  // Sort and pick top items
  // AI Registry Tools goes first if it's in the filtered list
  const aiRegistryTools = filteredServers.find(s => s.path === AI_REGISTRY_TOOLS_PATH);
  const otherServers = filteredServers.filter(s => s.path !== AI_REGISTRY_TOOLS_PATH);
  const sortedOther = _sortServersByRating(otherServers);

  const featuredServers: Server[] = [];
  if (aiRegistryTools) {
    featuredServers.push(aiRegistryTools);
  }
  featuredServers.push(...sortedOther.slice(0, MAX_SERVERS - featuredServers.length));

  const featuredAgents = _sortServersByRating(filteredAgents).slice(0, MAX_AGENTS);
  const featuredSkills = _sortSkillsByStars(filteredSkills).slice(0, MAX_SKILLS);

  return { featuredServers, featuredAgents, featuredSkills };
}


const DiscoverTab: React.FC<DiscoverTabProps> = ({
  servers,
  agents,
  skills,
  loading,
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
  const { featuredServers, featuredAgents, featuredSkills } = useMemo(
    () => _getFeaturedItems(servers, agents, skills, isSemanticActive ? '' : searchTerm),
    [servers, agents, skills, searchTerm, isSemanticActive]
  );

  const totalFeatured = featuredServers.length + featuredAgents.length + featuredSkills.length;

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
            onChange={(e) => setSearchTerm(e.target.value)}
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

        {/* Hint text */}
        {searchTerm && !isSemanticActive && (
          <p className="text-xs text-gray-500 mt-1 text-center">
            Press Enter for semantic search
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
        /* Featured Cards Grid */
        <div className="w-full max-w-5xl mx-auto px-4 mt-2 overflow-y-auto">
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
              {/* Servers section */}
              {featuredServers.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    MCP Servers
                  </h2>
                  <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {featuredServers.map(server => (
                      <DiscoverCard
                        key={server.path}
                        type="server"
                        item={server}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Agents section */}
              {featuredAgents.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    Agents
                  </h2>
                  <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {featuredAgents.map(agent => (
                      <DiscoverCard
                        key={agent.path}
                        type="agent"
                        item={agent}
                      />
                    ))}
                  </div>
                </div>
              )}

              {/* Skills section */}
              {featuredSkills.length > 0 && (
                <div>
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-2">
                    Skills
                  </h2>
                  <div className="grid gap-2 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                    {featuredSkills.map(skill => (
                      <DiscoverCard
                        key={skill.path}
                        type="skill"
                        item={skill}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DiscoverTab;
