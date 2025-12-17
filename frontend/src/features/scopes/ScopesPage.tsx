import { useState, useMemo } from 'react';
import {
  ShieldCheckIcon,
  ServerIcon,
  MagnifyingGlassIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ExclamationTriangleIcon,
  WrenchScrewdriverIcon,
  UserGroupIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, Spinner, EmptyState, Input } from '@/components/ui';
import {
  useScopeCatalog,
  filterScopes,
  getServersDisplay,
  getMethodsDisplay,
  getToolsDisplay,
} from './hooks';
import { ScopeExplainerCard } from './ScopeExplainerCard';
import { CreateScopeModal } from './CreateScopeModal';
import { EditScopeModal } from './EditScopeModal';
import { useAuth } from '@/contexts/AuthContext';
import { useToast } from '@/components/ui/Toast';
import { useDeleteScope } from './hooks';
import { TypeToConfirmDialog } from '@/components/ui/TypeToConfirmDialog';
import { normalizeError } from '@/lib/errors';
import type { ScopeInfo, ScopeDefinition } from '@/api/types';

// ============================================================================
// Scope Card Component
// ============================================================================

interface ScopeCardProps {
  scopeInfo: ScopeInfo;
  scopeDefinition?: ScopeDefinition;
  isExpanded: boolean;
  onToggle: () => void;
  canEdit?: boolean;
  onEdit?: () => void;
  canDelete?: boolean;
  onDelete?: () => void;
}

function ScopeCard({
  scopeInfo,
  scopeDefinition,
  isExpanded,
  onToggle,
  canEdit,
  onEdit,
  canDelete,
  onDelete,
}: ScopeCardProps) {
  const isWildcard =
    scopeInfo.all_servers && scopeInfo.all_methods && scopeInfo.all_tools;

  return (
    <Card className="overflow-hidden">
      {/* Header - Clickable */}
      <div className="flex items-stretch">
        <button
          type="button"
          onClick={onToggle}
          className="flex-1 text-left px-4 py-3 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700/50 transition-colors"
        >
          <div className="flex items-center space-x-3 min-w-0">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
              <ShieldCheckIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div className="min-w-0">
              <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
                {scopeInfo.name}
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
                {getServersDisplay(scopeInfo)}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            {isWildcard && (
              <Badge variant="warning" size="sm">
                Unrestricted
              </Badge>
            )}
            {scopeInfo.agent_actions.length > 0 && (
              <Badge variant="info" size="sm">
                {scopeInfo.agent_actions.length} actions
              </Badge>
            )}
            {isExpanded ? (
              <ChevronDownIcon className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronRightIcon className="w-5 h-5 text-gray-400" />
            )}
          </div>
        </button>

        {(canEdit || canDelete) && (
          <div className="flex items-center pr-4 gap-2">
            {canEdit && onEdit && (
              <Button variant="secondary" size="sm" onClick={onEdit}>
                Edit
              </Button>
            )}
            {canDelete && onDelete && (
              <Button variant="danger" size="sm" onClick={onDelete}>
                Delete
              </Button>
            )}
          </div>
        )}
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-4 space-y-4">
          {/* Quick Summary */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Servers */}
            <div className="flex items-start space-x-2">
              <ServerIcon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  Servers
                </p>
                <p className="text-sm text-gray-900 dark:text-gray-100">
                  {scopeInfo.all_servers ? (
                    <span className="text-amber-600 dark:text-amber-400">
                      All servers (*)
                    </span>
                  ) : scopeInfo.servers.length > 0 ? (
                    scopeInfo.servers.join(', ')
                  ) : (
                    <span className="text-gray-400">None</span>
                  )}
                </p>
              </div>
            </div>

            {/* Methods */}
            <div className="flex items-start space-x-2">
              <AdjustmentsHorizontalIcon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  Methods
                </p>
                <p className="text-sm text-gray-900 dark:text-gray-100">
                  {scopeInfo.all_methods ? (
                    <span className="text-amber-600 dark:text-amber-400">
                      All methods
                    </span>
                  ) : (
                    getMethodsDisplay(scopeInfo)
                  )}
                </p>
              </div>
            </div>

            {/* Tools */}
            <div className="flex items-start space-x-2">
              <WrenchScrewdriverIcon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400">
                  Tools
                </p>
                <p className="text-sm text-gray-900 dark:text-gray-100">
                  {scopeInfo.all_tools ? (
                    <span className="text-amber-600 dark:text-amber-400">
                      All tools (*)
                    </span>
                  ) : (
                    getToolsDisplay(scopeInfo)
                  )}
                </p>
              </div>
            </div>
          </div>

          {/* Agent Actions */}
          {scopeInfo.agent_actions.length > 0 && (
            <div className="flex items-start space-x-2">
              <UserGroupIcon className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                  Agent Actions
                </p>
                <div className="flex flex-wrap gap-1">
                  {scopeInfo.agent_actions.map((action) => (
                    <Badge key={action} variant="neutral" size="sm">
                      {action}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Full Explainer */}
          {scopeDefinition && (
            <div className="pt-2 border-t border-gray-200 dark:border-gray-700">
              <ScopeExplainerCard scope={scopeDefinition} />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function ScopesPage() {
  const { catalog, scopes, isLoading, isError, isAvailable, groupMappings } =
    useScopeCatalog();
  const { user } = useAuth();
  const toast = useToast();
  const { deleteScope, isDeleting } = useDeleteScope();
  const canManageScopes =
    !!user?.is_admin || !!user?.groups?.includes('enforceai-admin');
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedScopes, setExpandedScopes] = useState<Set<string>>(new Set());
  const [filterServer, setFilterServer] = useState('');
  const [showCreateScope, setShowCreateScope] = useState(false);
  const [editingScopeName, setEditingScopeName] = useState<string | null>(null);
  const [deletingScopeName, setDeletingScopeName] = useState<string | null>(null);

  // Get unique servers for filter dropdown
  const allServers = useMemo(() => {
    const servers = new Set<string>();
    scopes.forEach((scope) => {
      scope.servers.forEach((s) => servers.add(s));
    });
    return Array.from(servers).sort();
  }, [scopes]);

  // Filter scopes
  const filteredScopes = useMemo(() => {
    let filtered = filterScopes(scopes, searchQuery);

    if (filterServer) {
      filtered = filtered.filter(
        (scope) => scope.all_servers || scope.servers.includes(filterServer)
      );
    }

    return filtered.sort((a, b) => a.name.localeCompare(b.name));
  }, [scopes, searchQuery, filterServer]);

  // Toggle scope expansion
  const toggleScope = (scopeName: string) => {
    setExpandedScopes((prev) => {
      const next = new Set(prev);
      if (next.has(scopeName)) {
        next.delete(scopeName);
      } else {
        next.add(scopeName);
      }
      return next;
    });
  };

  // Expand/collapse all
  const expandAll = () => {
    setExpandedScopes(new Set(filteredScopes.map((s) => s.name)));
  };

  const collapseAll = () => {
    setExpandedScopes(new Set());
  };

  // Loading state
  if (isLoading) {
    return (
      <PageContent>
        <PageHeader
          title="Scopes and Policy"
          description="View the scope catalog and policy definitions"
          breadcrumbs={[{ name: 'Scopes' }]}
        />
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      </PageContent>
    );
  }

  // Error or not available state
  if (isError || !isAvailable) {
    return (
      <PageContent>
        <PageHeader
          title="Scopes and Policy"
          description="View the scope catalog and policy definitions"
          breadcrumbs={[{ name: 'Scopes' }]}
        />
        <EmptyState
          icon={<ExclamationTriangleIcon className="w-12 h-12" />}
          title="Scope Catalog Unavailable"
          description="The scope catalog could not be loaded. This may be because the API endpoint is not configured or the catalog file is not available."
        />
        <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
          <h4 className="text-sm font-medium text-blue-800 dark:text-blue-300 mb-2">
            About Scopes
          </h4>
          <p className="text-sm text-blue-700 dark:text-blue-400">
            Scopes define what servers, methods, and tools an agent can access.
            They are configured in the <code>scopes.yml</code> file on the
            server. Contact your administrator if you need access to specific
            scopes.
          </p>
        </div>
      </PageContent>
    );
  }

  return (
    <PageContent>
      <PageHeader
        title="Scopes and Policy"
        description="View the scope catalog and policy definitions"
        breadcrumbs={[{ name: 'Scopes' }]}
        actions={
          canManageScopes ? (
            <Button
              variant="primary"
              size="sm"
              onClick={() => setShowCreateScope(true)}
            >
              Create Scope
            </Button>
          ) : null
        }
      />

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
              <ShieldCheckIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {scopes.length}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Total Scopes
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
              <ServerIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {allServers.length}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Servers Referenced
              </p>
            </div>
          </div>
        </Card>
        <Card className="p-4">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
              <UserGroupIcon className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                {Object.keys(groupMappings).length}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Group Mappings
              </p>
            </div>
          </div>
        </Card>
      </div>

      {/* Search and Filters */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <Input
              type="text"
              placeholder="Search scopes, servers, or tools..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          {/* Server Filter */}
          <div className="sm:w-48">
            <select
              value={filterServer}
              onChange={(e) => setFilterServer(e.target.value)}
              className="block w-full rounded-md border-gray-300 dark:border-gray-600 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm dark:bg-gray-700 dark:text-gray-100"
            >
              <option value="">All servers</option>
              {allServers.map((server) => (
                <option key={server} value={server}>
                  {server}
                </option>
              ))}
            </select>
          </div>

          {/* Expand/Collapse */}
          <div className="flex gap-2">
            <Button variant="secondary" size="sm" onClick={expandAll}>
              Expand All
            </Button>
            <Button variant="secondary" size="sm" onClick={collapseAll}>
              Collapse All
            </Button>
          </div>
        </div>
      </div>

      {/* Results Count */}
      <div className="mb-4 text-sm text-gray-500 dark:text-gray-400">
        Showing {filteredScopes.length} of {scopes.length} scopes
        {searchQuery && ` matching "${searchQuery}"`}
        {filterServer && ` with access to "${filterServer}"`}
      </div>

      {/* Scope List */}
      {filteredScopes.length === 0 ? (
        <EmptyState
          icon={<MagnifyingGlassIcon className="w-8 h-8" />}
          title="No matching scopes"
          description="Try adjusting your search query or filter."
        />
      ) : (
        <div className="space-y-3">
          {filteredScopes.map((scopeInfo) => (
            <ScopeCard
              key={scopeInfo.name}
              scopeInfo={scopeInfo}
              scopeDefinition={catalog?.scopes[scopeInfo.name]}
              isExpanded={expandedScopes.has(scopeInfo.name)}
              onToggle={() => toggleScope(scopeInfo.name)}
              canEdit={canManageScopes}
              onEdit={() => setEditingScopeName(scopeInfo.name)}
              canDelete={canManageScopes}
              onDelete={() => setDeletingScopeName(scopeInfo.name)}
            />
          ))}
        </div>
      )}

      {/* Group Mappings Section */}
      {Object.keys(groupMappings).length > 0 && (
        <div className="mt-8">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
            Group Mappings
          </h2>
          <Card>
            <div className="divide-y divide-gray-200 dark:divide-gray-700">
              {Object.entries(groupMappings).map(([group, mappedScopes]) => (
                <div key={group} className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <UserGroupIcon className="w-4 h-4 text-gray-400" />
                      <span className="font-medium text-gray-900 dark:text-gray-100">
                        {group}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {mappedScopes.map((scope) => (
                        <Badge
                          key={scope}
                          variant="info"
                          size="sm"
                          className="cursor-pointer hover:opacity-80"
                          onClick={() => {
                            setSearchQuery(scope);
                            setExpandedScopes(new Set([scope]));
                          }}
                        >
                          {scope}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      <CreateScopeModal
        open={showCreateScope}
        onClose={() => setShowCreateScope(false)}
        catalogEtag={catalog?.etag}
        onSuccess={(scopeName) => {
          setShowCreateScope(false);
          setSearchQuery(scopeName);
          setExpandedScopes(new Set([scopeName]));
        }}
      />

      {editingScopeName && catalog?.scopes[editingScopeName] && (
        <EditScopeModal
          open={!!editingScopeName}
          onClose={() => setEditingScopeName(null)}
          scopeName={editingScopeName}
          scope={catalog.scopes[editingScopeName]}
          catalogEtag={catalog.etag}
          onSuccess={(scopeName) => {
            setEditingScopeName(null);
            setSearchQuery(scopeName);
            setExpandedScopes(new Set([scopeName]));
          }}
        />
      )}

      {deletingScopeName && (
        <TypeToConfirmDialog
          open={!!deletingScopeName}
          onClose={() => setDeletingScopeName(null)}
          title="Delete Scope"
          message="This permanently removes the scope from the policy catalog. Deletion will be blocked if the scope is referenced by group mappings."
          confirmValue={deletingScopeName}
          confirmLabel="Delete Scope"
          loading={isDeleting}
          onConfirm={async () => {
            if (!catalog?.etag) {
              toast.error('Delete failed', 'Catalog ETag missing. Refresh and try again.');
              return;
            }

            try {
              await deleteScope(deletingScopeName, catalog.etag);
              toast.success('Scope deleted', `Deleted ${deletingScopeName}`);

              setDeletingScopeName(null);
              if (searchQuery === deletingScopeName) {
                setSearchQuery('');
              }
            } catch (err) {
              const normalized = normalizeError(err);
              if (normalized.status === 412) {
                toast.error('Delete failed', 'Catalog changed; refresh and retry.');
              } else {
                toast.error('Delete failed', normalized.message);
              }
            }
          }}
        />
      )}
    </PageContent>
  );
}
