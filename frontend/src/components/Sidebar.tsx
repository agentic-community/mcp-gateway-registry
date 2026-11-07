import React, { Fragment, useEffect, useState } from 'react';
import clsx from 'clsx';
import { Dialog, Transition } from '@headlessui/react';
import { Link, useLocation } from 'react-router-dom';
import {
  XMarkIcon,
  FunnelIcon,
  ChartBarIcon,
  KeyIcon,
  ArrowLeftIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ClipboardIcon,
  CheckIcon,
  ArrowDownTrayIcon,
  TagIcon,
  LinkIcon,
  ShieldCheckIcon,
  WrenchScrewdriverIcon,
  EyeIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { isEgressAuthEnabled } from '../utils/egressAuth';
import { DESKTOP_QUERY, useMediaQuery } from '../hooks/useMediaQuery';

interface SidebarProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
  stats: {
    total: number;
    enabled: number;
    disabled: number;
    withIssues: number;
  };
  activeFilter: string;
  setActiveFilter: (filter: string) => void;
  availableTags: string[];
  selectedTags: string[];
  onTagSelect: (tag: string) => void;
}

const NAV_ITEM_CLASS =
  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors';
const NAV_ITEM_IDLE = 'text-fg-muted hover:bg-surface-hover hover:text-fg';
const NAV_ITEM_ACTIVE = 'bg-accent-soft text-accent-soft-fg';


interface AccessUser {
  is_admin?: boolean;
  can_modify_servers?: boolean;
  auth_method?: string;
  provider?: string;
}


/** Access level of the signed-in user, shown under their name. */
const AccessLevel: React.FC<{ user: AccessUser }> = ({ user }) => {
  const level = user.is_admin
    ? { label: 'Admin access', icon: ShieldCheckIcon, tone: 'text-success-600 dark:text-success-400' }
    : user.can_modify_servers
      ? { label: 'Modify access', icon: WrenchScrewdriverIcon, tone: 'text-info-600 dark:text-info-400' }
      : { label: 'Read-only access', icon: EyeIcon, tone: 'text-fg-muted' };
  const Icon = level.icon;
  return (
    <div className={clsx('flex items-center gap-1.5 text-xs font-medium', level.tone)}>
      <Icon className="h-3.5 w-3.5" />
      <span>{level.label}</span>
      {user.auth_method === 'oauth2' && user.provider && (
        <span className="font-normal text-fg-subtle">({user.provider})</span>
      )}
    </div>
  );
};


const Sidebar: React.FC<SidebarProps> = ({ sidebarOpen, setSidebarOpen, stats, activeFilter, setActiveFilter, availableTags, selectedTags, onTagSelect }) => {
  // const { stats, activeFilter, setActiveFilter } = useServerStats();
  const { user } = useAuth();
  const location = useLocation();
  const isDesktop = useMediaQuery(DESKTOP_QUERY);
  const closeOnMobile = () => {
    if (!isDesktop) setSidebarOpen(false);
  };
  const [showScopes, setShowScopes] = useState(false);
  const [tagDropdownOpen, setTagDropdownOpen] = useState(false);
  const [tagSearch, setTagSearch] = useState('');
  const [tagHighlightIndex, setTagHighlightIndex] = useState(0);
  const tagDropdownRef = React.useRef<HTMLDivElement>(null);
  const [showTokenModal, setShowTokenModal] = useState(false);
  const [tokenData, setTokenData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string>('');
  const [egressEnabled, setEgressEnabled] = useState(false);

  useEffect(() => {
    let active = true;
    void isEgressAuthEnabled().then(enabled => {
      if (active) setEgressEnabled(enabled);
    });
    return () => {
      active = false;
    };
  }, []);

  const filters = [
    { key: 'all', label: 'All Services', count: 'total' },
    { key: 'enabled', label: 'Enabled', count: 'enabled' },
    { key: 'disabled', label: 'Disabled', count: 'disabled' },
    { key: 'unhealthy', label: 'With Issues', count: 'withIssues' },
  ];

  const isTokenPage = location.pathname === '/generate-token';

  // Close tag dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (tagDropdownRef.current && !tagDropdownRef.current.contains(e.target as Node)) {
        setTagDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Scope descriptions mapping
  const getScopeDescription = (scope: string) => {
    const scopeMappings: { [key: string]: string } = {
      'mcp-servers-restricted/read': 'Read access to restricted MCP servers',
      'mcp-servers/read': 'Read access to all MCP servers',
      'mcp-servers/write': 'Write access to MCP servers',
      'mcp-registry-user': 'Basic registry user permissions',
      'mcp-registry-admin': 'Full registry administration access',
      'health-check': 'Health check and monitoring access',
      'token-generation': 'Ability to generate access tokens',
      'server-management': 'Manage server configurations',
    };
    return scopeMappings[scope] || 'Custom permission scope';
  };

const fetchAdminTokens = async () => {
  setLoading(true);
  setError('');
  try {
    // Omit expires_in_hours so the server applies the configured default
    // lifetime (MCP_TOKEN_DEFAULT_TTL_HOURS). Hardcoding a value here breaks
    // when an operator lowers the max below it. Issue #1477.
    const requestData = {
      description: 'Generated via sidebar',
    };

    const response = await axios.post('/api/tokens/generate', requestData, {
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (response.data.success) {
      setTokenData(response.data);
      setShowTokenModal(true);
    }
  } catch (err: any) {
    setError(err.response?.data?.detail || 'Failed to generate token');
  } finally {
    setLoading(false);
  }
};


  const handleCopyTokens = async () => {
    if (!tokenData) return;

    const formattedData = JSON.stringify(tokenData, null, 2);
    try {
      await navigator.clipboard.writeText(formattedData);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleDownloadTokens = () => {
    if (!tokenData) return;

    const formattedData = JSON.stringify(tokenData, null, 2);
    const blob = new Blob([formattedData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mcp-registry-api-tokens-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const scopesToggle = user && !user.is_admin && user.scopes && user.scopes.length > 0 && (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setShowScopes(!showScopes)}
        aria-expanded={showScopes}
        className="flex w-full items-center justify-between py-1 text-xs text-fg-muted transition-colors hover:text-fg"
      >
        <span>Scopes ({user.scopes.length})</span>
        {showScopes ? <ChevronUpIcon className="h-3 w-3" /> : <ChevronDownIcon className="h-3 w-3" />}
      </button>

      {showScopes && (
        <div className="mt-2 max-h-32 space-y-2 overflow-y-auto">
          {user.scopes.map((scope) => (
            <div key={scope} className="rounded-md border border-line bg-surface p-2 text-xs">
              <div className="font-medium text-fg">{scope}</div>
              <div className="mt-0.5 text-fg-muted">{getScopeDescription(scope)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const userCard = (showTokenButton: boolean) =>
    user && (
      <div className="rounded-xl border border-line bg-surface-muted p-3">
        <div className="mb-1 truncate text-sm font-semibold text-fg" title={user.username}>
          {user.username}
        </div>
        <AccessLevel user={user} />

        {showTokenButton && (
          <div className="mt-3">
            <button
              type="button"
              onClick={fetchAdminTokens}
              disabled={loading}
              className="btn-secondary w-full py-1.5 text-xs"
            >
              {loading ? (
                <>
                  <div className="h-3 w-3 animate-spin rounded-full border-2 border-line border-t-accent" />
                  <span>Loading...</span>
                </>
              ) : (
                <>
                  <KeyIcon className="h-3.5 w-3.5" />
                  <span>Get JWT Token</span>
                </>
              )}
            </button>
            {error && <p className="mt-1 text-xs text-danger-soft-fg">{error}</p>}
          </div>
        )}

        {scopesToggle}
      </div>
    );

  const sectionHeading = (icon: React.ReactNode, label: string) => (
    <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-fg-subtle">
      {icon}
      <h3>{label}</h3>
    </div>
  );

  const sidebarContent = (
    <div className="flex h-full flex-col">
      {isTokenPage ? (
        /* Token Page - Show navigation and user info */
        <div className="flex-1 space-y-6 p-4 md:p-5">
          <nav className="space-y-1" aria-label="Token pages">
            <Link to="/" className={clsx(NAV_ITEM_CLASS, NAV_ITEM_IDLE)} onClick={closeOnMobile}>
              <ArrowLeftIcon className="h-4 w-4" />
              <span>Back to Dashboard</span>
            </Link>
            <Link to="/generate-token" className={clsx(NAV_ITEM_CLASS, NAV_ITEM_ACTIVE)} aria-current="page">
              <KeyIcon className="h-4 w-4" />
              <span>Generate Token</span>
            </Link>
            {egressEnabled && (
              <Link to="/connected-accounts" className={clsx(NAV_ITEM_CLASS, NAV_ITEM_IDLE)} onClick={closeOnMobile}>
                <LinkIcon className="h-4 w-4" />
                <span>Connected Accounts</span>
              </Link>
            )}
          </nav>

          {userCard(false)}

          <div className="rounded-xl border border-dashed border-line p-4 text-center">
            <KeyIcon className="mx-auto mb-3 h-10 w-10 text-accent-text" />
            <h3 className="mb-1 text-sm font-semibold text-fg">Token Generation</h3>
            <p className="mb-3 text-xs text-fg-muted">
              Create personal access tokens for programmatic access to MCP servers
            </p>
            <ul className="space-y-1 text-left text-xs text-fg-subtle">
              <li>Tokens inherit your current permissions</li>
              <li>Configure expiration time and scopes</li>
              <li>Use tokens for programmatic access</li>
            </ul>
          </div>
        </div>
      ) : (
        /* Dashboard - Show user info, filters and stats */
        <>
          <div className="border-b border-line p-4 md:p-5">{userCard(true)}</div>

          {/* Filters Section */}
          <div className="flex-1 p-4 md:p-5">
            {sectionHeading(<FunnelIcon className="h-4 w-4" />, 'Filter Services')}

            <div className="space-y-1">
              {filters.map((filter) => {
                const active = activeFilter === filter.key;
                return (
                  <button
                    key={filter.key}
                    type="button"
                    onClick={() => setActiveFilter(filter.key)}
                    aria-pressed={active}
                    className={clsx(NAV_ITEM_CLASS, 'w-full justify-between', active ? NAV_ITEM_ACTIVE : NAV_ITEM_IDLE)}
                  >
                    <span>{filter.label}</span>
                    <span
                      className={clsx(
                        'rounded-full px-2 py-0.5 text-xs tabular-nums',
                        active ? 'bg-surface/60 text-accent-soft-fg' : 'bg-surface-hover text-fg-muted',
                      )}
                    >
                      {stats[filter.count as keyof typeof stats]}
                    </span>
                  </button>
                );
              })}

              {/* Deprecated toggle - separate from main filters */}
              <div className="mt-3 border-t border-line pt-3">
                <button
                  type="button"
                  onClick={() => setActiveFilter(activeFilter === 'deprecated' ? 'all' : 'deprecated')}
                  aria-pressed={activeFilter === 'deprecated'}
                  className={clsx(
                    NAV_ITEM_CLASS,
                    'w-full',
                    activeFilter === 'deprecated'
                      ? 'bg-warning-50 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300'
                      : NAV_ITEM_IDLE,
                  )}
                >
                  <span>{activeFilter === 'deprecated' ? 'Showing deprecated' : 'Also show deprecated'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Tags Section */}
          {availableTags.length > 0 && (
            <div className="border-t border-line p-4 md:p-5">
              <div className="flex items-center justify-between">
                {sectionHeading(<TagIcon className="h-4 w-4" />, 'Filter by Tag')}
                {selectedTags.length > 0 && (
                  <button
                    type="button"
                    onClick={() => selectedTags.forEach(t => onTagSelect(t))}
                    className="mb-3 text-xs font-medium text-accent-text hover:underline"
                  >
                    Clear all
                  </button>
                )}
              </div>

              {/* Selected tag chips */}
              {selectedTags.length > 0 && (
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {selectedTags.map((tag) => (
                    <span
                      key={tag}
                      className="inline-flex items-center gap-1 rounded-full bg-accent-soft px-2.5 py-1 text-xs font-medium text-accent-soft-fg"
                    >
                      {tag}
                      <button
                        type="button"
                        onClick={() => onTagSelect(tag)}
                        className="rounded-full hover:text-fg"
                        aria-label={`Remove tag ${tag}`}
                      >
                        <XMarkIcon className="h-3 w-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}

              {/* Tag dropdown */}
              {(() => {
                const filteredTags = availableTags.filter(tag =>
                  !selectedTags.includes(tag) &&
                  tag.toLowerCase().includes(tagSearch.toLowerCase())
                );
                return (
                  <div className="relative" ref={tagDropdownRef}>
                    <input
                      type="text"
                      placeholder="Search tags..."
                      value={tagSearch}
                      onChange={(e) => {
                        setTagSearch(e.target.value);
                        setTagHighlightIndex(0);
                        setTagDropdownOpen(true);
                      }}
                      onFocus={() => {
                        setTagDropdownOpen(true);
                        setTagHighlightIndex(0);
                      }}
                      onKeyDown={(e) => {
                        if (!tagDropdownOpen || filteredTags.length === 0) return;
                        if (e.key === 'ArrowDown') {
                          e.preventDefault();
                          setTagHighlightIndex(prev => Math.min(prev + 1, filteredTags.length - 1));
                        } else if (e.key === 'ArrowUp') {
                          e.preventDefault();
                          setTagHighlightIndex(prev => Math.max(prev - 1, 0));
                        } else if (e.key === 'Enter') {
                          e.preventDefault();
                          const tag = filteredTags[tagHighlightIndex];
                          if (tag) {
                            onTagSelect(tag);
                            setTagSearch('');
                            setTagHighlightIndex(0);
                            setTagDropdownOpen(false);
                          }
                        } else if (e.key === 'Escape') {
                          setTagDropdownOpen(false);
                        }
                      }}
                      aria-label="Search tags"
                      className="input py-1.5 text-xs"
                    />
                    {tagDropdownOpen && (
                      <div className="absolute z-50 mt-1 max-h-40 w-full overflow-y-auto rounded-lg border border-line bg-surface-overlay py-1 shadow-lg">
                        {filteredTags.map((tag, idx) => (
                          <button
                            key={tag}
                            type="button"
                            onClick={() => {
                              onTagSelect(tag);
                              setTagSearch('');
                              setTagHighlightIndex(0);
                              setTagDropdownOpen(false);
                            }}
                            onMouseEnter={() => setTagHighlightIndex(idx)}
                            className={clsx(
                              'w-full px-3 py-1.5 text-left text-xs transition-colors',
                              idx === tagHighlightIndex ? 'bg-accent-soft text-accent-soft-fg' : 'text-fg-muted',
                            )}
                          >
                            {tag}
                          </button>
                        ))}
                        {filteredTags.length === 0 && (
                          <div className="px-3 py-2 text-xs text-fg-subtle">
                            No matching tags
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}

          {/* Statistics Section */}
          <div className="border-t border-line p-4 md:p-5">
            {sectionHeading(<ChartBarIcon className="h-4 w-4" />, 'Statistics')}

            <dl className="grid grid-cols-2 gap-2">
              {[
                { label: 'Total', value: stats.total, tone: 'text-fg' },
                { label: 'Enabled', value: stats.enabled, tone: 'text-success-600 dark:text-success-400' },
                { label: 'Disabled', value: stats.disabled, tone: 'text-fg-muted' },
                { label: 'Issues', value: stats.withIssues, tone: 'text-danger-600 dark:text-danger-400' },
              ].map(({ label, value, tone }) => (
                <div key={label} className="rounded-xl border border-line bg-surface-muted px-3 py-2.5">
                  <dt className="text-xs text-fg-subtle">{label}</dt>
                  <dd className={clsx('text-xl font-semibold tabular-nums', tone)}>{value}</dd>
                </div>
              ))}
            </dl>
          </div>
        </>
      )}
    </div>
  );

  return (
    <>
      {/* Mobile sidebar only */}
      {!isDesktop && (
        <Transition.Root show={sidebarOpen} as={Fragment}>
          <Dialog as="div" className="relative z-50" onClose={setSidebarOpen}>
            <Transition.Child
              as={Fragment}
              enter="transition-opacity ease-linear duration-300"
              enterFrom="opacity-0"
              enterTo="opacity-100"
              leave="transition-opacity ease-linear duration-300"
              leaveFrom="opacity-100"
              leaveTo="opacity-0"
            >
              <div className="fixed inset-0 bg-gray-950/60 backdrop-blur-sm" />
            </Transition.Child>

            <div className="fixed inset-0 flex">
              <Transition.Child
                as={Fragment}
                enter="transition ease-in-out duration-300 transform"
                enterFrom="-translate-x-full"
                enterTo="translate-x-0"
                leave="transition ease-in-out duration-300 transform"
                leaveFrom="translate-x-0"
                leaveTo="-translate-x-full"
              >
                <Dialog.Panel className="relative mr-16 flex w-full max-w-xs flex-1">
                  <Transition.Child
                    as={Fragment}
                    enter="ease-in-out duration-300"
                    enterFrom="opacity-0"
                    enterTo="opacity-100"
                    leave="ease-in-out duration-300"
                    leaveFrom="opacity-100"
                    leaveTo="opacity-0"
                  >
                    <div className="absolute left-full top-0 flex w-16 justify-center pt-5">
                      <button
                        type="button"
                        className="-m-2.5 p-2.5"
                        onClick={() => setSidebarOpen(false)}
                        aria-label="Close sidebar"
                      >
                        <XMarkIcon className="h-6 w-6 text-white" />
                      </button>
                    </div>
                  </Transition.Child>

                  <div className="flex grow flex-col overflow-y-auto border-r border-line bg-surface">
                    {sidebarContent}
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </Dialog>
        </Transition.Root>
      )}

      {/* Desktop sidebar only */}
      {isDesktop && (
        <Transition show={sidebarOpen} as={Fragment}>
          <Transition.Child
            as={Fragment}
            enter="transition ease-in-out duration-300 transform"
            enterFrom="-translate-x-full"
            enterTo="translate-x-0"
            leave="transition ease-in-out duration-300 transform"
            leaveFrom="translate-x-0"
            leaveTo="-translate-x-full"
          >
            <aside className="fixed bottom-0 left-0 top-16 z-40 w-64 overflow-y-auto border-r border-line bg-surface lg:w-72 xl:w-80">
              {sidebarContent}
            </aside>
          </Transition.Child>
        </Transition>
      )}

      {/* Token Modal */}
      <Transition appear show={showTokenModal} as={Fragment}>
        <Dialog as="div" className="relative z-50" onClose={() => setShowTokenModal(false)}>
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-gray-950/50 backdrop-blur-sm" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4 text-center">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-3xl transform overflow-hidden rounded-2xl border border-line bg-surface-overlay p-6 text-left align-middle shadow-xl transition-all">
                  <Dialog.Title
                    as="h3"
                    className="mb-4 text-lg font-semibold leading-6 text-fg"
                  >
                    JWT Access Token
                  </Dialog.Title>

                  {tokenData && (
                    <div className="space-y-4">
                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={handleCopyTokens}
                          className="btn-primary"
                        >
                          {copied ? (
                            <>
                              <CheckIcon className="h-4 w-4" />
                              <span>Copied!</span>
                            </>
                          ) : (
                            <>
                              <ClipboardIcon className="h-4 w-4" />
                              <span>Copy JSON</span>
                            </>
                          )}
                        </button>
                        <button
                          type="button"
                          onClick={handleDownloadTokens}
                          className="btn-secondary"
                        >
                          <ArrowDownTrayIcon className="h-4 w-4" />
                          <span>Download JSON</span>
                        </button>
                      </div>

                      {/* Human-readable lifetime, derived from the response's
                          expires_in (seconds). The lifetime is operator-
                          configurable (MCP_TOKEN_DEFAULT_TTL_HOURS), so read it
                          from the response rather than hardcoding a value. */}
                      {typeof tokenData?.tokens?.expires_in === 'number' && (
                        <p className="text-sm text-fg-muted">
                          This token expires in{' '}
                          <span className="font-medium">
                            {(() => {
                              const secs = tokenData.tokens.expires_in;
                              const hrs = secs / 3600;
                              const rounded = Math.round(hrs * 100) / 100;
                              return `${rounded} hour${rounded === 1 ? '' : 's'}`;
                            })()}
                          </span>{' '}
                          ({tokenData.tokens.expires_in} seconds).
                        </p>
                      )}

                      {/* Token Data Display */}
                      <div className="max-h-96 overflow-y-auto rounded-lg border border-line bg-surface-muted p-4">
                        <pre className="whitespace-pre-wrap break-all font-mono text-xs text-fg">
                          {JSON.stringify(tokenData, null, 2)}
                        </pre>
                      </div>

                      {/* Close Button */}
                      <div className="flex justify-end">
                        <button
                          type="button"
                          onClick={() => setShowTokenModal(false)}
                          className="btn-secondary"
                        >
                          Close
                        </button>
                      </div>
                    </div>
                  )}
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
    </>
  );
};

export default Sidebar;
