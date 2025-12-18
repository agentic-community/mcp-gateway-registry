import { NavLink, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  ServerStackIcon,
  CpuChipIcon,
  UserGroupIcon,
  KeyIcon,
  ShieldCheckIcon,
  WrenchScrewdriverIcon,
  DocumentMagnifyingGlassIcon,
  UserCircleIcon,
  Cog6ToothIcon,
  QuestionMarkCircleIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';
import { useState } from 'react';

interface NavItem {
  name: string;
  href: string;
  icon: typeof HomeIcon;
  badge?: string;
  adminOnly?: boolean;
  children?: { name: string; href: string }[];
}

interface NavSection {
  title?: string;
  items: NavItem[];
}

const navigation: NavSection[] = [
  {
    items: [
      { name: 'Overview', href: '/', icon: HomeIcon },
    ],
  },
  {
    title: 'Registry',
    items: [
      { name: 'Servers', href: '/servers', icon: ServerStackIcon },
      { name: 'A2A Agents', href: '/a2a-agents', icon: CpuChipIcon },
    ],
  },
  {
    title: 'EnforceAI',
    items: [
      { name: 'Agents', href: '/agents', icon: UserGroupIcon },
      {
        name: 'Credentials',
        href: '/credentials',
        icon: KeyIcon,
        children: [
          { name: 'API Keys', href: '/credentials/api-keys' },
          { name: 'Gateway Tokens', href: '/credentials/tokens' },
          { name: 'Upstream Credentials', href: '/credentials/upstream' },
        ],
      },
    ],
  },
  {
    title: 'Policy',
    items: [
      { name: 'Scopes', href: '/scopes', icon: ShieldCheckIcon },
      { name: 'Tools', href: '/tools', icon: WrenchScrewdriverIcon },
    ],
  },
  {
    title: 'Monitoring',
    items: [
      { name: 'Audit', href: '/audit', icon: DocumentMagnifyingGlassIcon },
    ],
  },
  {
    title: 'Administration',
    items: [
      { name: 'Admin', href: '/admin', icon: UserCircleIcon, adminOnly: true },
      { name: 'Network Policy', href: '/admin/egress-allowlist', icon: ShieldCheckIcon, adminOnly: true },
    ],
  },
  {
    items: [
      { name: 'Settings', href: '/settings', icon: Cog6ToothIcon },
      { name: 'Help', href: '/help', icon: QuestionMarkCircleIcon },
    ],
  },
];

interface SidebarProps {
  isOpen: boolean;
  isAdmin?: boolean;
  onClose?: () => void;
}

export function Sidebar({ isOpen, isAdmin = false, onClose }: SidebarProps) {
  const location = useLocation();
  const [expandedItems, setExpandedItems] = useState<string[]>([]);

  const toggleExpanded = (name: string) => {
    setExpandedItems((prev) =>
      prev.includes(name)
        ? prev.filter((item) => item !== name)
        : [...prev, name]
    );
  };

  const isActive = (href: string) => {
    if (href === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(href);
  };

  const handleNavClick = () => {
    // Close sidebar on mobile when clicking a nav item
    if (window.innerWidth < 768 && onClose) {
      onClose();
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <aside
      className={cn(
        'fixed left-0 top-16 bottom-0 z-40',
        'w-64 lg:w-72',
        'bg-white dark:bg-gray-800',
        'border-r border-gray-200 dark:border-gray-700',
        'overflow-y-auto',
        'transition-transform duration-300 ease-in-out',
        isOpen ? 'translate-x-0' : '-translate-x-full'
      )}
    >
      <nav className="flex flex-col h-full py-4">
        <div className="flex-1 space-y-1 px-3">
          {navigation.map((section, sectionIndex) => (
            <div key={sectionIndex} className="pb-2">
              {section.title && (
                <h3 className="px-3 pt-4 pb-2 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
                  {section.title}
                </h3>
              )}
              {section.items.map((item) => {
                // Skip admin-only items for non-admin users
                if (item.adminOnly && !isAdmin) {
                  return null;
                }

                const hasChildren = item.children && item.children.length > 0;
                const isExpanded = expandedItems.includes(item.name);
                const active = isActive(item.href);

                if (hasChildren) {
                  return (
                    <div key={item.name}>
                      <button
                        onClick={() => toggleExpanded(item.name)}
                        className={cn(
                          'w-full flex items-center justify-between px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                          active
                            ? 'bg-primary-50 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300'
                            : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                        )}
                      >
                        <div className="flex items-center">
                          <item.icon
                            className={cn(
                              'mr-3 h-5 w-5 flex-shrink-0',
                              active
                                ? 'text-primary-600 dark:text-primary-400'
                                : 'text-gray-400 dark:text-gray-500'
                            )}
                          />
                          {item.name}
                        </div>
                        {isExpanded ? (
                          <ChevronDownIcon className="h-4 w-4" />
                        ) : (
                          <ChevronRightIcon className="h-4 w-4" />
                        )}
                      </button>
                      {isExpanded && (
                        <div className="ml-8 mt-1 space-y-1">
                          {item.children?.map((child) => (
                            <NavLink
                              key={child.href}
                              to={child.href}
                              onClick={handleNavClick}
                              className={({ isActive: childActive }) =>
                                cn(
                                  'block px-3 py-2 text-sm rounded-lg transition-colors',
                                  childActive
                                    ? 'bg-primary-50 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300'
                                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
                                )
                              }
                            >
                              {child.name}
                            </NavLink>
                          ))}
                        </div>
                      )}
                    </div>
                  );
                }

                return (
                  <NavLink
                    key={item.name}
                    to={item.href}
                    onClick={handleNavClick}
                    className={cn(
                      'flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors',
                      active
                        ? 'bg-primary-50 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    )}
                  >
                    <item.icon
                      className={cn(
                        'mr-3 h-5 w-5 flex-shrink-0',
                        active
                          ? 'text-primary-600 dark:text-primary-400'
                          : 'text-gray-400 dark:text-gray-500'
                      )}
                    />
                    {item.name}
                    {item.badge && (
                      <span className="ml-auto inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-primary-100 dark:bg-primary-800 text-primary-800 dark:text-primary-200">
                        {item.badge}
                      </span>
                    )}
                  </NavLink>
                );
              })}
            </div>
          ))}
        </div>

        {/* Version info at bottom */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            Enforce Gateway v1.0
          </p>
        </div>
      </nav>
    </aside>
  );
}
