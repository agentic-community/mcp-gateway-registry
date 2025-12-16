import { useState, useEffect, type ReactNode } from 'react';
import { Header } from './Header';
import { Sidebar } from './Sidebar';
import { MobileNav } from './MobileNav';
import { SessionExpiryWarning } from '../auth';
import { cn } from '../../lib/cn';

interface AppShellProps {
  children: ReactNode;
  user: {
    username?: string;
    email?: string;
    is_admin?: boolean;
  } | null;
  onLogout: () => void;
}

export function AppShell({ children, user, onLogout }: AppShellProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Handle window resize
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      // Auto-close sidebar on mobile, open on desktop
      if (mobile && sidebarOpen) {
        setSidebarOpen(false);
      } else if (!mobile && !sidebarOpen) {
        setSidebarOpen(true);
      }
    };

    // Check initial state
    checkMobile();

    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  const handleMenuClick = () => {
    if (isMobile) {
      setMobileNavOpen(true);
    } else {
      setSidebarOpen((prev) => !prev);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <Header
        user={user}
        onMenuClick={handleMenuClick}
        onLogout={onLogout}
      />

      {/* Mobile navigation */}
      <MobileNav
        isOpen={mobileNavOpen}
        onClose={() => setMobileNavOpen(false)}
        isAdmin={user?.is_admin}
      />

      {/* Desktop sidebar */}
      <div className="hidden md:block">
        <Sidebar
          isOpen={sidebarOpen}
          isAdmin={user?.is_admin}
          onClose={() => setSidebarOpen(false)}
        />
      </div>

      {/* Main content area */}
      <main
        className={cn(
          'flex flex-col min-h-screen pt-16 transition-all duration-300',
          sidebarOpen && !isMobile ? 'md:ml-64 lg:ml-72' : ''
        )}
      >
        {children}
      </main>

      {/* Session expiry warning */}
      <SessionExpiryWarning />
    </div>
  );
}
