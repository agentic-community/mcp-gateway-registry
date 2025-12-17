import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@/test/utils';
import AdminPage from '../AdminPage';

describe('AdminPage', () => {
  describe('Page Rendering', () => {
    it('renders page header', () => {
      render(<AdminPage />);

      expect(screen.getByText('Admin Dashboard')).toBeInTheDocument();
      expect(
        screen.getByText('Manage users, permissions, and system configuration')
      ).toBeInTheDocument();
    });

    it('displays admin warning banner', () => {
      render(<AdminPage />);

      expect(screen.getByText('Admin Mode')).toBeInTheDocument();
      expect(
        screen.getByText(/You are viewing the admin interface with elevated permissions/)
      ).toBeInTheDocument();
    });

    it('displays quick stats cards', () => {
      render(<AdminPage />);

      expect(screen.getByText('Total Users')).toBeInTheDocument();
      expect(screen.getByText('Active Agents')).toBeInTheDocument();
      expect(screen.getByText('Revoked Agents')).toBeInTheDocument();
      expect(screen.getByText('Recent Actions')).toBeInTheDocument();
    });
  });

  describe('Admin Sections', () => {
    it('renders admin tools section', () => {
      render(<AdminPage />);

      expect(screen.getByText('Admin Tools')).toBeInTheDocument();
    });

    it('displays User Directory section', () => {
      render(<AdminPage />);

      expect(screen.getByText('User Directory')).toBeInTheDocument();
      expect(
        screen.getByText('Search and manage users, view user details and agent assignments')
      ).toBeInTheDocument();
    });

    it('displays Access Control section with Phase 14 badge', () => {
      render(<AdminPage />);

      expect(screen.getByText('Access Control')).toBeInTheDocument();
      expect(screen.getByText('Phase 14')).toBeInTheDocument();
    });

    it('displays Audit & Logs section', () => {
      render(<AdminPage />);

      expect(screen.getByText('Audit & Logs')).toBeInTheDocument();
      expect(screen.getByText('View admin actions and system audit events')).toBeInTheDocument();
    });

    it('displays System Configuration section', () => {
      render(<AdminPage />);

      expect(screen.getByText('System Configuration')).toBeInTheDocument();
      expect(
        screen.getByText('View operator configuration and system settings')
      ).toBeInTheDocument();
    });

    it('renders Open buttons for each section', () => {
      render(<AdminPage />);

      const openButtons = screen.getAllByText('Open');
      expect(openButtons.length).toBe(4);
    });
  });

  describe('Important Notes', () => {
    it('displays important notes section', () => {
      render(<AdminPage />);

      expect(screen.getByText('Important Notes')).toBeInTheDocument();
    });

    it('shows audit logging notice', () => {
      render(<AdminPage />);

      expect(
        screen.getByText(/All admin actions are logged and auditable/)
      ).toBeInTheDocument();
    });

    it('shows cross-user operation notice', () => {
      render(<AdminPage />);

      expect(
        screen.getByText(/Cross-user operations require confirmation and should include a reason/)
      ).toBeInTheDocument();
    });

    it('shows agent revocation notice', () => {
      render(<AdminPage />);

      expect(
        screen.getByText(/Agent revocations are permanent/)
      ).toBeInTheDocument();
    });

    it('links to management documentation', () => {
      render(<AdminPage />);

      expect(
        screen.getByText('enforceai/instructions/ENFORCEAI_MANAGEMENT.md')
      ).toBeInTheDocument();
    });
  });
});
