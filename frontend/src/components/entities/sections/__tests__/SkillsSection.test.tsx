import React from 'react';
import { render, screen } from '@testing-library/react';

// SkillsSection pulls in SkillCard -> react-markdown (ESM). The empty-state
// tests never render a card, so mock the markdown deps to keep jest's module
// loader happy (same pattern as SkillResources.test.tsx).
jest.mock('react-markdown', () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));

import SkillsSection from '../SkillsSection';

// Minimal prop factory: SkillsSection is pure presentation, so most handlers
// can be no-ops. Individual tests override just the fields they exercise.
const baseProps = {
  paginatedSkills: [],
  filteredCount: 0,
  loading: false,
  error: null,
  isFiltered: false,
  hasListAccess: true,
  canModify: false,
  page: 0,
  totalPages: 1,
  pageSize: 50,
  onPageChange: () => {},
  authToken: null,
  onAddSkill: () => {},
  onToggle: () => {},
  onEdit: () => {},
  onDelete: () => {},
  onRefreshSuccess: () => {},
  onShowToast: () => {},
  onSkillUpdate: () => {},
  canToggleSkill: () => false,
};

describe('SkillsSection empty state', () => {
  it('shows the admin-access hint when the caller lacks list_skills', () => {
    render(<SkillsSection {...baseProps} hasListAccess={false} />);

    expect(
      screen.getByText("You don't have access to view skills"),
    ).toBeInTheDocument();
    expect(screen.getByText(/list_skills/)).toBeInTheDocument();
    // The "no skills registered yet" copy must NOT appear — it would be
    // misleading when skills exist but are hidden by the discovery gate.
    expect(
      screen.queryByText(/No skills are registered yet/),
    ).not.toBeInTheDocument();
  });

  it('shows the ordinary empty copy when the caller has list access', () => {
    render(<SkillsSection {...baseProps} hasListAccess={true} />);

    expect(screen.getByText('No skills found')).toBeInTheDocument();
    expect(screen.getByText(/No skills are registered yet/)).toBeInTheDocument();
    expect(
      screen.queryByText("You don't have access to view skills"),
    ).not.toBeInTheDocument();
  });
});
