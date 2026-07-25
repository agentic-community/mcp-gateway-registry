import React from 'react';
import { PlusIcon } from '@heroicons/react/24/outline';
import SkillCard from '../../SkillCard';
import type { Skill } from '../../../hooks/useSkills';
import Pagination from '../../Pagination';
import EntityGrid from '../EntityGrid';
import EmptyState from '../EmptyState';

interface SkillsSectionProps {
  /** Page of skills to render (already filtered + paginated by the parent). */
  paginatedSkills: Skill[];
  /** Full filtered list length, for pagination + empty-state messaging. */
  filteredCount: number;
  loading: boolean;
  error: string | null;
  /** True when a search term or non-default lifecycle filter is active. */
  isFiltered: boolean;
  /**
   * True when the caller holds the `list_skills` discovery scope (or is admin).
   * When false, the empty state explains that skill access is admin-managed
   * instead of claiming no skills are registered.
   */
  hasListAccess: boolean;
  canModify: boolean;
  page: number;
  totalPages: number;
  pageSize: number;
  onPageChange: (page: number) => void;
  authToken?: string | null;
  onAddSkill: () => void;
  onToggle: (path: string, enabled: boolean) => void;
  onEdit: (skill: Skill) => void;
  onDelete: (path: string) => void;
  onRefreshSuccess: () => void;
  onShowToast: (message: string, type: 'success' | 'error') => void;
  onSkillUpdate: (path: string, updates: Partial<Skill>) => void;
  canToggleSkill: (skill: Skill) => boolean;
}

/**
 * The "Agent Skills" Dashboard collection: header + add button, pagination,
 * and an amber-accented grid of SkillCards with shared empty/error states.
 *
 * Pure presentation over the data + handlers the Dashboard owns — extracted
 * from the inline block to shrink Dashboard.tsx and reuse EntityGrid/EmptyState.
 */
const SkillsSection: React.FC<SkillsSectionProps> = ({
  paginatedSkills,
  filteredCount,
  loading,
  error,
  isFiltered,
  hasListAccess,
  canModify,
  page,
  totalPages,
  pageSize,
  onPageChange,
  authToken,
  onAddSkill,
  onToggle,
  onEdit,
  onDelete,
  onRefreshSuccess,
  onShowToast,
  onSkillUpdate,
  canToggleSkill,
}) => {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">Agent Skills</h2>
        {canModify && (
          <button
            onClick={onAddSkill}
            className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 rounded-lg transition-colors"
          >
            <PlusIcon className="h-4 w-4 mr-1" />
            Add Skill
          </button>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex justify-center mb-4">
          <Pagination
            currentPage={page}
            totalPages={totalPages}
            totalItems={filteredCount}
            pageSize={pageSize}
            onPageChange={onPageChange}
          />
        </div>
      )}

      {error ? (
        <EmptyState tone="error" title="Failed to load skills" subtitle={error} />
      ) : loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600"></div>
        </div>
      ) : filteredCount === 0 ? (
        <EmptyState
          tone="amber"
          title={
            hasListAccess ? 'No skills found' : "You don't have access to view skills"
          }
          subtitle={
            !hasListAccess ? (
              <>
                Skill discovery is managed by your registry administrator. Ask them to grant
                your group the "list_skills" permission so skills appear here.{' '}
                <a
                  href="https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/faq/granting-skill-and-agent-discovery-permissions.md"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-amber-600 dark:text-amber-400 hover:underline"
                >
                  Learn more
                </a>
              </>
            ) : isFiltered ? (
              'Press Enter in the search bar to search semantically'
            ) : (
              'No skills are registered yet'
            )
          }
          cta={
            hasListAccess && !isFiltered && canModify ? (
              <button
                onClick={onAddSkill}
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-amber-600 hover:bg-amber-700 transition-colors"
              >
                <PlusIcon className="h-4 w-4 mr-2" />
                Register Skill
              </button>
            ) : undefined
          }
        />
      ) : (
        <EntityGrid>
          {paginatedSkills.map((skill) => (
            <SkillCard
              key={skill.path}
              skill={skill}
              onToggle={onToggle}
              onEdit={onEdit}
              onDelete={onDelete}
              canModify={canModify}
              canToggle={canToggleSkill(skill)}
              onRefreshSuccess={onRefreshSuccess}
              onShowToast={onShowToast}
              onSkillUpdate={onSkillUpdate}
              authToken={authToken}
            />
          ))}
        </EntityGrid>
      )}
    </div>
  );
};

export default SkillsSection;
