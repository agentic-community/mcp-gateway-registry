import React from 'react';
import {
  ExclamationTriangleIcon,
  ArrowTopRightOnSquareIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import DetailsModal from './DetailsModal';
import type { ExistingEntity, EntityType } from '../types/duplicateCheck';


interface DuplicateCheckModalProps {
  isOpen: boolean;
  onClose: () => void;
  onProceed: () => void;
  onPickExisting: (entity: ExistingEntity) => void;
  /** Exact-URL hits (across any entity type). Rendered as the prominent warning. */
  collisionWith: ExistingEntity[];
  /** Similarity-based hits (across any entity type). Rendered as a secondary list. */
  advisoryMatches: ExistingEntity[];
  isLoading?: boolean;
}


const ENTITY_LABEL: Record<EntityType, string> = {
  mcp_server: 'server',
  a2a_agent: 'agent',
  skill: 'skill',
};


const ENTITY_LABEL_PLURAL: Record<EntityType, string> = {
  mcp_server: 'servers',
  a2a_agent: 'agents',
  skill: 'skills',
};


function pluralize(entityType: EntityType, count: number): string {
  return count === 1
    ? ENTITY_LABEL[entityType]
    : ENTITY_LABEL_PLURAL[entityType];
}


/**
 * Human-readable label for a heterogeneous list of matches.
 *
 * The dedup checks are cross-entity: a server registration can
 * surface an agent or skill, and the advisory list is always
 * cross-entity. When every match in the list shares the same
 * entity_type, return that type's label ("1 server", "3 skills");
 * otherwise fall back to entity-agnostic copy ("3 entries") to
 * avoid lying about the list's composition.
 */
function describeMatches(matches: { entity_type: EntityType }[]): string {
  if (matches.length === 0) {
    return '';
  }
  const types = new Set(matches.map((m) => m.entity_type));
  if (types.size === 1) {
    const onlyType = matches[0].entity_type;
    return `${matches.length} ${pluralize(onlyType, matches.length)}`;
  }
  return `${matches.length} ${matches.length === 1 ? 'entry' : 'entries'}`;
}


/**
 * Renders one card per existing entity with its name, path, owner,
 * and a "View" button. Used for both the collision_with section
 * (prominent, exact-URL) and the advisory_matches section
 * (subdued, similarity-based).
 *
 * Redacted entries (caller cannot view the entity per visibility
 * rules) have blank path/name and render as a non-actionable
 * placeholder.
 */
const EntityCard: React.FC<{
  entity: ExistingEntity;
  onPick: () => void;
  variant: 'collision' | 'advisory';
}> = ({ entity, onPick, variant }) => {
  const isRedacted = !entity.path && !entity.name;

  if (isRedacted) {
    return (
      <li className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-gray-50 dark:bg-gray-900">
        <p className="text-sm italic text-gray-500 dark:text-gray-400">
          A {ENTITY_LABEL[entity.entity_type]} at this URL is already
          registered, but you don&apos;t have permission to view it.
          Contact your registry administrator if this conflicts with
          your registration.
        </p>
      </li>
    );
  }

  return (
    <li className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-800">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium text-gray-900 dark:text-gray-100 truncate">
              {entity.name}
            </span>
            <code className="text-xs text-gray-500 dark:text-gray-400 truncate">
              {entity.path}
            </code>
            <span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200">
              {ENTITY_LABEL[entity.entity_type]}
            </span>
          </div>
          {entity.owner && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Owner: {entity.owner}
            </p>
          )}
          {variant === 'advisory' && entity.relevance_score != null && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Relevance: {entity.relevance_score.toFixed(2)}
            </p>
          )}
          {entity.match_reason && (
            <p className="mt-1 text-xs italic text-gray-500 dark:text-gray-400">
              {entity.match_reason}
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={onPick}
          className="flex-shrink-0 inline-flex items-center gap-1 px-2.5 py-1.5 text-xs font-medium rounded border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700"
        >
          View
          <ArrowTopRightOnSquareIcon className="h-3.5 w-3.5" />
        </button>
      </div>
    </li>
  );
};


/**
 * Modal that surfaces likely-duplicate existing entities before the
 * user actually registers a new one. Shows two distinct sections:
 *
 * - **collision_with** (URL match): prominent warning. These are
 *   existing entities that share the proposed identity URL — the
 *   strongest signal that the user might be registering the same
 *   thing twice.
 * - **advisory_matches** (similarity): subdued secondary list.
 *   These are entities whose name/description matched the proposed
 *   registration through the semantic search backend.
 *
 * The modal does not block registration — the user can always
 * "Register anyway."
 */
const DuplicateCheckModal: React.FC<DuplicateCheckModalProps> = ({
  isOpen,
  onClose,
  onProceed,
  onPickExisting,
  collisionWith,
  advisoryMatches,
  isLoading = false,
}) => {
  const hasCollision = collisionWith.length > 0;
  const hasAdvisory = advisoryMatches.length > 0;

  // The advisory section can return entries of any entity type, not
  // just the type the user is registering. Build the title from
  // what's actually in the list.
  const title = hasCollision
    ? 'A matching entry already exists'
    : `Similar ${describeMatches(advisoryMatches)} found`;

  return (
    <DetailsModal
      title={title}
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="2xl"
      zIndexClass="z-[60]"
    >
      <div className="space-y-4">
        {hasCollision && (
          <>
            <div className="flex items-start gap-3 p-3 rounded-lg bg-red-50 border border-red-200 dark:bg-red-900/30 dark:border-red-700">
              <ExclamationTriangleIcon className="h-5 w-5 mt-0.5 text-red-600 dark:text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-800 dark:text-red-200">
                The URL you&apos;re registering matches{' '}
                {collisionWith.length === 1
                  ? 'an existing entry'
                  : `${collisionWith.length} existing entries`}
                . You can pick the existing one, edit your entry, or register
                a new one anyway.
              </p>
            </div>
            <ul className="space-y-3">
              {collisionWith.map((entity, index) => (
                <EntityCard
                  key={`collision-${index}-${entity.path}`}
                  entity={entity}
                  onPick={() => onPickExisting(entity)}
                  variant="collision"
                />
              ))}
            </ul>
          </>
        )}

        {hasAdvisory && (
          <>
            <div className="flex items-start gap-3 p-3 rounded-lg bg-blue-50 border border-blue-200 dark:bg-blue-900/30 dark:border-blue-700">
              <InformationCircleIcon className="h-5 w-5 mt-0.5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <p className="text-sm text-blue-800 dark:text-blue-200">
                {hasCollision ? 'You may also want to consider these similar entries:' : (
                  <>
                    We found {describeMatches(advisoryMatches)} similar to
                    what you&apos;re about to register. Pick an existing one,
                    edit your entry, or register a new one anyway.
                  </>
                )}
              </p>
            </div>
            <ul className="space-y-3 max-h-96 overflow-y-auto">
              {advisoryMatches.map((entity, index) => (
                <EntityCard
                  key={`advisory-${index}-${entity.path}`}
                  entity={entity}
                  onPick={() => onPickExisting(entity)}
                  variant="advisory"
                />
              ))}
            </ul>
          </>
        )}

        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 disabled:opacity-50"
          >
            Edit my entry
          </button>
          <button
            type="button"
            onClick={onProceed}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 rounded-lg disabled:opacity-50"
          >
            {isLoading ? 'Registering...' : 'Register anyway'}
          </button>
        </div>
      </div>
    </DetailsModal>
  );
};


export default DuplicateCheckModal;
