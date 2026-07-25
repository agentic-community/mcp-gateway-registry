import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { PencilIcon, XMarkIcon, CheckIcon, NoSymbolIcon } from '@heroicons/react/24/outline';
import SearchableSelect from '../SearchableSelect';
import {
  setRateLimitMembership,
  deleteRateLimitMembership,
  quarantineAdd,
  quarantineRemove,
  QUARANTINE_CALLER_GROUP,
  RateLimitMembership,
} from '../../hooks/useRateLimits';

interface RateLimitGroupsEditorProps {
  // 'user' (subject = username) or 'client' (subject = client_id).
  subjectType: 'user' | 'client';
  subject: string;
  // Shared data fetched ONCE by the parent list (not per row): the defined
  // caller-group names as select options, and all memberships. This keeps a large
  // Users/M2M list from issuing N x 2 GETs.
  groupOptions: { value: string; label: string }[];
  memberships: RateLimitMembership[];
  // Called after a successful save so the parent can refetch the shared membership list.
  onSaved: () => void;
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
  // Quarantine (add/remove) is an admin-only kill switch; the toggle is hidden
  // for non-admins (the backend also enforces admin on the endpoint).
  isAdmin?: boolean;
}

/**
 * Inline display + editor for a caller's RATE-LIMIT group membership, reused on
 * the IAM Users and M2M lists. Shows the subject's current rate-limit groups as
 * chips and lets an admin edit them via a multi-select of the DEFINED caller
 * (group) rate-limit definitions -- so only real groups can be assigned.
 *
 * This is distinct from a user's IdP/authz groups: rate-limit membership lives in
 * the rate_limit_memberships collection and never affects scopes. Definitions and
 * memberships are fetched once by the parent and passed in as props.
 */
const RateLimitGroupsEditor: React.FC<RateLimitGroupsEditorProps> = ({
  subjectType,
  subject,
  groupOptions,
  memberships,
  onSaved,
  onShowToast,
  isAdmin = false,
}) => {
  const [editing, setEditing] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [isSaving, setIsSaving] = useState(false);
  const [isQuarantining, setIsQuarantining] = useState(false);

  // This subject's current membership groups.
  const currentGroups = useMemo(() => {
    const m = memberships.find(
      (x) => x.subject_type === subjectType && x.subject === subject,
    );
    return m?.groups || [];
  }, [memberships, subjectType, subject]);

  // A caller (user or client) is quarantined iff it belongs to the reserved
  // quarantine-callers group. Toggling it is the admin-only kill switch.
  const isQuarantined = currentGroups.includes(QUARANTINE_CALLER_GROUP);

  const handleToggleQuarantine = useCallback(async () => {
    setIsQuarantining(true);
    try {
      if (isQuarantined) {
        await quarantineRemove(subjectType, subject);
        onShowToast(`Removed ${subjectType}:${subject} from quarantine`, 'success');
      } else {
        await quarantineAdd(subjectType, subject);
        onShowToast(`Quarantined ${subjectType}:${subject} (all traffic blocked)`, 'success');
      }
      onSaved();
    } catch (err: any) {
      onShowToast(
        err?.response?.data?.detail || err?.message || 'Failed to update quarantine',
        'error',
      );
    } finally {
      setIsQuarantining(false);
    }
  }, [isQuarantined, subjectType, subject, onSaved, onShowToast]);

  useEffect(() => {
    if (editing) setSelected(new Set(currentGroups));
  }, [editing, currentGroups]);

  const addGroup = useCallback((name: string) => {
    if (name) setSelected((prev) => new Set(prev).add(name));
  }, []);

  const removeGroup = useCallback((name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      next.delete(name);
      return next;
    });
  }, []);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const groups = Array.from(selected);
      const id = `${subjectType}:${subject}`;
      if (groups.length === 0) {
        // No groups left -> delete the membership record entirely.
        await deleteRateLimitMembership(id);
      } else {
        await setRateLimitMembership(subjectType, subject, groups);
      }
      onShowToast('Rate-limit groups updated', 'success');
      setEditing(false);
      onSaved();
    } catch (err: any) {
      onShowToast(
        err?.response?.data?.detail || err?.message || 'Failed to update rate-limit groups',
        'error',
      );
    } finally {
      setIsSaving(false);
    }
  };

  if (!editing) {
    // Rate-limit group chips exclude the reserved quarantine group; quarantine
    // state is shown as its own red badge + toggle so it reads as a kill switch,
    // not a rate group.
    const rateChips = currentGroups.filter((g) => g !== QUARANTINE_CALLER_GROUP);
    return (
      <div className="flex flex-wrap gap-1 items-center">
        {isQuarantined && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300">
            <NoSymbolIcon className="h-3 w-3" />
            quarantined
          </span>
        )}
        {rateChips.map((g) => (
          <span
            key={g}
            className="inline-block px-2 py-0.5 text-xs rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300"
          >
            {g}
          </span>
        ))}
        {rateChips.length === 0 && !isQuarantined && (
          <span className="text-gray-400 text-xs">{'—'}</span>
        )}
        <button
          onClick={() => setEditing(true)}
          className="ml-2 p-1 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400"
          title="Edit rate-limit groups"
        >
          <PencilIcon className="h-3.5 w-3.5" />
        </button>
        {isAdmin && (
          <button
            onClick={handleToggleQuarantine}
            disabled={isQuarantining}
            className={`p-1 disabled:opacity-50 ${
              isQuarantined
                ? 'text-red-500 hover:text-red-700 dark:hover:text-red-300'
                : 'text-gray-400 hover:text-red-600 dark:hover:text-red-400'
            }`}
            title={
              isQuarantined
                ? 'Remove from quarantine (restore traffic)'
                : 'Quarantine (block ALL data-plane traffic)'
            }
          >
            <NoSymbolIcon className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {Array.from(selected).map((g) => (
          <span
            key={g}
            className="inline-flex items-center px-2 py-1 text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full"
          >
            {g}
            <button
              type="button"
              onClick={() => removeGroup(g)}
              className="ml-1 hover:text-blue-900 dark:hover:text-blue-100"
            >
              <XMarkIcon className="h-3 w-3" />
            </button>
          </span>
        ))}
        {selected.size === 0 && (
          <span className="text-xs text-gray-400 italic">No rate-limit groups</span>
        )}
      </div>

      <div className="max-w-sm">
        <SearchableSelect
          options={groupOptions}
          value=""
          onChange={addGroup}
          placeholder={
            groupOptions.length === 0
              ? 'No rate-limit groups defined yet'
              : 'Add a rate-limit group...'
          }
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="flex items-center px-3 py-1 text-xs text-white bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          <CheckIcon className="h-3 w-3 mr-1" />
          {isSaving ? 'Saving...' : 'Save'}
        </button>
        <button
          onClick={() => setEditing(false)}
          className="px-3 py-1 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
        >
          Cancel
        </button>
      </div>
    </div>
  );
};

export default RateLimitGroupsEditor;
