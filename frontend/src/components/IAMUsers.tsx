import React, { useState, useMemo, useCallback } from 'react';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  EyeIcon,
  EyeSlashIcon,
  PencilIcon,
  XMarkIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';
import { useIAMUsers, useIAMGroups, createHumanUser, deleteUser, updateUserGroups, CreateHumanUserPayload } from '../hooks/useIAM';
import DeleteConfirmation from './DeleteConfirmation';
import SearchableSelect from './SearchableSelect';
import ListStateBoundary from './iam/ListStateBoundary';
import RateLimitGroupsEditor from './iam/RateLimitGroupsEditor';
import { useAuth } from '../contexts/AuthContext';
import {
  useRateLimitDefinitions,
  useRateLimitMemberships,
  CALLER_ENTITY_TYPE,
} from '../hooks/useRateLimits';

interface IAMUsersProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

type View = 'list' | 'create';

/**
 * Form validation errors -- follows the same pattern as FederationPeerForm.
 */
interface FormErrors {
  username?: string;
  email?: string;
  first_name?: string;
  last_name?: string;
  password?: string;
}

const IAMUsers: React.FC<IAMUsersProps> = ({ onShowToast }) => {
  const { user: currentUser } = useAuth();
  const isAdmin = currentUser?.is_admin ?? false;
  const { users, isLoading, error, refetch } = useIAMUsers();
  const { groups } = useIAMGroups();
  // Rate-limit definitions + memberships fetched ONCE here and passed to each
  // row's editor (avoids N x 2 GETs on a large user list).
  const { definitions: rlDefinitions } = useRateLimitDefinitions();
  const { memberships: rlMemberships, refetch: refetchRlMemberships } = useRateLimitMemberships();
  const rlGroupOptions = useMemo(() => {
    const names = new Set<string>();
    for (const d of rlDefinitions) {
      // Both caller and caller_target are caller-side group memberships a user can
      // join; the limiter matches a caller's groups against both axes.
      if (
        (d.axis === 'caller' || d.axis === 'caller_target') &&
        d.entity_type === CALLER_ENTITY_TYPE
      ) {
        names.add(d.name);
      }
    }
    return Array.from(names).sort().map((n) => ({ value: n, label: n }));
  }, [rlDefinitions]);
  const [searchQuery, setSearchQuery] = useState('');
  const [view, setView] = useState<View>('list');

  // Create form state
  const [formUsername, setFormUsername] = useState('');
  const [formEmail, setFormEmail] = useState('');
  const [formFirstName, setFormFirstName] = useState('');
  const [formLastName, setFormLastName] = useState('');
  const [formPassword, setFormPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [formGroups, setFormGroups] = useState<Set<string>>(new Set());
  const [isCreating, setIsCreating] = useState(false);
  const [errors, setErrors] = useState<FormErrors>({});

  // Delete state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // Edit groups state
  const [editingUser, setEditingUser] = useState<string | null>(null);
  const [editGroups, setEditGroups] = useState<Set<string>>(new Set());
  const [isSavingGroups, setIsSavingGroups] = useState(false);

  const filteredUsers = useMemo(() => {
    if (!searchQuery) return users;
    const q = searchQuery.toLowerCase();
    return users.filter(
      (u) =>
        u.username.toLowerCase().includes(q) ||
        (u.email || '').toLowerCase().includes(q) ||
        (u.first_name || '').toLowerCase().includes(q) ||
        (u.last_name || '').toLowerCase().includes(q)
    );
  }, [users, searchQuery]);

  const resetForm = useCallback(() => {
    setFormUsername('');
    setFormEmail('');
    setFormFirstName('');
    setFormLastName('');
    setFormPassword('');
    setShowPassword(false);
    setFormGroups(new Set());
    setErrors({});
  }, []);

  const toggleGroup = (groupName: string) => {
    setFormGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  };

  /** Clear a single field error when the user edits that field. */
  const clearError = (field: keyof FormErrors) => {
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: undefined }));
    }
  };

  /** Validate all fields. Returns true if valid. */
  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    if (!formUsername.trim()) newErrors.username = 'Username is required';
    if (!formEmail.trim()) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formEmail.trim())) {
      newErrors.email = 'Enter a valid email address';
    }
    if (!formFirstName.trim()) newErrors.first_name = 'First name is required';
    if (!formLastName.trim()) newErrors.last_name = 'Last name is required';
    if (!formPassword) newErrors.password = 'Password is required';

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleCreate = async () => {
    if (!validateForm()) return;
    setIsCreating(true);
    try {
      const payload: CreateHumanUserPayload = {
        username: formUsername.trim(),
        email: formEmail.trim(),
        first_name: formFirstName.trim(),
        last_name: formLastName.trim(),
        password: formPassword,
        groups: formGroups.size > 0 ? Array.from(formGroups) : undefined,
      };
      await createHumanUser(payload);
      onShowToast(`User "${formUsername}" created successfully`, 'success');
      resetForm();
      setView('list');
      await refetch();
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const message = Array.isArray(detail)
        ? detail.map((d: any) => d.msg).join(', ')
        : detail || 'Failed to create user';
      onShowToast(message, 'error');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (username: string) => {
    await deleteUser(username);
    onShowToast(`User "${username}" deleted`, 'success');
    setDeleteTarget(null);
    await refetch();
  };

  const startEditGroups = (username: string, currentGroups: string[]) => {
    setEditingUser(username);
    setEditGroups(new Set(currentGroups));
  };

  const cancelEditGroups = () => {
    setEditingUser(null);
    setEditGroups(new Set());
  };

  const handleSaveGroups = async () => {
    if (!editingUser) return;
    setIsSavingGroups(true);
    try {
      const result = await updateUserGroups(editingUser, Array.from(editGroups));
      const addedCount = result.added?.length || 0;
      const removedCount = result.removed?.length || 0;
      if (addedCount > 0 || removedCount > 0) {
        onShowToast(
          `Groups updated: ${addedCount} added, ${removedCount} removed`,
          'success'
        );
      } else {
        onShowToast('No changes made', 'info');
      }
      setEditingUser(null);
      setEditGroups(new Set());
      await refetch();
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Failed to update groups';
      onShowToast(message, 'error');
    } finally {
      setIsSavingGroups(false);
    }
  };

  const toggleEditGroup = (groupName: string) => {
    setEditGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  };

  const addGroupToEdit = (groupName: string) => {
    if (groupName && !editGroups.has(groupName)) {
      setEditGroups((prev) => {
        const next = new Set(prev);
        next.add(groupName);
        return next;
      });
    }
  };

  const removeGroupFromEdit = (groupName: string) => {
    setEditGroups((prev) => {
      const next = new Set(prev);
      next.delete(groupName);
      return next;
    });
  };

  // Helper: input border class based on error state
  const inputClass = (field: keyof FormErrors) =>
    `w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
      errors[field] ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
    }`;

  // ─── Create View ──────────────────────────────────────────────
  if (view === 'create') {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; Users &gt; Create
          </h2>
          <button
            onClick={() => { resetForm(); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" />
            Back to List
          </button>
        </div>

        <div className="space-y-4 max-w-lg">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Username *</label>
            <input type="text" value={formUsername}
              onChange={(e) => { setFormUsername(e.target.value); clearError('username'); }}
              placeholder="e.g. jdoe"
              className={inputClass('username')} />
            {errors.username && <p className="mt-1 text-sm text-red-500">{errors.username}</p>}
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Email *</label>
            <input type="email" value={formEmail}
              onChange={(e) => { setFormEmail(e.target.value); clearError('email'); }}
              placeholder="user@example.com"
              className={inputClass('email')} />
            {errors.email && <p className="mt-1 text-sm text-red-500">{errors.email}</p>}
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">First Name *</label>
              <input type="text" value={formFirstName}
                onChange={(e) => { setFormFirstName(e.target.value); clearError('first_name'); }}
                className={inputClass('first_name')} />
              {errors.first_name && <p className="mt-1 text-sm text-red-500">{errors.first_name}</p>}
            </div>
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Last Name *</label>
              <input type="text" value={formLastName}
                onChange={(e) => { setFormLastName(e.target.value); clearError('last_name'); }}
                className={inputClass('last_name')} />
              {errors.last_name && <p className="mt-1 text-sm text-red-500">{errors.last_name}</p>}
            </div>
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Password *</label>
            <div className="relative">
              <input
                type={showPassword ? 'text' : 'password'}
                value={formPassword}
                onChange={(e) => { setFormPassword(e.target.value); clearError('password'); }}
                placeholder="Initial password"
                className={`${inputClass('password')} pr-10`}
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                title={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeSlashIcon className="h-4 w-4" /> : <EyeIcon className="h-4 w-4" />}
              </button>
            </div>
            {errors.password && <p className="mt-1 text-sm text-red-500">{errors.password}</p>}
          </div>

          {/* Group selection */}
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-2">Groups</label>
            <div className="space-y-2 max-h-48 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg p-3">
              {groups.length === 0 ? (
                <p className="text-xs text-gray-400">No groups available</p>
              ) : (
                groups.map((g) => (
                  <label key={g.name} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={formGroups.has(g.name)}
                      onChange={() => toggleGroup(g.name)}
                      className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{g.name}</span>
                  </label>
                ))
              )}
            </div>
          </div>
        </div>

        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button onClick={() => { resetForm(); setView('list'); }}
            className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800">
            Cancel
          </button>
          <button onClick={handleCreate}
            disabled={isCreating}
            className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed">
            {isCreating ? 'Creating...' : 'Create User'}
          </button>
        </div>
      </div>
    );
  }

  // ─── List View ────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">IAM &gt; Users</h2>
        <div className="flex items-center space-x-2">
          <button onClick={refetch} className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200" title="Refresh">
            <ArrowPathIcon className="h-5 w-5" />
          </button>
          <button onClick={() => setView('create')}
            className="flex items-center px-3 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700">
            <PlusIcon className="h-4 w-4 mr-1" /> Create User
          </button>
        </div>
      </div>

      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search users..."
          className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent" />
      </div>

      <ListStateBoundary
        isLoading={isLoading}
        error={error}
        isEmpty={filteredUsers.length === 0}
        emptyMessage={
          searchQuery ? 'No users match your search.' : 'No users yet. Create your first user.'
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Username</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Email</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Name</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Groups</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Rate-limit Groups</th>
                <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map((u) => (
                <React.Fragment key={u.username}>
                  <tr className="table-row border-b border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">{u.username}</td>
                    <td className="py-3 px-4 text-gray-600 dark:text-gray-400">{u.email || '\u2014'}</td>
                    <td className="py-3 px-4 text-gray-600 dark:text-gray-400">
                      {[u.first_name, u.last_name].filter(Boolean).join(' ') || '\u2014'}
                    </td>
                    <td className="py-3 px-4">
                      <div className="flex flex-wrap gap-1 items-center">
                        {(u.groups || []).map((g) => (
                          <span key={g} className="inline-block px-2 py-0.5 text-xs rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                            {g}
                          </span>
                        ))}
                        {(!u.groups || u.groups.length === 0) && <span className="text-gray-400 text-xs">{'\u2014'}</span>}
                        <button
                          onClick={() => startEditGroups(u.username, u.groups || [])}
                          className="ml-2 p-1 text-gray-400 hover:text-purple-600 dark:hover:text-purple-400"
                          title="Edit groups"
                        >
                          <PencilIcon className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </td>
                    <td className="py-3 px-4">
                      <RateLimitGroupsEditor
                        subjectType="user"
                        subject={u.username}
                        groupOptions={rlGroupOptions}
                        memberships={rlMemberships}
                        onSaved={refetchRlMemberships}
                        onShowToast={onShowToast}
                        isAdmin={isAdmin}
                      />
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button onClick={() => setDeleteTarget(u.username)} className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400" title="Delete user">
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                  {deleteTarget === u.username && (
                    <tr>
                      <td colSpan={6} className="p-2">
                        <DeleteConfirmation
                          entityType="user"
                          entityName={u.username}
                          entityPath={u.username}
                          onConfirm={handleDelete}
                          onCancel={() => setDeleteTarget(null)}
                        />
                      </td>
                    </tr>
                  )}
                  {editingUser === u.username && (
                    <tr className="bg-purple-50 dark:bg-purple-900/10">
                      <td colSpan={6} className="p-4">
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              Edit Groups for {u.username}
                            </span>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={cancelEditGroups}
                                className="px-3 py-1 text-xs text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
                              >
                                Cancel
                              </button>
                              <button
                                onClick={handleSaveGroups}
                                disabled={isSavingGroups}
                                className="flex items-center px-3 py-1 text-xs text-white bg-purple-600 rounded hover:bg-purple-700 disabled:opacity-50"
                              >
                                <CheckIcon className="h-3 w-3 mr-1" />
                                {isSavingGroups ? 'Saving...' : 'Save'}
                              </button>
                            </div>
                          </div>

                          {/* Selected groups as removable tags */}
                          <div className="flex flex-wrap gap-2">
                            {Array.from(editGroups).map((groupName) => (
                              <span
                                key={groupName}
                                className="inline-flex items-center px-2 py-1 text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full"
                              >
                                {groupName}
                                <button
                                  type="button"
                                  onClick={() => removeGroupFromEdit(groupName)}
                                  className="ml-1 hover:text-purple-900 dark:hover:text-purple-100"
                                >
                                  <XMarkIcon className="h-3 w-3" />
                                </button>
                              </span>
                            ))}
                            {editGroups.size === 0 && (
                              <span className="text-xs text-gray-400 italic">No groups assigned</span>
                            )}
                          </div>

                          {/* Searchable dropdown to add groups */}
                          <div className="max-w-sm">
                            <SearchableSelect
                              options={groups
                                .filter((g) => !editGroups.has(g.name))
                                .map((g) => ({
                                  value: g.name,
                                  label: g.name,
                                  description: g.path || undefined,
                                }))}
                              value=""
                              onChange={addGroupToEdit}
                              placeholder="Search and add groups..."
                              maxDescriptionWords={5}
                            />
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </ListStateBoundary>
    </div>
  );
};

export default IAMUsers;
