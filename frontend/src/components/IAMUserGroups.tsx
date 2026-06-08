import React, { useState, useMemo, useEffect, useCallback, useRef } from 'react';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  PencilIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import {
  useUserGroups,
  createUserGroup,
  updateUserGroup,
  deleteUserGroup,
  createPingFederateUser,
  CreateUserGroupPayload,
  PatchUserGroupPayload,
  IdPUserGroup,
} from '../hooks/useIAM';
import { useRegistryConfig } from '../hooks/useRegistryConfig';
import DeleteConfirmation from './DeleteConfirmation';
import Pagination from './Pagination';

/**
 * IAMUserGroups renders the "User Groups" tab of the IAM Settings page.
 *
 * It manages records in the MongoDB ``idp_user_groups`` collection. These
 * records back the auth server's user-to-group fallback for IdPs that do not
 * carry group memberships in JWTs (e.g. PingFederate). See issue #1127.
 *
 * Mirrors the visual & behavior pattern of IAMM2M.tsx.
 */
interface IAMUserGroupsProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

type View = 'list' | 'register' | 'edit';

interface FormErrors {
  username?: string;
  groups?: string;
  email?: string;
  pingfederatePassword?: string;
}

// Minimum password length for PingFederate Simple PCV creation. Mirrors the
// backend's validation; kept here too so we can fail fast in the UI without a
// round trip.
const PINGFEDERATE_PASSWORD_MIN_LENGTH = 8;

// Mirrors the backend regex at registry/schemas/idp_user_group.py:22
const USERNAME_REGEX = /^[A-Za-z0-9_\-.@]{1,256}$/;

const PROVIDER_STYLES: Record<string, string> = {
  manual: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300',
  pingfederate: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  okta: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  auth0: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  keycloak: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  entra: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
};

const PAGE_SIZE = 25;

// Default no-op formatter for missing dates.
const EM_DASH = '—';


const ProviderBadge: React.FC<{ provider: string }> = ({ provider }) => {
  const style =
    PROVIDER_STYLES[provider] ??
    'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300';
  return (
    <span
      className={`inline-block px-2 py-0.5 text-xs rounded-full font-mono ${style}`}
    >
      {provider}
    </span>
  );
};


const extractDetail = (err: any, fallback: string): string => {
  const detail = err?.response?.data?.detail;
  if (Array.isArray(detail)) {
    return (
      detail.map((d: any) => d?.msg).filter(Boolean).join(', ') || fallback
    );
  }
  return detail || fallback;
};


const IAMUserGroups: React.FC<IAMUserGroupsProps> = ({ onShowToast }) => {
  // ─── Config (issue #1127) ────────────────────────────────────
  // The Register form gains an optional "Also create in PingFederate"
  // checkbox when the active provider is PingFederate AND the auth server
  // is configured to manage Simple PCV users. Backend exposes a single
  // boolean to gate the UI affordance.
  const { config } = useRegistryConfig();
  const pingfederateUserManagementEnabled =
    config?.pingfederate_user_management_enabled ?? false;

  // ─── List state ──────────────────────────────────────────────
  const [searchInput, setSearchInput] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [page, setPage] = useState(0);
  const [providerFilter] = useState<string>('');

  // Debounce search input -- 300ms.
  useEffect(() => {
    const handle = setTimeout(() => {
      setDebouncedSearch(searchInput.trim());
      setPage(0);
    }, 300);
    return () => clearTimeout(handle);
  }, [searchInput]);

  const { items, total, isLoading, error, refetch } = useUserGroups({
    skip: page * PAGE_SIZE,
    limit: PAGE_SIZE,
    q: debouncedSearch || undefined,
    provider: providerFilter || undefined,
  });

  const totalPages = useMemo(
    () => Math.max(1, Math.ceil(total / PAGE_SIZE)),
    [total]
  );

  // ─── View / form state ───────────────────────────────────────
  const [view, setView] = useState<View>('list');

  // Form fields (used for both register and edit).
  const [formUsername, setFormUsername] = useState('');
  const [formGroups, setFormGroups] = useState<string[]>([]);
  const [groupTagInput, setGroupTagInput] = useState('');
  const [formEmail, setFormEmail] = useState('');
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Register-only: optional companion creation in PingFederate's Simple PCV.
  // When the checkbox is on and the form is valid, we issue a second POST to
  // /api/iam/user-groups/{username}/pingfederate-user after the registry
  // record is created. The registry never persists this password.
  const [createInPingFederate, setCreateInPingFederate] = useState(false);
  const [pingfederatePassword, setPingfederatePassword] = useState('');

  // Edit-only: snapshot of the original record so we can compute the diff.
  const [editTarget, setEditTarget] = useState<IdPUserGroup | null>(null);

  // Per-row inline state.
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [togglingUsername, setTogglingUsername] = useState<string | null>(null);

  const groupInputRef = useRef<HTMLInputElement | null>(null);

  const resetForm = useCallback(() => {
    setFormUsername('');
    setFormGroups([]);
    setGroupTagInput('');
    setFormEmail('');
    setErrors({});
    setEditTarget(null);
    setCreateInPingFederate(false);
    setPingfederatePassword('');
  }, []);

  // ─── Group tag helpers ───────────────────────────────────────
  const addGroupTag = (raw: string) => {
    const value = raw.trim();
    if (!value) return;
    setFormGroups((prev) => (prev.includes(value) ? prev : [...prev, value]));
    setGroupTagInput('');
    if (errors.groups) setErrors((p) => ({ ...p, groups: undefined }));
  };

  const removeGroupTag = (value: string) => {
    setFormGroups((prev) => prev.filter((g) => g !== value));
  };

  const handleGroupKeyDown = (
    e: React.KeyboardEvent<HTMLInputElement>
  ) => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault();
      addGroupTag(groupTagInput);
    } else if (e.key === 'Backspace' && !groupTagInput && formGroups.length > 0) {
      // Pop the last tag for quick correction.
      e.preventDefault();
      setFormGroups((prev) => prev.slice(0, -1));
    }
  };

  // ─── Submit handlers ─────────────────────────────────────────
  const validate = (mode: 'register' | 'edit'): FormErrors => {
    const next: FormErrors = {};
    if (mode === 'register') {
      const trimmed = formUsername.trim();
      if (!trimmed) {
        next.username = 'Username is required';
      } else if (!USERNAME_REGEX.test(trimmed)) {
        next.username =
          'Allowed characters: letters, digits, _ - . @ (1-256 chars)';
      }
    }
    if (formEmail && formEmail.length > 512) {
      next.email = 'Email must be 512 characters or fewer';
    }
    // Register-only: when the user opted to also create the account in
    // PingFederate, the password is required and must meet the min length.
    if (mode === 'register' && createInPingFederate) {
      if (!pingfederatePassword) {
        next.pingfederatePassword = 'Password is required';
      } else if (pingfederatePassword.length < PINGFEDERATE_PASSWORD_MIN_LENGTH) {
        next.pingfederatePassword = `Password must be at least ${PINGFEDERATE_PASSWORD_MIN_LENGTH} characters`;
      }
    }
    return next;
  };

  const handleRegister = async () => {
    // If the user typed a value into the tag input but didn't press Enter,
    // accept it as a tag before submitting.
    if (groupTagInput.trim()) addGroupTag(groupTagInput);

    const validationErrors = validate('register');
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) return;

    setIsSubmitting(true);
    // Step A: create the registry-side idp_user_groups record. If this fails,
    // we surface the error and bail; no PingFederate call is attempted.
    let registryRecordCreated = false;
    const username = formUsername.trim();
    try {
      const payload: CreateUserGroupPayload = {
        username,
        groups: formGroups,
      };
      const trimmedEmail = formEmail.trim();
      if (trimmedEmail) payload.email = trimmedEmail;

      await createUserGroup(payload);
      registryRecordCreated = true;
    } catch (err: any) {
      onShowToast(
        extractDetail(err, 'Failed to register user group mapping'),
        'error'
      );
      setIsSubmitting(false);
      return;
    }

    // Step B: optional companion creation in PingFederate's Simple PCV.
    // A failure here is a partial-success: the registry record exists, so we
    // surface a distinct toast that points the user at the row for retry.
    if (registryRecordCreated && createInPingFederate) {
      try {
        await createPingFederateUser(username, pingfederatePassword);
        onShowToast(
          `"${username}" created in registry and PingFederate`,
          'success'
        );
      } catch (err: any) {
        const detail = extractDetail(err, 'unknown error');
        onShowToast(
          `Registry record created, but PingFederate user creation failed: ${detail}. You can retry from the user's row.`,
          'error'
        );
      }
    } else if (registryRecordCreated) {
      onShowToast(
        `User group mapping for "${username}" created`,
        'success'
      );
    }

    resetForm();
    setView('list');
    await refetch();
    setIsSubmitting(false);
  };

  const handleEditOpen = (record: IdPUserGroup) => {
    setEditTarget(record);
    setFormUsername(record.username);
    setFormGroups([...(record.groups || [])]);
    setFormEmail(record.email || '');
    setGroupTagInput('');
    setErrors({});
    setView('edit');
  };

  const handleUpdate = async () => {
    if (!editTarget) return;
    if (groupTagInput.trim()) addGroupTag(groupTagInput);

    const validationErrors = validate('edit');
    setErrors(validationErrors);
    if (Object.keys(validationErrors).length > 0) return;

    // Build a patch with only fields that actually changed (mirrors the
    // backend's model_dump(exclude_unset=True) semantics).
    const patch: PatchUserGroupPayload = {};

    const originalGroups = editTarget.groups || [];
    const groupsChanged =
      originalGroups.length !== formGroups.length ||
      originalGroups.some((g, i) => g !== formGroups[i]);
    if (groupsChanged) patch.groups = formGroups;

    const trimmedEmail = formEmail.trim();
    const originalEmail = editTarget.email || '';
    if (trimmedEmail !== originalEmail) {
      patch.email = trimmedEmail || null;
    }

    if (Object.keys(patch).length === 0) {
      onShowToast('No changes to save', 'info');
      return;
    }

    setIsSubmitting(true);
    try {
      await updateUserGroup(editTarget.username, patch);
      onShowToast(`Updated "${editTarget.username}"`, 'success');
      resetForm();
      setView('list');
      await refetch();
    } catch (err: any) {
      onShowToast(
        extractDetail(err, 'Failed to update user group mapping'),
        'error'
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (username: string) => {
    try {
      await deleteUserGroup(username);
      onShowToast('User group mapping deleted', 'success');
    } catch (err: any) {
      onShowToast(
        extractDetail(err, 'Failed to delete user group mapping'),
        'error'
      );
      throw err;
    }
    setDeleteTarget(null);
    await refetch();
  };

  const handleToggleEnabled = async (record: IdPUserGroup) => {
    if (record.provider !== 'manual') return;
    setTogglingUsername(record.username);
    try {
      await updateUserGroup(record.username, { enabled: !record.enabled });
      onShowToast(
        `"${record.username}" ${record.enabled ? 'disabled' : 'enabled'}`,
        'success'
      );
      await refetch();
    } catch (err: any) {
      onShowToast(extractDetail(err, 'Failed to toggle enabled state'), 'error');
    } finally {
      setTogglingUsername(null);
    }
  };

  const formatDate = (iso?: string): string => {
    if (!iso) return EM_DASH;
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleString();
    } catch {
      return iso;
    }
  };

  // ─── Edit View ───────────────────────────────────────────────
  if (view === 'edit' && editTarget) {
    return renderForm({
      title: `IAM > User Groups > Edit "${editTarget.username}"`,
      submitLabel: isSubmitting ? 'Updating...' : 'Save Changes',
      onSubmit: handleUpdate,
      onCancel: () => {
        resetForm();
        setView('list');
      },
      isSubmitting,
      mode: 'edit',
    });
  }

  // ─── Register View ───────────────────────────────────────────
  if (view === 'register') {
    return renderForm({
      title: 'IAM > User Groups > Register User Group',
      submitLabel: isSubmitting ? 'Registering...' : 'Register',
      onSubmit: handleRegister,
      onCancel: () => {
        resetForm();
        setView('list');
      },
      isSubmitting,
      mode: 'register',
    });
  }

  // ─── List View ───────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          IAM &gt; User Groups
        </h2>
        <div className="flex items-center space-x-2">
          <button
            onClick={refetch}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            title="Refresh"
            aria-label="Refresh"
          >
            <ArrowPathIcon className="h-5 w-5" />
          </button>
          <button
            onClick={() => {
              resetForm();
              setView('register');
            }}
            className="flex items-center px-3 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700"
            title="Register a user-to-group mapping for an IdP that does not carry groups in JWTs"
          >
            <PlusIcon className="h-4 w-4 mr-1" /> Register User Group
          </button>
        </div>
      </div>

      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-3 text-sm text-gray-700 dark:text-gray-300">
        These records back the auth server's user-to-group fallback for IdPs
        that do not carry group memberships in JWTs (for example
        PingFederate). The auth server reads from this collection when the
        JWT's groups claim is empty for a configured fallback provider.
      </div>

      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          value={searchInput}
          onChange={(e) => setSearchInput(e.target.value)}
          placeholder="Search by username..."
          className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
      </div>

      {isLoading && (
        <div className="flex justify-center py-12">
          <ArrowPathIcon className="h-6 w-6 text-gray-400 animate-spin" />
        </div>
      )}
      {error && !isLoading && (
        <div className="text-center py-8 text-red-500 dark:text-red-400 text-sm">
          {error}
        </div>
      )}
      {!isLoading && !error && items.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400 text-sm max-w-xl mx-auto">
          {debouncedSearch
            ? `No user-to-group mappings match "${debouncedSearch}".`
            : "No user-to-group mappings yet. These are used for IdPs that don't carry group memberships in JWTs (e.g. PingFederate). Click 'Register User Group' to add one."}
        </div>
      )}

      {!isLoading && !error && items.length > 0 && (
        <>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200 dark:border-gray-700">
                  <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Username
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Groups
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Provider
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Enabled
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Email
                  </th>
                  <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Registered by
                  </th>
                  <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.map((u) => {
                  const isManual = u.provider === 'manual';
                  return (
                    <React.Fragment key={u.username}>
                      <tr className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                        <td className="py-3 px-4 text-gray-900 dark:text-white font-medium font-mono">
                          {u.username}
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex flex-wrap gap-1">
                            {(u.groups || []).map((g) => (
                              <span
                                key={g}
                                className="inline-block px-2 py-0.5 text-xs rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300"
                              >
                                {g}
                              </span>
                            ))}
                            {(!u.groups || u.groups.length === 0) && (
                              <span className="text-gray-400 text-xs">
                                {EM_DASH}
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <ProviderBadge provider={u.provider} />
                        </td>
                        <td className="py-3 px-4">
                          <button
                            onClick={() => handleToggleEnabled(u)}
                            disabled={!isManual || togglingUsername === u.username}
                            aria-disabled={!isManual}
                            aria-label={
                              u.enabled ? 'Disable mapping' : 'Enable mapping'
                            }
                            title={
                              isManual
                                ? u.enabled
                                  ? 'Click to disable'
                                  : 'Click to enable'
                                : 'Managed by IdP sync; cannot toggle here'
                            }
                            className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors ${
                              u.enabled
                                ? 'bg-purple-600'
                                : 'bg-gray-300 dark:bg-gray-600'
                            } ${
                              !isManual || togglingUsername === u.username
                                ? 'opacity-50 cursor-not-allowed'
                                : 'hover:opacity-80'
                            }`}
                          >
                            <span
                              className={`inline-block h-3 w-3 transform rounded-full bg-white transition-transform ${
                                u.enabled ? 'translate-x-5' : 'translate-x-1'
                              }`}
                            />
                          </button>
                        </td>
                        <td className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400">
                          {u.email || (
                            <span className="text-gray-400">{EM_DASH}</span>
                          )}
                        </td>
                        <td
                          className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400"
                          title={
                            u.created_at
                              ? `Created at ${formatDate(u.created_at)}`
                              : undefined
                          }
                        >
                          {u.created_by || (
                            <span className="text-gray-400">{EM_DASH}</span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-right">
                          <div className="flex items-center justify-end space-x-2">
                            <button
                              onClick={() => handleEditOpen(u)}
                              disabled={!isManual}
                              aria-disabled={!isManual}
                              className="p-1 text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 disabled:opacity-40 disabled:cursor-not-allowed"
                              title={
                                isManual
                                  ? 'Edit mapping'
                                  : 'Managed by IdP sync; cannot edit here'
                              }
                              aria-label="Edit mapping"
                            >
                              <PencilIcon className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => setDeleteTarget(u.username)}
                              disabled={!isManual}
                              aria-disabled={!isManual}
                              className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 disabled:opacity-40 disabled:cursor-not-allowed"
                              title={
                                isManual
                                  ? 'Delete mapping'
                                  : 'Managed by IdP sync; cannot delete here'
                              }
                              aria-label="Delete mapping"
                            >
                              <TrashIcon className="h-4 w-4" />
                            </button>
                          </div>
                        </td>
                      </tr>
                      {deleteTarget === u.username && (
                        <tr>
                          <td colSpan={7} className="p-2">
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
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex justify-end pt-2">
            <Pagination
              currentPage={page}
              totalPages={totalPages}
              totalItems={total}
              pageSize={PAGE_SIZE}
              onPageChange={setPage}
            />
          </div>
        </>
      )}
    </div>
  );

  // ─── Inline form renderer ────────────────────────────────────
  // Defined as a closure so it can read the form state & handlers above
  // without prop-drilling. Returned from each branch above.
  function renderForm(opts: {
    title: string;
    submitLabel: string;
    onSubmit: () => void;
    onCancel: () => void;
    isSubmitting: boolean;
    mode: 'register' | 'edit';
  }) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            {opts.title}
          </h2>
          <button
            onClick={opts.onCancel}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" /> Back to List
          </button>
        </div>

        <div className="space-y-4 max-w-lg">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
              Username *
            </label>
            <input
              type="text"
              value={formUsername}
              disabled={opts.mode === 'edit'}
              onChange={(e) => {
                setFormUsername(e.target.value);
                if (errors.username) {
                  setErrors((p) => ({ ...p, username: undefined }));
                }
              }}
              placeholder="e.g. alice@example.com"
              className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white font-mono focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
                errors.username
                  ? 'border-red-500'
                  : 'border-gray-300 dark:border-gray-600'
              } ${opts.mode === 'edit' ? 'opacity-60 cursor-not-allowed' : ''}`}
            />
            {errors.username && (
              <p className="mt-1 text-sm text-red-500">{errors.username}</p>
            )}
            {opts.mode === 'edit' && (
              <p className="mt-1 text-xs text-gray-400">
                Username is the primary key and cannot be edited.
              </p>
            )}
          </div>

          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
              Groups
            </label>
            <div
              className={`flex flex-wrap items-center gap-1 px-2 py-1.5 border rounded-lg bg-white dark:bg-gray-900 focus-within:ring-2 focus-within:ring-purple-500 ${
                errors.groups
                  ? 'border-red-500'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
              onClick={() => groupInputRef.current?.focus()}
            >
              {formGroups.map((g) => (
                <span
                  key={g}
                  className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300"
                >
                  {g}
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      removeGroupTag(g);
                    }}
                    className="hover:text-purple-900 dark:hover:text-purple-100"
                    aria-label={`Remove ${g}`}
                  >
                    <XMarkIcon className="h-3 w-3" />
                  </button>
                </span>
              ))}
              <input
                ref={groupInputRef}
                type="text"
                value={groupTagInput}
                onChange={(e) => setGroupTagInput(e.target.value)}
                onKeyDown={handleGroupKeyDown}
                onBlur={() => {
                  if (groupTagInput.trim()) addGroupTag(groupTagInput);
                }}
                placeholder={
                  formGroups.length === 0
                    ? 'Type a group name and press Enter'
                    : ''
                }
                className="flex-1 min-w-[8rem] px-1 py-0.5 bg-transparent text-sm text-gray-900 dark:text-white outline-none"
              />
            </div>
            {errors.groups && (
              <p className="mt-1 text-sm text-red-500">{errors.groups}</p>
            )}
            <p className="mt-1 text-xs text-gray-400">
              Press Enter or comma to add a tag. Backspace on an empty input
              removes the last tag.
            </p>
          </div>

          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
              Email (optional)
            </label>
            <input
              type="email"
              value={formEmail}
              onChange={(e) => {
                setFormEmail(e.target.value);
                if (errors.email) setErrors((p) => ({ ...p, email: undefined }));
              }}
              placeholder="alice@example.com"
              className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
                errors.email
                  ? 'border-red-500'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
            />
            {errors.email && (
              <p className="mt-1 text-sm text-red-500">{errors.email}</p>
            )}
          </div>

          {/*
            Issue #1127: optional companion creation in PingFederate's Simple
            PCV. Only available on Register (Edit cannot create a brand-new
            PCV entry) and only when the backend reports that PingFederate
            user management is enabled.
          */}
          {opts.mode === 'register' && pingfederateUserManagementEnabled && (
            <div className="space-y-3">
              <label className="flex items-start space-x-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={createInPingFederate}
                  onChange={(e) => {
                    setCreateInPingFederate(e.target.checked);
                    if (!e.target.checked) {
                      // Clear password + any related error when the user
                      // turns the option off so the form is in a clean state
                      // if they toggle it on again later.
                      setPingfederatePassword('');
                      if (errors.pingfederatePassword) {
                        setErrors((p) => ({
                          ...p,
                          pingfederatePassword: undefined,
                        }));
                      }
                    }
                  }}
                  className="mt-0.5 h-4 w-4 rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500"
                />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  Also create this user in PingFederate (Simple PCV)
                </span>
              </label>

              {createInPingFederate && (
                <div>
                  <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
                    PingFederate password *
                  </label>
                  <input
                    type="password"
                    value={pingfederatePassword}
                    onChange={(e) => {
                      setPingfederatePassword(e.target.value);
                      if (errors.pingfederatePassword) {
                        setErrors((p) => ({
                          ...p,
                          pingfederatePassword: undefined,
                        }));
                      }
                    }}
                    minLength={PINGFEDERATE_PASSWORD_MIN_LENGTH}
                    autoComplete="new-password"
                    className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
                      errors.pingfederatePassword
                        ? 'border-red-500'
                        : 'border-gray-300 dark:border-gray-600'
                    }`}
                  />
                  {errors.pingfederatePassword && (
                    <p className="mt-1 text-sm text-red-500">
                      {errors.pingfederatePassword}
                    </p>
                  )}
                  <p className="mt-1 text-xs text-gray-400">
                    Used to create the user in PingFederate's Simple Password
                    Credential Validator. The registry never stores this
                    password.
                  </p>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={opts.onCancel}
            className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
          >
            Cancel
          </button>
          <button
            onClick={opts.onSubmit}
            disabled={opts.isSubmitting}
            className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {opts.submitLabel}
          </button>
        </div>
      </div>
    );
  }
};

export default IAMUserGroups;
