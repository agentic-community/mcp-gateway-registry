import React, { useState, useMemo, useCallback } from 'react';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  ClipboardDocumentIcon,
  EyeIcon,
  EyeSlashIcon,
  PencilIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import {
  useM2MClients,
  useIAMGroups,
  createM2MAccount,
  registerM2MClient,
  patchM2MClient,
  deleteM2MClient,
  CreateM2MPayload,
  RegisterM2MClientPayload,
  PatchM2MClientPayload,
  M2MCredentials,
  M2MClient,
} from '../hooks/useIAM';
import DeleteConfirmation from './DeleteConfirmation';

interface IAMM2MProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

type View = 'list' | 'create' | 'credentials' | 'edit' | 'register';

interface FormErrors {
  name?: string;
  groups?: string;
}

interface RegisterFormErrors {
  clientId?: string;
  clientName?: string;
  groups?: string;
}

// Mirrors the backend regex at registry/schemas/idp_m2m_client.py:18.
const CLIENT_ID_REGEX = /^[A-Za-z0-9_\-.:]{1,256}$/;

const PROVIDER_STYLES: Record<string, string> = {
  manual: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300',
  okta: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  auth0: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  keycloak: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
  entra: 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300',
};

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
    return detail.map((d: any) => d?.msg).filter(Boolean).join(', ') || fallback;
  }
  return detail || fallback;
};

const IAMM2M: React.FC<IAMM2MProps> = ({ onShowToast }) => {
  const { clients, isLoading, error, refetch } = useM2MClients();
  const { groups } = useIAMGroups();
  const [searchQuery, setSearchQuery] = useState('');
  const [view, setView] = useState<View>('list');
  const [showCreateHelp, setShowCreateHelp] = useState(false);

  // Create form state (legacy "Create M2M Account" flow).
  const [formName, setFormName] = useState('');
  const [formDescription, setFormDescription] = useState('');
  const [formGroups, setFormGroups] = useState<Set<string>>(new Set());
  const [isCreating, setIsCreating] = useState(false);
  const [errors, setErrors] = useState<FormErrors>({});

  // Register form state (new "Register existing client" flow).
  const [registerClientId, setRegisterClientId] = useState('');
  const [registerClientName, setRegisterClientName] = useState('');
  const [registerDescription, setRegisterDescription] = useState('');
  const [registerGroups, setRegisterGroups] = useState<Set<string>>(new Set());
  const [isRegistering, setIsRegistering] = useState(false);
  const [registerErrors, setRegisterErrors] = useState<RegisterFormErrors>({});

  // Credentials display (legacy flow only).
  const [credentials, setCredentials] = useState<M2MCredentials | null>(null);
  const [showSecret, setShowSecret] = useState(false);

  // Delete state.
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // Edit state.
  const [editTarget, setEditTarget] = useState<M2MClient | null>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const filteredClients = useMemo(() => {
    if (!searchQuery) return clients;
    const q = searchQuery.toLowerCase();
    return clients.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.client_id.toLowerCase().includes(q)
    );
  }, [clients, searchQuery]);

  const resetCreateForm = useCallback(() => {
    setFormName('');
    setFormDescription('');
    setFormGroups(new Set());
    setErrors({});
  }, []);

  const resetRegisterForm = useCallback(() => {
    setRegisterClientId('');
    setRegisterClientName('');
    setRegisterDescription('');
    setRegisterGroups(new Set());
    setRegisterErrors({});
  }, []);

  const toggleCreateGroup = (groupName: string) => {
    setFormGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  };

  const toggleRegisterGroup = (groupName: string) => {
    setRegisterGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  };

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text);
      onShowToast(`${label} copied to clipboard`, 'info');
    } catch {
      onShowToast('Failed to copy to clipboard', 'error');
    }
  };

  const handleCreate = async () => {
    const newErrors: FormErrors = {};
    if (!formName.trim()) newErrors.name = 'Name is required';
    if (formGroups.size === 0) newErrors.groups = 'At least one group is required';
    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    setIsCreating(true);
    try {
      const payload: CreateM2MPayload = {
        name: formName.trim(),
        description: formDescription.trim() || undefined,
        groups: Array.from(formGroups),
      };
      const creds = await createM2MAccount(payload);
      setCredentials(creds);
      setView('credentials');
      onShowToast(`M2M account "${formName}" created`, 'success');
      resetCreateForm();
    } catch (err: any) {
      onShowToast(extractDetail(err, 'Failed to create M2M account'), 'error');
    } finally {
      setIsCreating(false);
    }
  };

  const handleRegister = async () => {
    const newErrors: RegisterFormErrors = {};
    const trimmedId = registerClientId.trim();
    if (!trimmedId) {
      newErrors.clientId = 'Client ID is required';
    } else if (!CLIENT_ID_REGEX.test(trimmedId)) {
      newErrors.clientId =
        'Allowed characters: letters, digits, _ - . : (1-256 chars)';
    }
    if (!registerClientName.trim()) {
      newErrors.clientName = 'Client name is required';
    }
    if (registerGroups.size === 0) {
      newErrors.groups = 'At least one group is required';
    }
    setRegisterErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    setIsRegistering(true);
    try {
      const payload: RegisterM2MClientPayload = {
        client_id: trimmedId,
        client_name: registerClientName.trim(),
        description: registerDescription.trim() || undefined,
        groups: Array.from(registerGroups),
      };
      await registerM2MClient(payload);
      onShowToast(
        `Registered M2M client "${payload.client_name}"`,
        'success'
      );
      resetRegisterForm();
      setView('list');
      await refetch();
    } catch (err: any) {
      onShowToast(
        extractDetail(err, 'Failed to register M2M client'),
        'error'
      );
    } finally {
      setIsRegistering(false);
    }
  };

  const handleDelete = async (clientId: string) => {
    try {
      await deleteM2MClient(clientId);
      onShowToast('M2M client deleted', 'success');
    } catch (err: any) {
      onShowToast(extractDetail(err, 'Failed to delete M2M client'), 'error');
      throw err;
    }
    setDeleteTarget(null);
    await refetch();
  };

  const handleEdit = (client: M2MClient) => {
    setEditTarget(client);
    setFormGroups(new Set(client.groups || []));
    setView('edit');
  };

  const handleUpdate = async () => {
    if (!editTarget) return;

    const newErrors: FormErrors = {};
    if (formGroups.size === 0) newErrors.groups = 'At least one group is required';
    setErrors(newErrors);
    if (Object.keys(newErrors).length > 0) return;

    setIsUpdating(true);
    try {
      const payload: PatchM2MClientPayload = {
        groups: Array.from(formGroups),
      };
      await patchM2MClient(editTarget.client_id, payload);
      onShowToast(`Groups updated for "${editTarget.name}"`, 'success');
      setEditTarget(null);
      setFormGroups(new Set());
      setView('list');
      await refetch();
    } catch (err: any) {
      onShowToast(extractDetail(err, 'Failed to update groups'), 'error');
    } finally {
      setIsUpdating(false);
    }
  };

  // ─── Credentials View (after legacy-flow creation) ────────────
  if (view === 'credentials' && credentials) {
    return (
      <div className="space-y-6">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          IAM &gt; M2M Accounts &gt; Credentials
        </h2>

        <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-6 space-y-4">
          <p className="text-sm font-medium text-green-800 dark:text-green-200">
            M2M Account Created Successfully
          </p>

          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs text-gray-500 dark:text-gray-400">Client ID</span>
                <p className="text-sm font-mono text-gray-900 dark:text-white">{credentials.client_id}</p>
              </div>
              <button onClick={() => copyToClipboard(credentials.client_id, 'Client ID')}
                className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200" title="Copy">
                <ClipboardDocumentIcon className="h-4 w-4" />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs text-gray-500 dark:text-gray-400">Client Secret</span>
                <p className="text-sm font-mono text-gray-900 dark:text-white">
                  {showSecret ? credentials.client_secret : '••••••••••••••••'}
                </p>
              </div>
              <div className="flex items-center space-x-1">
                <button onClick={() => setShowSecret(!showSecret)}
                  className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200" title={showSecret ? 'Hide' : 'Show'}>
                  {showSecret ? <EyeSlashIcon className="h-4 w-4" /> : <EyeIcon className="h-4 w-4" />}
                </button>
                <button onClick={() => copyToClipboard(credentials.client_secret, 'Client Secret')}
                  className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200" title="Copy">
                  <ClipboardDocumentIcon className="h-4 w-4" />
                </button>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded p-3">
            <p className="text-xs text-yellow-800 dark:text-yellow-200">
              Save these credentials now. The client secret cannot be retrieved later.
            </p>
          </div>
        </div>

        <button
          onClick={() => { setCredentials(null); setShowSecret(false); setView('list'); refetch(); }}
          className="flex items-center text-sm text-purple-600 dark:text-purple-400 hover:underline"
        >
          <ArrowLeftIcon className="h-4 w-4 mr-1" />
          Back to M2M Accounts List
        </button>
      </div>
    );
  }

  // ─── Edit View ────────────────────────────────────────────────
  if (view === 'edit' && editTarget) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; M2M Accounts &gt; Edit "{editTarget.name}"
          </h2>
          <button onClick={() => { setFormGroups(new Set()); setEditTarget(null); setErrors({}); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
            <ArrowLeftIcon className="h-4 w-4 mr-1" /> Back to List
          </button>
        </div>

        <div className="space-y-4 max-w-lg">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-2">Groups *</label>
            <div className={`space-y-2 max-h-48 overflow-y-auto rounded-lg p-3 ${
              errors.groups ? 'border-2 border-red-500' : 'border border-gray-200 dark:border-gray-700'
            }`}>
              {groups.length === 0 ? (
                <p className="text-xs text-gray-400">No groups available</p>
              ) : (
                groups.map((g) => (
                  <label key={g.name} className="flex items-center space-x-2 cursor-pointer">
                    <input type="checkbox" checked={formGroups.has(g.name)}
                      onChange={() => { toggleCreateGroup(g.name); if (errors.groups) setErrors((p) => ({ ...p, groups: undefined })); }}
                      className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{g.name}</span>
                  </label>
                ))
              )}
            </div>
            {errors.groups && <p className="mt-1 text-sm text-red-500">{errors.groups}</p>}
          </div>
        </div>

        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button onClick={() => { setFormGroups(new Set()); setEditTarget(null); setErrors({}); setView('list'); }}
            className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600">
            Cancel
          </button>
          <button onClick={handleUpdate} disabled={isUpdating}
            className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed">
            {isUpdating ? 'Updating...' : 'Update Groups'}
          </button>
        </div>
      </div>
    );
  }

  // ─── Create View (legacy flow) ────────────────────────────────
  if (view === 'create') {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; M2M Accounts &gt; Create
          </h2>
          <button onClick={() => { resetCreateForm(); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
            <ArrowLeftIcon className="h-4 w-4 mr-1" /> Back to List
          </button>
        </div>

        <div className="space-y-4 max-w-lg">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Name *</label>
            <input type="text" value={formName}
              onChange={(e) => { setFormName(e.target.value); if (errors.name) setErrors((p) => ({ ...p, name: undefined })); }}
              placeholder="e.g. ci-pipeline"
              className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
                errors.name ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
              }`} />
            {errors.name && <p className="mt-1 text-sm text-red-500">{errors.name}</p>}
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Description (optional)</label>
            <input type="text" value={formDescription} onChange={(e) => setFormDescription(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent" />
          </div>

          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-2">Groups *</label>
            <div className={`space-y-2 max-h-48 overflow-y-auto rounded-lg p-3 ${
              errors.groups ? 'border-2 border-red-500' : 'border border-gray-200 dark:border-gray-700'
            }`}>
              {groups.length === 0 ? (
                <p className="text-xs text-gray-400">No groups available</p>
              ) : (
                groups.map((g) => (
                  <label key={g.name} className="flex items-center space-x-2 cursor-pointer">
                    <input type="checkbox" checked={formGroups.has(g.name)}
                      onChange={() => { toggleCreateGroup(g.name); if (errors.groups) setErrors((p) => ({ ...p, groups: undefined })); }}
                      className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{g.name}</span>
                  </label>
                ))
              )}
            </div>
            {errors.groups && <p className="mt-1 text-sm text-red-500">{errors.groups}</p>}
          </div>
        </div>

        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button onClick={() => { resetCreateForm(); setView('list'); }}
            className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600">
            Cancel
          </button>
          <button onClick={handleCreate} disabled={isCreating}
            className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed">
            {isCreating ? 'Creating...' : 'Create Account'}
          </button>
        </div>
      </div>
    );
  }

  // ─── Register View (new flow for existing IdP clients) ────────
  if (view === 'register') {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; M2M Accounts &gt; Register Existing Client
          </h2>
          <button onClick={() => { resetRegisterForm(); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
            <ArrowLeftIcon className="h-4 w-4 mr-1" /> Back to List
          </button>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 text-sm text-gray-700 dark:text-gray-300">
          Register a <code className="font-mono">client_id</code> that already exists in your
          IdP so the registry can attach groups to it. No IdP Admin API call is made; supply
          the client secret to your application out-of-band.
        </div>

        <div className="space-y-4 max-w-lg">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Client ID *</label>
            <input type="text" value={registerClientId}
              onChange={(e) => { setRegisterClientId(e.target.value); if (registerErrors.clientId) setRegisterErrors((p) => ({ ...p, clientId: undefined })); }}
              placeholder="e.g. my-pipeline-client"
              className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white font-mono focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
                registerErrors.clientId ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
              }`} />
            {registerErrors.clientId && <p className="mt-1 text-sm text-red-500">{registerErrors.clientId}</p>}
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Client Name *</label>
            <input type="text" value={registerClientName}
              onChange={(e) => { setRegisterClientName(e.target.value); if (registerErrors.clientName) setRegisterErrors((p) => ({ ...p, clientName: undefined })); }}
              placeholder="Human-readable name"
              className={`w-full px-3 py-2 border rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent ${
                registerErrors.clientName ? 'border-red-500' : 'border-gray-300 dark:border-gray-600'
              }`} />
            {registerErrors.clientName && <p className="mt-1 text-sm text-red-500">{registerErrors.clientName}</p>}
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Description (optional)</label>
            <input type="text" value={registerDescription} onChange={(e) => setRegisterDescription(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-purple-500 focus:border-transparent" />
          </div>

          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-2">Groups *</label>
            <div className={`space-y-2 max-h-48 overflow-y-auto rounded-lg p-3 ${
              registerErrors.groups ? 'border-2 border-red-500' : 'border border-gray-200 dark:border-gray-700'
            }`}>
              {groups.length === 0 ? (
                <p className="text-xs text-gray-400">No groups available</p>
              ) : (
                groups.map((g) => (
                  <label key={g.name} className="flex items-center space-x-2 cursor-pointer">
                    <input type="checkbox" checked={registerGroups.has(g.name)}
                      onChange={() => { toggleRegisterGroup(g.name); if (registerErrors.groups) setRegisterErrors((p) => ({ ...p, groups: undefined })); }}
                      className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500" />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{g.name}</span>
                  </label>
                ))
              )}
            </div>
            {registerErrors.groups && <p className="mt-1 text-sm text-red-500">{registerErrors.groups}</p>}
          </div>
        </div>

        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button onClick={() => { resetRegisterForm(); setView('list'); }}
            className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600">
            Cancel
          </button>
          <button onClick={handleRegister} disabled={isRegistering}
            className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed">
            {isRegistering ? 'Registering...' : 'Register'}
          </button>
        </div>
      </div>
    );
  }

  // ─── List View ────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">IAM &gt; M2M Accounts</h2>
        <div className="flex items-center space-x-2">
          <button onClick={refetch} className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200" title="Refresh">
            <ArrowPathIcon className="h-5 w-5" />
          </button>
          <button onClick={() => setShowCreateHelp((v) => !v)}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
            title="Help: which button should I use?"
            aria-label="Help">
            <InformationCircleIcon className="h-5 w-5" />
          </button>
          <button onClick={() => { resetRegisterForm(); setView('register'); }}
            className="flex items-center px-3 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
            title="Register a client_id that already exists in your IdP (no IdP Admin API token required)">
            <PlusIcon className="h-4 w-4 mr-1" /> Register existing client
          </button>
          <button onClick={() => setView('create')}
            className="flex items-center px-3 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700"
            title="Create a new M2M account via the IdP (requires IdP Admin API token)">
            <PlusIcon className="h-4 w-4 mr-1" /> Create M2M Account
          </button>
        </div>
      </div>

      {showCreateHelp && (
        <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-blue-50 dark:bg-blue-900/20 p-4 text-sm text-gray-700 dark:text-gray-300 space-y-2">
          <p>
            <strong>Create M2M Account</strong> creates a new client application inside
            your IdP (Keycloak, Entra, Okta, Auth0) via its Admin API and returns the
            client secret. Use this if your registry has an IdP Admin API token
            configured.
          </p>
          <p>
            <strong>Register existing client</strong> records a <code className="font-mono">client_id</code>
            you have already created in your IdP so the registry can attach groups to
            it. No IdP Admin API call is made; supply the secret to your application
            out-of-band. Use this for Entra tenants without
            <code className="font-mono"> Application.ReadWrite.*</code>, Okta tenants
            without Admin API access, or any environment where you don't want the
            registry to manage IdP app lifecycle.
          </p>
        </div>
      )}

      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search M2M accounts..."
          className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white text-sm focus:ring-2 focus:ring-purple-500 focus:border-transparent" />
      </div>

      {isLoading && (
        <div className="flex justify-center py-12"><ArrowPathIcon className="h-6 w-6 text-gray-400 animate-spin" /></div>
      )}
      {error && !isLoading && (
        <div className="text-center py-8 text-red-500 dark:text-red-400 text-sm">{error}</div>
      )}
      {!isLoading && !error && filteredClients.length === 0 && (
        <div className="text-center py-12 text-gray-500 dark:text-gray-400">
          {searchQuery ? 'No accounts match your search.' : 'No M2M accounts yet. Create your first service account.'}
        </div>
      )}

      {!isLoading && !error && filteredClients.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Name</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Provider</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Groups</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Registered by</th>
                <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredClients.map((c) => {
                const isManual = c.provider === 'manual';
                return (
                  <React.Fragment key={c.client_id}>
                    <tr className="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-800/50">
                      <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">
                        <div>{c.name}</div>
                        <div className="text-xs text-gray-400 font-mono">{c.client_id}</div>
                      </td>
                      <td className="py-3 px-4">
                        <ProviderBadge provider={c.provider} />
                      </td>
                      <td className="py-3 px-4">
                        <div className="flex flex-wrap gap-1">
                          {(c.groups || []).map((g) => (
                            <span key={g} className="inline-block px-2 py-0.5 text-xs rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300">
                              {g}
                            </span>
                          ))}
                          {(!c.groups || c.groups.length === 0) && <span className="text-gray-400 text-xs">{'—'}</span>}
                        </div>
                      </td>
                      <td
                        className="py-3 px-4 text-sm text-gray-600 dark:text-gray-400"
                        title={c.created_at ? `Created at ${c.created_at}` : undefined}
                      >
                        {c.created_by || <span className="text-gray-400">{'—'}</span>}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <div className="flex items-center justify-end space-x-2">
                          <button
                            onClick={() => handleEdit(c)}
                            disabled={!isManual}
                            aria-disabled={!isManual}
                            className="p-1 text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 disabled:opacity-40 disabled:cursor-not-allowed"
                            title={isManual ? 'Edit groups' : 'Managed by IdP sync; cannot edit here'}
                          >
                            <PencilIcon className="h-4 w-4" />
                          </button>
                          <button
                            onClick={() => setDeleteTarget(c.client_id)}
                            disabled={!isManual}
                            aria-disabled={!isManual}
                            className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400 disabled:opacity-40 disabled:cursor-not-allowed"
                            title={isManual ? 'Delete account' : 'Managed by IdP sync; cannot delete here'}
                          >
                            <TrashIcon className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                    {deleteTarget === c.client_id && (
                      <tr>
                        <td colSpan={5} className="p-2">
                          <DeleteConfirmation
                            entityType="m2m"
                            entityName={c.name}
                            entityPath={c.client_id}
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
      )}
    </div>
  );
};

export default IAMM2M;
