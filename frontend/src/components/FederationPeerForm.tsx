import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ArrowPathIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import {
  useFederationPeer,
  createPeer,
  updatePeer,
  PeerFormData,
} from '../hooks/useFederationPeers';


/**
 * Props for the FederationPeerForm component.
 */
interface FederationPeerFormProps {
  peerId?: string;
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


/**
 * Form validation errors interface.
 */
interface FormErrors {
  peer_id?: string;
  name?: string;
  endpoint?: string;
  federation_token?: string;
  sync_interval_minutes?: string;
  whitelist?: string;
  tag_filters?: string;
}


/**
 * FederationPeerForm component for adding or editing a peer registry.
 *
 * Provides a form with validation for configuring peer connection settings,
 * authentication, and sync options.
 */
const FederationPeerForm: React.FC<FederationPeerFormProps> = ({
  peerId,
  onShowToast,
}) => {
  const navigate = useNavigate();
  const isEditMode = !!peerId;

  const { peer, isLoading: isLoadingPeer, error: loadError } = useFederationPeer(peerId);

  // Form state
  const [formData, setFormData] = useState<PeerFormData>({
    peer_id: '',
    name: '',
    endpoint: '',
    enabled: true,
    sync_mode: 'all',
    whitelist_servers: [],
    whitelist_agents: [],
    tag_filters: [],
    sync_interval_minutes: 60,
    sync_local_servers: false,
    federation_token: '',
  });

  // Whitelist and tags as comma-separated strings for easier editing
  const [whitelistText, setWhitelistText] = useState('');
  const [tagFiltersText, setTagFiltersText] = useState('');

  // Form state
  const [errors, setErrors] = useState<FormErrors>({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Populate form in edit mode
  useEffect(() => {
    if (peer) {
      setFormData({
        peer_id: peer.peer_id,
        name: peer.name,
        endpoint: peer.endpoint,
        enabled: peer.enabled,
        sync_mode: peer.sync_mode,
        whitelist_servers: peer.whitelist_servers || [],
        whitelist_agents: peer.whitelist_agents || [],
        tag_filters: peer.tag_filters || [],
        sync_interval_minutes: peer.sync_interval_minutes,
        sync_local_servers: peer.sync_local_servers ?? false,
        federation_token: '', // Don't populate token for security
      });

      // Combine whitelists for display
      const whitelistItems = [
        ...(peer.whitelist_servers || []).map((s) => `server:${s}`),
        ...(peer.whitelist_agents || []).map((a) => `agent:${a}`),
      ];
      setWhitelistText(whitelistItems.join(', '));
      setTagFiltersText((peer.tag_filters || []).join(', '));
    }
  }, [peer]);

  /**
   * Handle input field changes.
   */
  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>
  ) => {
    const { name, value, type } = e.target;
    const newValue = type === 'checkbox' ? (e.target as HTMLInputElement).checked : value;

    setFormData((prev) => ({
      ...prev,
      [name]: name === 'sync_interval_minutes' ? parseInt(value) || 60 : newValue,
    }));

    // Clear error for this field
    if (errors[name as keyof FormErrors]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  /**
   * Validate form data.
   */
  const validateForm = (): boolean => {
    const newErrors: FormErrors = {};

    // Peer ID validation
    if (!formData.peer_id.trim()) {
      newErrors.peer_id = 'Peer ID is required';
    } else if (!/^[a-zA-Z0-9-_]+$/.test(formData.peer_id)) {
      newErrors.peer_id = 'Peer ID must be alphanumeric with dashes or underscores only';
    }

    // Name validation
    if (!formData.name.trim()) {
      newErrors.name = 'Display name is required';
    }

    // Endpoint validation
    if (!formData.endpoint.trim()) {
      newErrors.endpoint = 'Endpoint URL is required';
    } else if (!formData.endpoint.startsWith('http://') && !formData.endpoint.startsWith('https://')) {
      newErrors.endpoint = 'Endpoint must be a valid HTTP or HTTPS URL';
    }

    // Token validation (required for new peers)
    if (!isEditMode && !formData.federation_token?.trim()) {
      newErrors.federation_token = 'Federation token is required';
    }

    // Sync interval validation
    if (formData.sync_interval_minutes < 5 || formData.sync_interval_minutes > 1440) {
      newErrors.sync_interval_minutes = 'Sync interval must be between 5 and 1440 minutes';
    }

    // Whitelist validation when sync_mode is 'whitelist'
    if (formData.sync_mode === 'whitelist') {
      const items = whitelistText.split(',').map((s) => s.trim()).filter(Boolean);
      if (items.length === 0) {
        newErrors.whitelist = 'At least one whitelist item is required';
      }
    }

    // Tag filter validation when sync_mode is 'tag_filter'
    if (formData.sync_mode === 'tag_filter') {
      const tags = tagFiltersText.split(',').map((s) => s.trim()).filter(Boolean);
      if (tags.length === 0) {
        newErrors.tag_filters = 'At least one tag is required';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handle form submission.
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      // Parse whitelist items
      const whitelistItems = whitelistText.split(',').map((s) => s.trim()).filter(Boolean);
      const whitelistServers: string[] = [];
      const whitelistAgents: string[] = [];

      for (const item of whitelistItems) {
        if (item.startsWith('server:')) {
          whitelistServers.push(item.substring(7));
        } else if (item.startsWith('agent:')) {
          whitelistAgents.push(item.substring(6));
        } else {
          // Default to server if no prefix
          whitelistServers.push(item);
        }
      }

      // Parse tag filters
      const tagFilters = tagFiltersText.split(',').map((s) => s.trim()).filter(Boolean);

      const payload: PeerFormData = {
        ...formData,
        whitelist_servers: whitelistServers,
        whitelist_agents: whitelistAgents,
        tag_filters: tagFilters,
      };

      // Don't send empty token on edit (keep existing)
      if (isEditMode && !payload.federation_token) {
        delete payload.federation_token;
      }

      if (isEditMode) {
        await updatePeer(peerId!, payload);
        onShowToast(`Peer "${formData.name}" has been updated`, 'success');
      } else {
        await createPeer(payload);
        onShowToast(`Peer "${formData.name}" has been added`, 'success');
      }

      navigate('/settings/federation/peers');
    } catch (err: any) {
      const errorMessage =
        err.response?.data?.detail ||
        err.message ||
        `Failed to ${isEditMode ? 'update' : 'create'} peer`;
      onShowToast(errorMessage, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Loading state for edit mode
  if (isEditMode && isLoadingPeer) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 bg-surface-hover rounded animate-pulse" />
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-16 bg-surface-hover rounded animate-pulse" />
          ))}
        </div>
      </div>
    );
  }

  // Error state for edit mode
  if (isEditMode && loadError) {
    return (
      <div className="text-center py-12">
        <ExclamationCircleIcon className="h-12 w-12 mx-auto text-red-500 mb-4" />
        <h3 className="text-lg font-medium text-fg mb-2">
          Failed to Load Peer
        </h3>
        <p className="text-fg-subtle mb-4">{loadError}</p>
        <button
          onClick={() => navigate('/settings/federation/peers')}
          className="btn-secondary"
        >
          Back to Peers
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-fg">
            {isEditMode ? 'Edit Peer' : 'Add Peer'}
          </h2>
          <p className="text-sm text-fg-subtle">
            {isEditMode
              ? 'Update peer registry configuration'
              : 'Configure a new peer registry for federation'}
          </p>
        </div>
        <button
          onClick={() => navigate('/settings/federation/peers')}
          className="flex items-center text-fg-muted hover:text-fg transition-colors"
        >
          <ArrowLeftIcon className="h-5 w-5 mr-2" />
          Back to List
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Basic Information */}
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-fg uppercase tracking-wider">
            Basic Information
          </h3>

          {/* Peer ID */}
          <div>
            <label
              htmlFor="peer_id"
              className="block text-sm font-medium text-fg-muted mb-1"
            >
              Peer ID <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              id="peer_id"
              name="peer_id"
              value={formData.peer_id}
              onChange={handleChange}
              disabled={isEditMode}
              placeholder="e.g., lob-a-registry"
              className={`input
                         ${errors.peer_id ? 'border-red-500' : ''}
                         ${isEditMode ? 'opacity-50 cursor-not-allowed' : ''}`}
            />
            {errors.peer_id && (
              <p className="mt-1 text-sm text-red-500">{errors.peer_id}</p>
            )}
            <p className="mt-1 text-xs text-fg-subtle">
              Unique identifier for this peer (alphanumeric, dashes, underscores)
            </p>
          </div>

          {/* Display Name */}
          <div>
            <label
              htmlFor="name"
              className="block text-sm font-medium text-fg-muted mb-1"
            >
              Display Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="e.g., LOB-A Registry"
              className={`input ${errors.name ? 'border-red-500' : ''}`}
            />
            {errors.name && (
              <p className="mt-1 text-sm text-red-500">{errors.name}</p>
            )}
          </div>

          {/* Endpoint URL */}
          <div>
            <label
              htmlFor="endpoint"
              className="block text-sm font-medium text-fg-muted mb-1"
            >
              Endpoint URL <span className="text-red-500">*</span>
            </label>
            <input
              type="url"
              id="endpoint"
              name="endpoint"
              value={formData.endpoint}
              onChange={handleChange}
              placeholder="https://lob-a-registry.company.com"
              className={`input ${errors.endpoint ? 'border-red-500' : ''}`}
            />
            {errors.endpoint && (
              <p className="mt-1 text-sm text-red-500">{errors.endpoint}</p>
            )}
            <p className="mt-1 text-xs text-fg-subtle">
              Base URL of the peer registry API
            </p>
          </div>

          {/* Enabled toggle */}
          <div className="flex items-center">
            <input
              type="checkbox"
              id="enabled"
              name="enabled"
              checked={formData.enabled}
              onChange={handleChange}
              className="h-4 w-4 rounded border-line-strong bg-surface text-purple-600 focus:ring-focus"
            />
            <label
              htmlFor="enabled"
              className="ml-2 text-sm text-fg-muted"
            >
              Enable sync from this peer
            </label>
          </div>
        </div>

        {/* Authentication */}
        <div className="space-y-4 pt-4 border-t border-line">
          <h3 className="text-sm font-medium text-fg uppercase tracking-wider">
            Authentication
          </h3>

          {/* Federation Token */}
          <div>
            <label
              htmlFor="federation_token"
              className="block text-sm font-medium text-fg-muted mb-1"
            >
              Federation Static Token {!isEditMode && <span className="text-red-500">*</span>}
            </label>
            <input
              type="password"
              id="federation_token"
              name="federation_token"
              value={formData.federation_token || ''}
              onChange={handleChange}
              placeholder={isEditMode ? '(leave blank to keep existing)' : 'Enter token from peer registry'}
              autoComplete="off"
              className={`input ${errors.federation_token ? 'border-red-500' : ''}`}
            />
            {errors.federation_token && (
              <p className="mt-1 text-sm text-red-500">{errors.federation_token}</p>
            )}
            <p className="mt-1 text-xs text-fg-subtle">
              {isEditMode
                ? 'Leave blank to keep existing token, or enter a new value to update'
                : 'The FEDERATION_STATIC_TOKEN value from the peer registry'}
            </p>
          </div>
        </div>

        {/* Sync Configuration */}
        <div className="space-y-4 pt-4 border-t border-line">
          <h3 className="text-sm font-medium text-fg uppercase tracking-wider">
            Sync Configuration
          </h3>

          {/* Sync Mode */}
          <div>
            <label
              htmlFor="sync_mode"
              className="block text-sm font-medium text-fg-muted mb-1"
            >
              Sync Mode
            </label>
            <select
              id="sync_mode"
              name="sync_mode"
              value={formData.sync_mode}
              onChange={handleChange}
              className="input"
            >
              <option value="all">All Public Items</option>
              <option value="whitelist">Whitelist Specific Items</option>
              <option value="tag_filter">Filter by Tags</option>
            </select>
          </div>

          {/* Whitelist (shown when sync_mode is 'whitelist') */}
          {formData.sync_mode === 'whitelist' && (
            <div>
              <label
                htmlFor="whitelist"
                className="block text-sm font-medium text-fg-muted mb-1"
              >
                Whitelist Items
              </label>
              <textarea
                id="whitelist"
                value={whitelistText}
                onChange={(e) => setWhitelistText(e.target.value)}
                placeholder="server:/finance-tools, agent:/code-reviewer"
                rows={3}
                className={`input ${errors.whitelist ? 'border-red-500' : ''}`}
              />
              {errors.whitelist && (
                <p className="mt-1 text-sm text-red-500">{errors.whitelist}</p>
              )}
              <p className="mt-1 text-xs text-fg-subtle">
                Comma-separated list. Prefix with "server:" or "agent:" (default: server)
              </p>
            </div>
          )}

          {/* Tag Filters (shown when sync_mode is 'tag_filter') */}
          {formData.sync_mode === 'tag_filter' && (
            <div>
              <label
                htmlFor="tag_filters"
                className="block text-sm font-medium text-fg-muted mb-1"
              >
                Tag Filters
              </label>
              <input
                type="text"
                id="tag_filters"
                value={tagFiltersText}
                onChange={(e) => setTagFiltersText(e.target.value)}
                placeholder="production, approved, finance"
                className={`input ${errors.tag_filters ? 'border-red-500' : ''}`}
              />
              {errors.tag_filters && (
                <p className="mt-1 text-sm text-red-500">{errors.tag_filters}</p>
              )}
              <p className="mt-1 text-xs text-fg-subtle">
                Comma-separated list of tags. Only items with these tags will be synced.
              </p>
            </div>
          )}

          {/* Sync Interval */}
          <div>
            <label
              htmlFor="sync_interval_minutes"
              className="block text-sm font-medium text-fg-muted mb-1"
            >
              Sync Interval (minutes)
            </label>
            <input
              type="number"
              id="sync_interval_minutes"
              name="sync_interval_minutes"
              value={formData.sync_interval_minutes}
              onChange={handleChange}
              min={5}
              max={1440}
              className={`input ${errors.sync_interval_minutes ? 'border-red-500' : ''}`}
            />
            {errors.sync_interval_minutes && (
              <p className="mt-1 text-sm text-red-500">{errors.sync_interval_minutes}</p>
            )}
            <p className="mt-1 text-xs text-fg-subtle">
              How often to sync from this peer (5-1440 minutes)
            </p>
          </div>

          {/* Accept local (stdio) servers from this peer */}
          <div className="md:col-span-2">
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                name="sync_local_servers"
                checked={!!formData.sync_local_servers}
                onChange={handleChange}
                className="mt-1 h-4 w-4 rounded border-line-strong bg-surface text-purple-600 focus:ring-focus"
              />
              <span className="text-sm">
                <span className="font-medium text-fg-muted">
                  Sync local (stdio) servers from this peer
                </span>
                <span className="block text-xs text-fg-subtle mt-0.5">
                  Local servers are executable launch recipes that run on developer machines.
                  Only enable for trusted peers. Default: off.
                </span>
              </span>
            </label>
          </div>
        </div>

        {/* Form Actions */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-line">
          <button
            type="button"
            onClick={() => navigate('/settings/federation/peers')}
            disabled={isSubmitting}
            className="btn-secondary"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting}
            className="btn-primary"
          >
            {isSubmitting && <ArrowPathIcon className="h-4 w-4 animate-spin" />}
            {isEditMode ? 'Save Changes' : 'Add Peer'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default FederationPeerForm;
