import React, { useState } from 'react';
import axios from 'axios';
import DetailsModal from './DetailsModal';


/**
 * Source types supported by this modal.
 */
export type RegistrySourceType = 'aws_registry' | 'anthropic' | 'asor';


/**
 * Props for the AddRegistryEntryModal component.
 */
interface AddRegistryEntryModalProps {
  isOpen: boolean;
  onClose: () => void;
  sourceType: RegistrySourceType;
  onSuccess: () => void;
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


/**
 * Form data for AWS Registry source.
 */
interface AwsRegistryFormData {
  registry_id: string;
  aws_account_id: string;
  aws_region: string;
  assume_role_arn: string;
  descriptor_types: string[];
  sync_status_filter: string;
}


/**
 * All available descriptor types for AWS Registry.
 */
const ALL_DESCRIPTOR_TYPES = ['MCP', 'A2A', 'CUSTOM', 'AGENT_SKILLS'];


/**
 * Source type display labels.
 */
const SOURCE_TITLES: Record<RegistrySourceType, string> = {
  aws_registry: 'Add AWS Agent Registry',
  anthropic: 'Add Anthropic Server',
  asor: 'Add ASOR Agent',
};


/**
 * CSS classes for form inputs.
 */
const INPUT_CLASS =
  'w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg ' +
  'bg-white dark:bg-gray-900 text-gray-900 dark:text-white ' +
  'focus:ring-2 focus:ring-purple-500 focus:border-transparent ' +
  'placeholder-gray-400 dark:placeholder-gray-500 text-sm';


/**
 * CSS classes for form labels.
 */
const LABEL_CLASS = 'block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1';


/**
 * Default form data for AWS Registry.
 */
function _defaultAwsFormData(): AwsRegistryFormData {
  return {
    registry_id: '',
    aws_account_id: '',
    aws_region: '',
    assume_role_arn: '',
    descriptor_types: [...ALL_DESCRIPTOR_TYPES],
    sync_status_filter: 'APPROVED',
  };
}


/**
 * Modal for adding a new entry to any federation source.
 *
 * Renders different form fields based on sourceType:
 * - aws_registry: multi-field form for AWS Agent Registry
 * - anthropic: single field for server name
 * - asor: single field for agent ID
 */
const AddRegistryEntryModal: React.FC<AddRegistryEntryModalProps> = ({
  isOpen,
  onClose,
  sourceType,
  onSuccess,
  onShowToast,
}) => {
  // Simple string fields for Anthropic/ASOR
  const [simpleValue, setSimpleValue] = useState('');

  // Multi-field form for AWS Registry
  const [awsForm, setAwsForm] = useState<AwsRegistryFormData>(_defaultAwsFormData());

  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);


  /**
   * Reset all form state when closing.
   */
  const handleClose = () => {
    setSimpleValue('');
    setAwsForm(_defaultAwsFormData());
    setErrors({});
    setIsSubmitting(false);
    onClose();
  };


  /**
   * Extract region and account ID from an ARN string.
   * ARN format: arn:aws:bedrock-agentcore:<region>:<account_id>:registry/...
   * Returns extracted values as soon as enough colon-separated parts are present.
   */
  const extractFromArn = (arn: string): { region: string; accountId: string } | null => {
    const trimmed = arn.trim();
    if (!trimmed.startsWith('arn:')) return null;

    const parts = trimmed.split(':');
    // parts[3] = region, parts[4] = account_id
    const region = parts.length > 3 ? parts[3] : '';
    const accountId = parts.length > 4 ? parts[4] : '';

    // Only return if we have at least one useful value
    if (region || accountId) {
      return { region, accountId };
    }
    return null;
  };


  /**
   * Handle changes to AWS Registry form fields.
   * Auto-populates region and account ID when registry_id is an ARN.
   */
  const handleAwsChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;

    if (name === 'registry_id') {
      const extracted = extractFromArn(value);
      setAwsForm((prev) => ({
        ...prev,
        registry_id: value,
        aws_region: extracted?.region ?? prev.aws_region,
        aws_account_id: extracted?.accountId ?? prev.aws_account_id,
      }));
    } else {
      setAwsForm((prev) => ({ ...prev, [name]: value }));
    }

    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
  };


  /**
   * Toggle a descriptor type checkbox.
   */
  const handleDescriptorToggle = (dtype: string) => {
    setAwsForm((prev) => {
      const current = prev.descriptor_types;
      const updated = current.includes(dtype)
        ? current.filter((d) => d !== dtype)
        : [...current, dtype];
      return { ...prev, descriptor_types: updated };
    });
  };


  /**
   * Validate the form before submission.
   */
  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (sourceType === 'anthropic') {
      if (!simpleValue.trim()) {
        newErrors.server_name = 'Server name is required';
      }
    } else if (sourceType === 'asor') {
      if (!simpleValue.trim()) {
        newErrors.agent_id = 'Agent ID is required';
      }
    } else if (sourceType === 'aws_registry') {
      if (!awsForm.registry_id.trim()) {
        newErrors.registry_id = 'Registry ID is required';
      }
      if (awsForm.descriptor_types.length === 0) {
        newErrors.descriptor_types = 'At least one descriptor type is required';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };


  /**
   * Submit the form to add a new entry.
   */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsSubmitting(true);
    try {
      if (sourceType === 'anthropic') {
        await axios.post(
          `/api/federation/config/default/anthropic/servers?server_name=${encodeURIComponent(simpleValue.trim())}`
        );
        onShowToast(`Server '${simpleValue.trim()}' added`, 'success');
      } else if (sourceType === 'asor') {
        await axios.post(
          `/api/federation/config/default/asor/agents?agent_id=${encodeURIComponent(simpleValue.trim())}`
        );
        onShowToast(`Agent '${simpleValue.trim()}' added`, 'success');
      } else if (sourceType === 'aws_registry') {
        const payload: Record<string, any> = {
          registry_id: awsForm.registry_id.trim(),
          descriptor_types: awsForm.descriptor_types,
          sync_status_filter: awsForm.sync_status_filter,
        };
        // Only include optional fields if filled
        if (awsForm.aws_account_id.trim()) {
          payload.aws_account_id = awsForm.aws_account_id.trim();
        }
        if (awsForm.aws_region.trim()) {
          payload.aws_region = awsForm.aws_region.trim();
        }
        if (awsForm.assume_role_arn.trim()) {
          payload.assume_role_arn = awsForm.assume_role_arn.trim();
        }
        await axios.post('/api/federation/config/default/aws_registry/registries', payload);
        onShowToast(`Registry '${awsForm.registry_id.trim()}' added`, 'success');
      }
      handleClose();
      onSuccess();
    } catch (err: any) {
      const detail = err?.response?.data?.detail || 'Failed to add entry';
      onShowToast(detail, 'error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const maxWidth = sourceType === 'aws_registry' ? 'lg' : 'md';

  return (
    <DetailsModal
      title={SOURCE_TITLES[sourceType]}
      isOpen={isOpen}
      onClose={handleClose}
      maxWidth={maxWidth as any}
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Anthropic: single server_name field */}
        {sourceType === 'anthropic' && (
          <div>
            <label className={LABEL_CLASS}>Server Name</label>
            <input
              type="text"
              value={simpleValue}
              onChange={(e) => {
                setSimpleValue(e.target.value);
                if (errors.server_name) setErrors((prev) => ({ ...prev, server_name: '' }));
              }}
              disabled={isSubmitting}
              className={INPUT_CLASS}
              placeholder="io.github.owner/server-name"
              autoFocus
            />
            {errors.server_name && (
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">{errors.server_name}</p>
            )}
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              The server identifier from the Anthropic MCP Registry
            </p>
          </div>
        )}

        {/* ASOR: single agent_id field */}
        {sourceType === 'asor' && (
          <div>
            <label className={LABEL_CLASS}>Agent ID</label>
            <input
              type="text"
              value={simpleValue}
              onChange={(e) => {
                setSimpleValue(e.target.value);
                if (errors.agent_id) setErrors((prev) => ({ ...prev, agent_id: '' }));
              }}
              disabled={isSubmitting}
              className={INPUT_CLASS}
              placeholder="my_agent_id"
              autoFocus
            />
            {errors.agent_id && (
              <p className="text-sm text-red-600 dark:text-red-400 mt-1">{errors.agent_id}</p>
            )}
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              The agent identifier from the ASOR registry
            </p>
          </div>
        )}

        {/* AWS Registry: multi-field form */}
        {sourceType === 'aws_registry' && (
          <>
            {/* Registry ID (required) */}
            <div>
              <label className={LABEL_CLASS}>
                Registry ID <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                name="registry_id"
                value={awsForm.registry_id}
                onChange={handleAwsChange}
                disabled={isSubmitting}
                className={INPUT_CLASS}
                placeholder="arn:aws:bedrock-agentcore:us-east-1:123456789012:registry/rXXXXXXXX"
                autoFocus
              />
              {errors.registry_id && (
                <p className="text-sm text-red-600 dark:text-red-400 mt-1">{errors.registry_id}</p>
              )}
            </div>

            {/* Two-column layout for optional fields */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={LABEL_CLASS}>AWS Account ID</label>
                <input
                  type="text"
                  name="aws_account_id"
                  value={awsForm.aws_account_id}
                  onChange={handleAwsChange}
                  disabled={isSubmitting}
                  className={INPUT_CLASS}
                  placeholder="123456789012"
                />
              </div>
              <div>
                <label className={LABEL_CLASS}>AWS Region</label>
                <input
                  type="text"
                  name="aws_region"
                  value={awsForm.aws_region}
                  onChange={handleAwsChange}
                  disabled={isSubmitting}
                  className={INPUT_CLASS}
                  placeholder="us-east-1"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Leave empty to use the global region
                </p>
              </div>
            </div>

            {/* Assume Role ARN */}
            <div>
              <label className={LABEL_CLASS}>
                Assume Role ARN <span className="text-gray-400 font-normal">(optional)</span>
              </label>
              <input
                type="text"
                name="assume_role_arn"
                value={awsForm.assume_role_arn}
                onChange={handleAwsChange}
                disabled={isSubmitting}
                className={INPUT_CLASS}
                placeholder="arn:aws:iam::123456789012:role/FederationReadOnly"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Only needed if adding a registry from a different AWS account
              </p>
            </div>

            {/* Descriptor Types checkboxes */}
            <div>
              <label className={LABEL_CLASS}>Descriptor Types</label>
              <div className="flex flex-wrap gap-3 mt-1">
                {ALL_DESCRIPTOR_TYPES.map((dtype) => (
                  <label
                    key={dtype}
                    className="inline-flex items-center space-x-2 cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={awsForm.descriptor_types.includes(dtype)}
                      onChange={() => handleDescriptorToggle(dtype)}
                      disabled={isSubmitting}
                      className="rounded border-gray-300 dark:border-gray-600
                                 text-purple-600 focus:ring-purple-500"
                    />
                    <span className="text-sm text-gray-700 dark:text-gray-300">{dtype}</span>
                  </label>
                ))}
              </div>
              {errors.descriptor_types && (
                <p className="text-sm text-red-600 dark:text-red-400 mt-1">
                  {errors.descriptor_types}
                </p>
              )}
            </div>

            {/* Sync Status Filter */}
            <div>
              <label className={LABEL_CLASS}>Sync Status Filter</label>
              <select
                name="sync_status_filter"
                value={awsForm.sync_status_filter}
                onChange={handleAwsChange}
                disabled={isSubmitting}
                className={INPUT_CLASS}
              >
                <option value="APPROVED">APPROVED</option>
                <option value="PENDING">PENDING</option>
                <option value="REJECTED">REJECTED</option>
              </select>
            </div>
          </>
        )}

        {/* Action buttons */}
        <div className="flex justify-end space-x-3 pt-2">
          <button
            type="button"
            onClick={handleClose}
            disabled={isSubmitting}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300
                       bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
                       rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700
                       disabled:opacity-50 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={isSubmitting}
            className="px-4 py-2 text-sm font-medium text-white bg-purple-600
                       rounded-lg hover:bg-purple-700 disabled:opacity-50
                       disabled:cursor-not-allowed transition-colors"
          >
            {isSubmitting ? 'Adding...' : 'Add'}
          </button>
        </div>
      </form>
    </DetailsModal>
  );
};


export default AddRegistryEntryModal;
