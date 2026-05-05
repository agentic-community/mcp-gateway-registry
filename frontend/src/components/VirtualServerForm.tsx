import React, { useState, useEffect, useMemo } from 'react';
import { XMarkIcon, PlusIcon, TrashIcon, CheckIcon } from '@heroicons/react/24/outline';
import {
  VirtualServerConfig,
  CreateVirtualServerRequest,
  UpdateVirtualServerRequest,
  ToolMapping,
} from '../types/virtualServer';
import ToolSelector from './ToolSelector';


/**
 * Props for the VirtualServerForm component.
 */
interface VirtualServerFormProps {
  virtualServer?: VirtualServerConfig | null;
  onSave: (
    data: CreateVirtualServerRequest | UpdateVirtualServerRequest,
  ) => Promise<void>;
  onCancel: () => void;
}


/**
 * Step definitions for the wizard.
 */
const STEPS = [
  { id: 'basics', label: 'Basics' },
  { id: 'tools', label: 'Tool Selection' },
  { id: 'config', label: 'Configuration' },
  { id: 'review', label: 'Review' },
] as const;

type StepId = typeof STEPS[number]['id'];


/**
 * Generate a URL path from a server name.
 */
function _generatePathFromName(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '');
  return `/virtual/${slug}`;
}


/**
 * VirtualServerForm provides a stepped wizard for creating or editing virtual
 * MCP servers.
 *
 * Steps: Basics -> Tool Selection -> Configuration -> Review
 *
 * In edit mode (when virtualServer prop is provided), the form is pre-populated
 * with existing data. In create mode, the path auto-generates from the name.
 */
const VirtualServerForm: React.FC<VirtualServerFormProps> = ({
  virtualServer,
  onSave,
  onCancel,
}) => {
  const isEditMode = !!virtualServer;

  const [currentStep, setCurrentStep] = useState<StepId>('basics');
  const [name, setName] = useState('');
  const [path, setPath] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [requiredScopes, setRequiredScopes] = useState('');
  const [toolMappings, setToolMappings] = useState<ToolMapping[]>([]);
  const [manualMappings, setManualMappings] = useState<
    Array<{ backend_server_path: string; tool_name: string; alias: string }>
  >([]);
  const [useToolSelector, setUseToolSelector] = useState(true);
  const [saving, setSaving] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [pathManuallyEdited, setPathManuallyEdited] = useState(false);

  const currentStepIndex = STEPS.findIndex((s) => s.id === currentStep);

  // Populate form when editing
  useEffect(() => {
    if (virtualServer) {
      setName(virtualServer.server_name);
      setPath(virtualServer.path);
      setDescription(virtualServer.description || '');
      setTags(virtualServer.tags?.join(', ') || '');
      setRequiredScopes(virtualServer.required_scopes?.join(', ') || '');
      setToolMappings(virtualServer.tool_mappings || []);
      setPathManuallyEdited(true);
    }
  }, [virtualServer]);

  // Auto-generate path from name in create mode
  useEffect(() => {
    if (!isEditMode && !pathManuallyEdited && name) {
      setPath(_generatePathFromName(name));
    }
  }, [name, isEditMode, pathManuallyEdited]);

  const handlePathChange = (value: string) => {
    setPath(value);
    setPathManuallyEdited(true);
  };

  const addManualMapping = () => {
    setManualMappings([
      ...manualMappings,
      { backend_server_path: '', tool_name: '', alias: '' },
    ]);
  };

  const removeManualMapping = (index: number) => {
    setManualMappings(manualMappings.filter((_, i) => i !== index));
  };

  const updateManualMapping = (
    index: number,
    field: 'backend_server_path' | 'tool_name' | 'alias',
    value: string,
  ) => {
    setManualMappings(
      manualMappings.map((m, i) =>
        i === index ? { ...m, [field]: value } : m,
      ),
    );
  };

  // Combine all tool mappings from both modes
  const allMappings: ToolMapping[] = useMemo(() => [
    ...toolMappings,
    ...manualMappings
      .filter((m) => m.backend_server_path && m.tool_name)
      .map((m) => ({
        tool_name: m.tool_name,
        backend_server_path: m.backend_server_path,
        alias: m.alias || null,
        backend_version: null,
      })),
  ], [toolMappings, manualMappings]);

  const parsedTags = useMemo(() =>
    tags.split(',').map((t) => t.trim()).filter(Boolean),
    [tags]
  );

  const parsedScopes = useMemo(() =>
    requiredScopes.split(',').map((s) => s.trim()).filter(Boolean),
    [requiredScopes]
  );

  // Validate current step before advancing
  const validateStep = (step: StepId): string | null => {
    if (step === 'basics') {
      if (!name.trim()) return 'Server name is required';
      if (!path.trim()) return 'Server path is required';
    }
    return null;
  };

  const goToNext = () => {
    const error = validateStep(currentStep);
    if (error) {
      setValidationError(error);
      return;
    }
    setValidationError(null);
    const nextIndex = currentStepIndex + 1;
    if (nextIndex < STEPS.length) {
      setCurrentStep(STEPS[nextIndex].id);
    }
  };

  const goToPrev = () => {
    setValidationError(null);
    const prevIndex = currentStepIndex - 1;
    if (prevIndex >= 0) {
      setCurrentStep(STEPS[prevIndex].id);
    }
  };

  const goToStep = (step: StepId) => {
    const targetIndex = STEPS.findIndex((s) => s.id === step);
    // Allow jumping backward freely, but forward only to visited steps
    if (targetIndex <= currentStepIndex) {
      setValidationError(null);
      setCurrentStep(step);
    }
  };

  const handleSubmit = async () => {
    setValidationError(null);

    const basicsError = validateStep('basics');
    if (basicsError) {
      setCurrentStep('basics');
      setValidationError(basicsError);
      return;
    }

    setSaving(true);
    try {
      if (isEditMode) {
        const updateData: UpdateVirtualServerRequest = {
          server_name: name.trim(),
          description: description.trim() || '',
          tool_mappings: allMappings,
          required_scopes: parsedScopes,
          tags: parsedTags,
        };
        await onSave(updateData);
      } else {
        const createData: CreateVirtualServerRequest = {
          server_name: name.trim(),
          path: path.trim(),
          description: description.trim(),
          tool_mappings: allMappings,
          required_scopes: parsedScopes,
          tags: parsedTags,
        };
        await onSave(createData);
      }
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } }; message?: string };
      setValidationError(
        axiosErr.response?.data?.detail ||
        axiosErr.message ||
        'Failed to save virtual server'
      );
    } finally {
      setSaving(false);
    }
  };

  // Render step indicator
  const renderStepIndicator = () => (
    <div className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
      {STEPS.map((step, index) => {
        const isActive = step.id === currentStep;
        const isCompleted = index < currentStepIndex;
        const isClickable = index <= currentStepIndex;

        return (
          <React.Fragment key={step.id}>
            {index > 0 && (
              <div className={`flex-1 h-0.5 mx-2 ${
                isCompleted ? 'bg-teal-500' : 'bg-gray-300 dark:bg-gray-600'
              }`} />
            )}
            <button
              type="button"
              onClick={() => isClickable && goToStep(step.id)}
              disabled={!isClickable}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300'
                  : isCompleted
                  ? 'text-teal-600 dark:text-teal-400 hover:bg-teal-50 dark:hover:bg-teal-900/20 cursor-pointer'
                  : 'text-gray-400 dark:text-gray-500 cursor-default'
              }`}
            >
              <span className={`flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold ${
                isActive
                  ? 'bg-teal-600 text-white'
                  : isCompleted
                  ? 'bg-teal-500 text-white'
                  : 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400'
              }`}>
                {isCompleted ? <CheckIcon className="h-3.5 w-3.5" /> : index + 1}
              </span>
              <span className="hidden sm:inline">{step.label}</span>
            </button>
          </React.Fragment>
        );
      })}
    </div>
  );

  // Step 1: Basics
  const renderBasicsStep = () => (
    <div className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Name <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. Dev Essentials"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-teal-500 focus:border-transparent"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Path <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          value={path}
          onChange={(e) => handlePathChange(e.target.value)}
          placeholder="/virtual/dev-essentials"
          disabled={isEditMode}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-teal-500 focus:border-transparent
                     disabled:opacity-50 disabled:cursor-not-allowed font-mono text-sm"
        />
        {!isEditMode && (
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Auto-generated from name. Must start with /virtual/.
          </p>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Description
        </label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what this virtual server provides..."
          rows={3}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-teal-500 focus:border-transparent"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Tags
        </label>
        <input
          type="text"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="development, tools, frontend (comma-separated)"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-teal-500 focus:border-transparent"
        />
      </div>
    </div>
  );

  // Step 2: Tool Selection
  const renderToolSelectionStep = () => (
    <div>
      <div className="flex items-center justify-between mb-3">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          Select tools to include in this virtual server
        </label>
        <button
          type="button"
          onClick={() => setUseToolSelector(!useToolSelector)}
          className="text-xs text-teal-600 dark:text-teal-400 hover:underline"
        >
          {useToolSelector ? 'Switch to manual entry' : 'Switch to tool picker'}
        </button>
      </div>

      {useToolSelector ? (
        <ToolSelector
          selectedTools={toolMappings}
          onToolsChange={setToolMappings}
        />
      ) : (
        <div className="space-y-3">
          {manualMappings.map((mapping, index) => (
            <div
              key={index}
              className="flex items-start gap-2 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg"
            >
              <div className="flex-1 space-y-2">
                <input
                  type="text"
                  value={mapping.backend_server_path}
                  onChange={(e) =>
                    updateManualMapping(index, 'backend_server_path', e.target.value)
                  }
                  placeholder="Backend server path (e.g. /github)"
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded
                             bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                />
                <input
                  type="text"
                  value={mapping.tool_name}
                  onChange={(e) =>
                    updateManualMapping(index, 'tool_name', e.target.value)
                  }
                  placeholder="Tool name"
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded
                             bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                />
                <input
                  type="text"
                  value={mapping.alias}
                  onChange={(e) =>
                    updateManualMapping(index, 'alias', e.target.value)
                  }
                  placeholder="Alias (optional)"
                  className="w-full px-2 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded
                             bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                />
              </div>
              <button
                type="button"
                onClick={() => removeManualMapping(index)}
                className="p-2 text-gray-400 hover:text-red-500 transition-colors"
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addManualMapping}
            className="flex items-center gap-2 px-3 py-2 text-sm text-teal-600 dark:text-teal-400
                       hover:bg-teal-50 dark:hover:bg-teal-900/20 rounded-lg transition-colors"
          >
            <PlusIcon className="h-4 w-4" />
            Add Tool Mapping
          </button>
        </div>
      )}
    </div>
  );

  // Step 3: Configuration (aliases, version pins, scopes)
  const renderConfigStep = () => (
    <div className="space-y-6">
      {/* Tool alias/version overrides */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Tool Aliases and Version Pins
        </label>
        {allMappings.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400 py-4 text-center bg-gray-50 dark:bg-gray-900/50 rounded-lg">
            No tools selected. Go back to add tools.
          </p>
        ) : (
          <div className="space-y-2">
            {allMappings.map((mapping, index) => (
              <div
                key={`${mapping.backend_server_path}-${mapping.tool_name}-${index}`}
                className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-lg"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-mono text-gray-900 dark:text-white truncate">
                    {mapping.tool_name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {mapping.backend_server_path}
                  </div>
                </div>
                <div className="w-40">
                  <input
                    type="text"
                    value={mapping.alias || ''}
                    onChange={(e) => {
                      if (index < toolMappings.length) {
                        const updated = [...toolMappings];
                        updated[index] = { ...updated[index], alias: e.target.value || null };
                        setToolMappings(updated);
                      } else {
                        const manualIndex = index - toolMappings.length;
                        if (manualIndex < manualMappings.length) {
                          updateManualMapping(manualIndex, 'alias', e.target.value);
                        }
                      }
                    }}
                    placeholder="Alias"
                    className="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded
                               bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  />
                </div>
                <div className="w-28">
                  <input
                    type="text"
                    value={mapping.backend_version || ''}
                    onChange={(e) => {
                      if (index < toolMappings.length) {
                        const updated = [...toolMappings];
                        updated[index] = { ...updated[index], backend_version: e.target.value || null };
                        setToolMappings(updated);
                      }
                      // Manual mappings don't have backend_version support
                    }}
                    placeholder="Version"
                    className="w-full px-2 py-1 text-xs border border-gray-300 dark:border-gray-600 rounded
                               bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Required Scopes */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Required Scopes
        </label>
        <input
          type="text"
          value={requiredScopes}
          onChange={(e) => setRequiredScopes(e.target.value)}
          placeholder="scope1, scope2 (comma-separated)"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-teal-500 focus:border-transparent"
        />
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Scopes required to access this virtual server. Leave empty for unrestricted access.
        </p>
      </div>
    </div>
  );

  // Step 4: Review
  const renderReviewStep = () => (
    <div className="space-y-4">
      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Server Details
        </h4>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <dt className="text-gray-500 dark:text-gray-400">Name</dt>
          <dd className="text-gray-900 dark:text-white font-medium">{name || '-'}</dd>
          <dt className="text-gray-500 dark:text-gray-400">Path</dt>
          <dd className="text-gray-900 dark:text-white font-mono text-xs">{path || '-'}</dd>
          <dt className="text-gray-500 dark:text-gray-400">Description</dt>
          <dd className="text-gray-900 dark:text-white">{description || '-'}</dd>
          <dt className="text-gray-500 dark:text-gray-400">Tags</dt>
          <dd className="text-gray-900 dark:text-white">
            {parsedTags.length > 0 ? parsedTags.join(', ') : '-'}
          </dd>
          <dt className="text-gray-500 dark:text-gray-400">Required Scopes</dt>
          <dd className="text-gray-900 dark:text-white">
            {parsedScopes.length > 0 ? parsedScopes.join(', ') : 'None (unrestricted)'}
          </dd>
        </dl>
      </div>

      <div className="bg-gray-50 dark:bg-gray-900/50 rounded-lg p-4">
        <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
          Tool Mappings ({allMappings.length})
        </h4>
        {allMappings.length === 0 ? (
          <p className="text-sm text-gray-500 dark:text-gray-400">No tools configured</p>
        ) : (
          <div className="space-y-1.5">
            {allMappings.map((mapping, index) => (
              <div
                key={`review-${mapping.backend_server_path}-${mapping.tool_name}-${index}`}
                className="flex items-center justify-between text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-gray-900 dark:text-white">
                    {mapping.alias || mapping.tool_name}
                  </span>
                  {mapping.alias && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      (from {mapping.tool_name})
                    </span>
                  )}
                </div>
                <span className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                  {mapping.backend_server_path}
                  {mapping.backend_version && ` @${mapping.backend_version}`}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Unique backend servers count */}
      <div className="text-sm text-gray-500 dark:text-gray-400">
        {(() => {
          const uniqueBackends = new Set(allMappings.map((m) => m.backend_server_path));
          return `${allMappings.length} tool(s) from ${uniqueBackends.size} backend server(s)`;
        })()}
      </div>
    </div>
  );

  // Render the current step content
  const renderCurrentStep = () => {
    switch (currentStep) {
      case 'basics':
        return renderBasicsStep();
      case 'tools':
        return renderToolSelectionStep();
      case 'config':
        return renderConfigStep();
      case 'review':
        return renderReviewStep();
    }
  };

  const isLastStep = currentStepIndex === STEPS.length - 1;
  const isFirstStep = currentStepIndex === 0;

  // Handle Escape key to close the modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !saving) {
        onCancel();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onCancel, saving]);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div
        className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] flex flex-col"
        role="dialog"
        aria-modal="true"
        aria-label={isEditMode ? 'Edit Virtual Server' : 'Create Virtual Server'}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            {isEditMode ? 'Edit Virtual Server' : 'Create Virtual Server'}
          </h2>
          <button
            onClick={onCancel}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Step indicator */}
        {renderStepIndicator()}

        {/* Step content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Validation error */}
          {validationError && (
            <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
              <p className="text-sm text-red-700 dark:text-red-300">
                {validationError}
              </p>
            </div>
          )}

          {renderCurrentStep()}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
          <button
            type="button"
            onClick={isFirstStep ? onCancel : goToPrev}
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300
                       bg-gray-100 dark:bg-gray-700 rounded-lg
                       hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors
                       disabled:opacity-50"
          >
            {isFirstStep ? 'Cancel' : 'Back'}
          </button>

          <div className="flex gap-3">
            {!isFirstStep && (
              <button
                type="button"
                onClick={onCancel}
                disabled={saving}
                className="px-4 py-2 text-sm font-medium text-gray-500 dark:text-gray-400
                           hover:text-gray-700 dark:hover:text-gray-200 transition-colors
                           disabled:opacity-50"
              >
                Cancel
              </button>
            )}
            {isLastStep ? (
              <button
                type="button"
                onClick={handleSubmit}
                disabled={saving}
                className="px-4 py-2 text-sm font-medium text-white
                           bg-teal-600 rounded-lg hover:bg-teal-700 transition-colors
                           disabled:opacity-50 flex items-center gap-2"
              >
                {saving && (
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                )}
                {isEditMode ? 'Save Changes' : 'Create Virtual Server'}
              </button>
            ) : (
              <button
                type="button"
                onClick={goToNext}
                className="px-4 py-2 text-sm font-medium text-white
                           bg-teal-600 rounded-lg hover:bg-teal-700 transition-colors"
              >
                Next
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VirtualServerForm;
