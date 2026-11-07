import React, { useState } from 'react';
import {
  KeyIcon,
  ClipboardIcon,
  ClipboardDocumentListIcon,
  CheckIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import axios from 'axios';
import { useAuth } from '../contexts/AuthContext';
import { fieldClass } from '../components/formFields/formClasses';

const TokenGeneration: React.FC = () => {
  const { user } = useAuth();
  const [formData, setFormData] = useState({
    description: '',
    expires_in_hours: 8,
    scopeMethod: 'current' as 'current' | 'custom',
    customScopes: '',
  });
  const [generatedToken, setGeneratedToken] = useState<string>('');
  const [tokenDetails, setTokenDetails] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string>('');

  const expirationOptions = [
    { value: 1, label: '1 hour' },
    { value: 8, label: '8 hours' },
    { value: 24, label: '24 hours' },
  ];

  const handleGenerateToken = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const requestData: any = {
        description: formData.description,
        expires_in_hours: formData.expires_in_hours,
      };

      // Handle scopes based on the selected method
      if (formData.scopeMethod === 'custom') {
        const customScopesText = formData.customScopes.trim();
        if (customScopesText) {
          try {
            const parsedScopes = JSON.parse(customScopesText);
            if (!Array.isArray(parsedScopes)) {
              throw new Error('Custom scopes must be a JSON array');
            }
            requestData.requested_scopes = parsedScopes;
          } catch (e) {
            setError('Invalid JSON format for custom scopes. Please provide a valid JSON array.');
            return;
          }
        }
      }
      // If using current scopes, we don't need to set requested_scopes - it will default to user's current scopes

      const response = await axios.post('/api/tokens/generate', requestData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.data.success) {
        setGeneratedToken(response.data.token_data.access_token);
        setTokenDetails(response.data);
      } else {
        throw new Error('Token generation failed');
      }
    } catch (error: any) {
      console.error('Failed to generate token:', error);
      setError(error.response?.data?.detail || 'Failed to generate token');
    } finally {
      setLoading(false);
    }
  };

  const handleCopyToken = async () => {
    try {
      await navigator.clipboard.writeText(generatedToken);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = generatedToken;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      textArea.style.top = '-999999px';
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();

      try {
        document.execCommand('copy');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (err) {
        console.error('Failed to copy token:', err);
      }

      document.body.removeChild(textArea);
    }
  };

  const validateCustomScopes = () => {
    if (formData.scopeMethod === 'custom' && formData.customScopes.trim()) {
      try {
        const parsed = JSON.parse(formData.customScopes);
        if (!Array.isArray(parsed)) {
          return 'Custom scopes must be a JSON array';
        }
        return null;
      } catch (e) {
        return 'Invalid JSON format';
      }
    }
    return null;
  };

  const scopeValidationError = validateCustomScopes();

  return (
    <div className="flex flex-col h-full">
      {/* Compact Header Section */}
      <div className="flex-shrink-0 pb-2">
        <div className="text-center">
          <div className="mx-auto w-10 h-10 bg-primary-100 dark:bg-primary-900 rounded-full flex items-center justify-center mb-2">
            <KeyIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
          </div>
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">Generate JWT Token</h1>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Generate a personal access token for programmatic access to MCP servers
          </p>
        </div>
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <div className="max-w-4xl mx-auto space-y-4 pb-6">
          {/* Current User Permissions - Compact */}
          <div className="card p-4">
            <h3 className="text-base font-semibold text-gray-900 dark:text-white mb-2">Your Current Permissions</h3>
            <div className="mb-2">
              <span className="text-xs font-medium text-gray-700 dark:text-gray-300">Current Scopes:</span>
              <div className="flex flex-wrap gap-1 mt-1">
                {user?.scopes && user.scopes.length > 0 ? (
                  user.scopes.map((scope) => (
                    <span key={scope} className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-info-100 text-info-800 dark:bg-info-900 dark:text-info-200">
                      {scope}
                    </span>
                  ))
                ) : (
                  <span className="text-xs text-gray-500 dark:text-gray-400">No scopes available</span>
                )}
              </div>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              <em>Generated tokens can have the same or fewer permissions than your current scopes.</em>
            </p>
          </div>

          {/* Token Configuration Form */}
          <div className="card p-4">
            <form onSubmit={handleGenerateToken} className="space-y-4">
              <h3 className="text-base font-semibold text-gray-900 dark:text-white">Token Configuration</h3>

              {/* Form Fields - Responsive Grid */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {/* Left Column */}
                <div className="space-y-3">
                  {/* Description */}
                  <div>
                    <label htmlFor="description" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Description (optional)
                    </label>
                    <input
                      type="text"
                      id="description"
                      className="input text-sm"
                      placeholder="e.g., Token for automation script"
                      value={formData.description}
                      onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                    />
                  </div>

                  {/* Expiration */}
                  <div>
                    <label htmlFor="expires_in_hours" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                      Expires In
                    </label>
                    <select
                      id="expires_in_hours"
                      className="input text-sm"
                      value={formData.expires_in_hours}
                      onChange={(e) => setFormData(prev => ({ ...prev, expires_in_hours: parseInt(e.target.value) }))}
                    >
                      {expirationOptions.map((option) => (
                        <option key={option.value} value={option.value}>
                          {option.label}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                      The maximum lifetime is configured by your operator
                      (MCP_TOKEN_MAX_TTL_HOURS, default 24h). Requests above the
                      configured maximum are rejected.
                    </p>
                  </div>
                </div>

                {/* Right Column */}
                <div className="space-y-3">
                  {/* Scope Configuration */}
                  <div>
                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Scope Configuration</h4>

                    <div className="space-y-2">
                      <label className="flex items-center space-x-2">
                        <input
                          type="radio"
                          name="scopeMethod"
                          value="current"
                          checked={formData.scopeMethod === 'current'}
                          onChange={(e) => setFormData(prev => ({ ...prev, scopeMethod: e.target.value as 'current' | 'custom' }))}
                          className="h-4 w-4 border-line-strong bg-surface text-accent focus:ring-focus"
                        />
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            Use my current scopes
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            Generate token with all your current permissions
                          </div>
                        </div>
                      </label>

                      <label className="flex items-center space-x-2">
                        <input
                          type="radio"
                          name="scopeMethod"
                          value="custom"
                          checked={formData.scopeMethod === 'custom'}
                          onChange={(e) => setFormData(prev => ({ ...prev, scopeMethod: e.target.value as 'current' | 'custom' }))}
                          className="h-4 w-4 border-line-strong bg-surface text-accent focus:ring-focus"
                        />
                        <div>
                          <div className="text-sm font-medium text-gray-900 dark:text-white">
                            Upload custom scopes (JSON)
                          </div>
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            Specify custom scopes in JSON format
                          </div>
                        </div>
                      </label>
                    </div>

                    {/* Custom Scopes JSON Input */}
                    {formData.scopeMethod === 'custom' && (
                      <div className="mt-3">
                        <label htmlFor="customScopes" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                          Custom Scopes (JSON format)
                        </label>
                        <textarea
                          id="customScopes"
                          className={`${fieldClass(scopeValidationError !== null)} h-24 font-mono text-xs`}
                          placeholder={`["mcp-servers-restricted/read", "mcp-registry-user"]`}
                          value={formData.customScopes}
                          onChange={(e) => setFormData(prev => ({ ...prev, customScopes: e.target.value }))}
                        />
                        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                          Enter a JSON array of scope names. Must be a subset of your current scopes.
                        </p>
                        {scopeValidationError && (
                          <p className="mt-1 text-xs text-danger-600 dark:text-danger-400">
                            {scopeValidationError}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={loading || scopeValidationError !== null}
                className="w-full btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed py-2 text-sm"
              >
                {loading ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <KeyIcon className="h-4 w-4" />
                    <span>Generate Token</span>
                  </>
                )}
              </button>

              {/* Error Display */}
              {error && (
                <div className="p-3 bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <ExclamationTriangleIcon className="h-4 w-4 text-danger-600 dark:text-danger-400" />
                    <span className="text-sm text-danger-800 dark:text-danger-200">{error}</span>
                  </div>
                </div>
              )}
            </form>
          </div>

          {/* Generated Token Result */}
          {generatedToken && tokenDetails && (
            <div className="card p-4 bg-success-50 dark:bg-success-900/20 border-success-200 dark:border-success-800">
              <div className="flex items-center space-x-2 mb-3">
                <CheckIcon className="h-5 w-5 text-success-600 dark:text-success-400" />
                <h3 className="text-lg font-semibold text-success-900 dark:text-success-100">
                  Token Generated Successfully
                </h3>
              </div>

              {/* Token Display */}
              <div className="relative mb-4">
                <div className="bg-surface-muted p-4 pr-12 rounded-lg border border-success-200 dark:border-success-700">
                  <code className="text-sm font-mono break-all text-fg">
                    {generatedToken}
                  </code>
                </div>

                <button
                  onClick={handleCopyToken}
                  className="btn-icon absolute top-2 right-2"
                  title={copied ? 'Copied!' : 'Copy token'}
                >
                  {copied ? (
                    <CheckIcon className="h-4 w-4 text-success-600 dark:text-success-400" />
                  ) : (
                    <ClipboardIcon className="h-4 w-4" />
                  )}
                </button>
              </div>

              {/* Token Details */}
              <div className="space-y-2 text-sm text-fg mb-4">
                <p><strong>Expires:</strong> {new Date(Date.now() + tokenDetails.token_data.expires_in * 1000).toLocaleString()}</p>
                <p><strong>Scopes:</strong> {tokenDetails.requested_scopes.join(', ')}</p>
                {tokenDetails.token_data.description && (
                  <p><strong>Description:</strong> {tokenDetails.token_data.description}</p>
                )}
              </div>

              {/* Usage Instructions */}
              <div className="p-4 bg-info-50 dark:bg-info-900/20 border border-info-200 dark:border-info-800 rounded-lg mb-4">
                <h4 className="flex items-center gap-1.5 text-sm font-semibold text-info-900 dark:text-info-100 mb-2">
                  <ClipboardDocumentListIcon className="h-4 w-4" aria-hidden="true" />
                  Usage Instructions
                </h4>
                <p className="text-sm text-info-800 dark:text-info-200 mb-2">Use this token in your API requests:</p>
                <code className="block text-sm bg-info-100 dark:bg-info-900/40 p-2 rounded font-mono text-info-900 dark:text-info-100">
                  Authorization: Bearer YOUR_TOKEN_HERE
                </code>
                <p className="text-xs text-info-600 dark:text-info-300 mt-2">Replace YOUR_TOKEN_HERE with the token above.</p>
              </div>

              {/* Security Warning */}
              <div className="p-4 bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg">
                <p className="flex items-start gap-1.5 text-sm text-warning-800 dark:text-warning-200">
                  <ExclamationTriangleIcon className="h-4 w-4 mt-0.5 flex-shrink-0" aria-hidden="true" />
                  <span><strong>Important:</strong> This token will not be shown again. Save it securely!</span>
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TokenGeneration;
