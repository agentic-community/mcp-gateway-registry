import React from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import { TagsField, FIELD, LABEL } from '../../formFields';
import Button from '../../Button';

/**
 * Shape of the new-server registration form, owned by the Dashboard.
 *
 * This is intentionally a minimal subset of the full ServerEditForm — quick
 * registration with the essentials. When the create/edit forms are unified
 * behind one mode-aware component, this becomes the create mode's field set.
 */
export interface ServerRegisterForm {
  name: string;
  path: string;
  proxyPass: string;
  description: string;
  official: boolean;
  tags: string[];
}

interface ServerRegisterModalProps {
  form: ServerRegisterForm;
  setForm: React.Dispatch<React.SetStateAction<ServerRegisterForm>>;
  loading: boolean;
  onSubmit: (e: React.FormEvent) => void;
  onClose: () => void;
}

/**
 * Minimal "Register New Server" form. Controlled by the Dashboard's registerForm
 * state; extracted from the inline Dashboard block to slim the page.
 */
const ServerRegisterModal: React.FC<ServerRegisterModalProps> = ({
  form,
  setForm,
  loading,
  onSubmit,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        role="dialog"
        aria-modal="true"
        className="bg-surface-overlay border border-line shadow-xl rounded-xl max-w-md w-full max-h-[90vh] overflow-y-auto"
      >
        <form onSubmit={onSubmit} className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-fg">
              Register New Server
            </h3>
            <button
              type="button"
              onClick={onClose}
              className="btn-icon"
              aria-label="Close"
            >
              <XMarkIcon className="h-6 w-6" />
            </button>
          </div>

          <div className="space-y-4">
            <div>
              <label className={LABEL}>Server Name *</label>
              <input
                type="text"
                required
                className={FIELD}
                value={form.name}
                onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., My Custom Server"
              />
            </div>

            <div>
              <label className={LABEL}>Path *</label>
              <input
                type="text"
                required
                className={FIELD}
                value={form.path}
                onChange={(e) => setForm((prev) => ({ ...prev, path: e.target.value }))}
                placeholder="/my-server"
              />
            </div>

            <div>
              <label className={LABEL}>Proxy URL *</label>
              <input
                type="url"
                required
                className={FIELD}
                value={form.proxyPass}
                onChange={(e) => setForm((prev) => ({ ...prev, proxyPass: e.target.value }))}
                placeholder="http://localhost:8080"
              />
            </div>

            <div>
              <label className={LABEL}>Description</label>
              <textarea
                className={FIELD}
                rows={3}
                value={form.description}
                onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
                placeholder="Brief description of the server"
              />
            </div>

            <TagsField
              value={form.tags}
              onChange={(tags) => setForm((prev) => ({ ...prev, tags }))}
            />
          </div>

          <div className="flex justify-end space-x-3 mt-6">
            <Button variant="secondary" fullWidth onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" type="submit" disabled={loading}>
              {loading ? 'Registering...' : 'Register Server'}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ServerRegisterModal;
