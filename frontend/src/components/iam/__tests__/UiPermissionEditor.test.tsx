import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import UiPermissionEditor from '../UiPermissionEditor';

/**
 * These tests pin the security-critical behavior that the group-form mutation
 * checkboxes must NEVER grant the admin-conferring literal "all" for MCP Server
 * or Agent mutations. Granting register_/modify_/delete_/toggle_service (or the
 * agent equivalents) for "all" promotes the whole group to full registry admin
 * (registry/auth/privileged_constants.py::is_admin_conferring_action). The UI
 * therefore writes the "*" wildcard, which grants the mutation across every
 * server/agent WITHOUT admin. Admin groups are created only via CLI scope import.
 */
describe('UiPermissionEditor mutation grants', () => {
  const baseProps = {
    uiPermissions: {} as Record<string, string>,
    entityScopeGroups: [],
    skillOptions: [],
    skillsLoading: false,
  };

  function renderEditor(overrides: Partial<typeof baseProps> = {}) {
    const setPermValue = jest.fn();
    render(
      <UiPermissionEditor
        {...baseProps}
        {...overrides}
        setPermValue={setPermValue}
      />,
    );
    return setPermValue;
  }

  it.each([
    'register_service',
    'modify_service',
    'delete_service',
    'toggle_service',
    'publish_agent',
    'modify_agent',
    'delete_agent',
    'toggle_agent',
  ])('grants "%s" as "*" (non-admin), never "all"', (scopeKey) => {
    const setPermValue = renderEditor();
    fireEvent.click(screen.getByLabelText((_, el) => el?.id === `perm-${scopeKey}`));
    expect(setPermValue).toHaveBeenCalledWith(scopeKey, '*');
    expect(setPermValue).not.toHaveBeenCalledWith(scopeKey, 'all');
  });

  it('grants non-admin-conferring skill mutations with "all" (unchanged)', () => {
    const setPermValue = renderEditor();
    fireEvent.click(screen.getByLabelText((_, el) => el?.id === 'perm-publish_skill'));
    expect(setPermValue).toHaveBeenCalledWith('publish_skill', 'all');
  });

  it('clears a mutation scope to empty string when unchecked', () => {
    // Pre-granted (checkbox rendered checked), then click to uncheck.
    const setPermValue = renderEditor({ uiPermissions: { register_service: '*' } });
    fireEvent.click(screen.getByLabelText((_, el) => el?.id === 'perm-register_service'));
    expect(setPermValue).toHaveBeenCalledWith('register_service', '');
  });

  it('renders a stored "*" grant as a checked box (read-back)', () => {
    renderEditor({ uiPermissions: { register_service: '*' } });
    const box = document.getElementById('perm-register_service') as HTMLInputElement;
    expect(box).toBeTruthy();
    expect(box.checked).toBe(true);
  });

  it('shows the note that admin groups are CLI-only', () => {
    renderEditor();
    expect(screen.getByText(/never make the group a\s+registry admin/i)).toBeInTheDocument();
    expect(screen.getByText(/CLI scope\s+import/i)).toBeInTheDocument();
  });
});
