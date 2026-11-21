/**
 * Tests for GroupMultiSelect: the allowed-groups dropdown on the OpenAPI server form.
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import GroupMultiSelect from '../GroupMultiSelect';
import OpenApiServerForm from '../OpenApiServerForm';
import { useIAMGroups } from '../../hooks/useIAM';

jest.mock('../../hooks/useIAM', () => ({ useIAMGroups: jest.fn() }));
const mockGroups = useIAMGroups as jest.Mock;

function _listed(): (string | null | undefined)[] {
  return screen.getAllByRole('checkbox').map((c) => c.closest('label')?.textContent);
}

function _groups(names: string[], error: string | null = null) {
  mockGroups.mockReturnValue({ groups: names.map((name) => ({ name })), isLoading: false, error, refetch: jest.fn() });
}

describe('GroupMultiSelect', () => {
  beforeEach(() => mockGroups.mockReset());

  it('lists the existing groups and toggles a selection', () => {
    _groups(['hr-team', 'eng']);
    const onChange = jest.fn();
    render(<GroupMultiSelect id="g" selected={[]} onChange={onChange} />);

    fireEvent.click(screen.getByRole('button', { name: /Choose groups/ }));
    fireEvent.click(screen.getByLabelText('hr-team'));

    expect(_listed()).toEqual(['eng', 'hr-team']);
    expect(onChange).toHaveBeenCalledWith(['hr-team']);
  });

  it('unchecks a selected group', () => {
    _groups(['hr-team', 'eng']);
    const onChange = jest.fn();
    render(<GroupMultiSelect id="g" selected={['hr-team', 'eng']} onChange={onChange} />);

    fireEvent.click(screen.getByRole('button', { name: /hr-team, eng/ }));
    fireEvent.click(screen.getByLabelText('hr-team'));

    expect(onChange).toHaveBeenCalledWith(['eng']);
  });

  it('keeps a selected group that no longer exists, so it can be removed', () => {
    _groups(['eng']);
    render(<GroupMultiSelect id="g" selected={['old-team']} onChange={jest.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: /old-team/ }));

    expect(screen.getByLabelText('old-team')).toBeChecked();
  });

  it('filters the list', () => {
    _groups(['hr-team', 'eng', 'hr-ops']);
    render(<GroupMultiSelect id="g" selected={[]} onChange={jest.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: /Choose groups/ }));
    fireEvent.change(screen.getByLabelText('Filter groups'), { target: { value: 'hr' } });

    expect(_listed()).toEqual(['hr-ops', 'hr-team']);
  });

  it('falls back to a text field that takes several groups typed one key at a time', () => {
    _groups([], 'Failed to load groups');
    const onChange = jest.fn();
    render(<GroupMultiSelect id="g" selected={[]} onChange={onChange} />);
    const field = screen.getByRole('textbox');

    for (const value of ['hr-team', 'hr-team,', 'hr-team, ', 'hr-team, e', 'hr-team, eng']) {
      fireEvent.change(field, { target: { value } });
    }

    expect(field).toHaveValue('hr-team, eng');
    expect(onChange).toHaveBeenLastCalledWith(['hr-team', 'eng']);
    expect(screen.getByText(/Could not load the group list/)).toBeInTheDocument();
  });

  it('closes on a click outside the dropdown', () => {
    _groups(['hr-team']);
    render(<GroupMultiSelect id="g" selected={[]} onChange={jest.fn()} />);
    fireEvent.click(screen.getByRole('button', { name: /Choose groups/ }));

    fireEvent.mouseDown(document.body);

    expect(screen.queryByLabelText('hr-team')).not.toBeInTheDocument();
  });
});

describe('OpenApiServerForm allowed groups', () => {
  it('sends the groups chosen in the dropdown', async () => {
    _groups(['hr-team', 'eng']);
    const onSave = jest.fn().mockResolvedValue(undefined);
    render(<OpenApiServerForm onSave={onSave} onCancel={jest.fn()} />);

    fireEvent.change(screen.getByLabelText(/Name/), { target: { value: 'NASA' } });
    fireEvent.change(screen.getByLabelText('Visibility'), { target: { value: 'group-restricted' } });
    fireEvent.click(screen.getByRole('button', { name: 'Allowed groups' }));
    fireEvent.click(screen.getByLabelText('hr-team'));
    fireEvent.click(screen.getByRole('button', { name: 'Create Server' }));

    await screen.findByRole('button', { name: 'Create Server' });
    expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ visibility: 'group-restricted', allowed_groups: ['hr-team'] }));
  });
});

describe('GroupMultiSelect inside the server form', () => {
  it('closes only the dropdown on Escape, then the form on a second Escape', () => {
    _groups(['hr-team']);
    const onCancel = jest.fn();
    render(<OpenApiServerForm onSave={jest.fn()} onCancel={onCancel} />);
    fireEvent.change(screen.getByLabelText('Visibility'), { target: { value: 'group-restricted' } });
    fireEvent.click(screen.getByRole('button', { name: 'Allowed groups' }));

    fireEvent.keyDown(screen.getByLabelText('Filter groups'), { key: 'Escape' });

    expect(screen.queryByLabelText('hr-team')).not.toBeInTheDocument();
    expect(onCancel).not.toHaveBeenCalled();

    fireEvent.keyDown(document.body, { key: 'Escape' });

    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
