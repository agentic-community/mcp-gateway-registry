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

    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['eng', 'hr-team']);
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

    expect(screen.getAllByRole('option').map((o) => o.textContent)).toEqual(['hr-ops', 'hr-team']);
  });

  it('falls back to a text field when the group list cannot load', () => {
    _groups([], 'Failed to load groups');
    const onChange = jest.fn();
    render(<GroupMultiSelect id="g" selected={[]} onChange={onChange} />);

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'hr-team, eng' } });

    expect(onChange).toHaveBeenCalledWith(['hr-team', 'eng']);
    expect(screen.getByText(/Could not load the group list/)).toBeInTheDocument();
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
