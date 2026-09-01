import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CustomEntityForm from '../CustomEntityForm';
import type { CustomTypeDescriptor } from '../../types/customEntity';

const descriptor: CustomTypeDescriptor = {
  name: 'dataset',
  display_name: 'Dataset',
  fields: [
    { name: 'owner', datatype: 'string', required: true } as any,
    { name: 'rows', datatype: 'number', required: false } as any,
  ],
} as CustomTypeDescriptor;

describe('CustomEntityForm', () => {
  it('renders create title and envelope + attribute fields', () => {
    render(
      <CustomEntityForm descriptor={descriptor} onSave={jest.fn()} onCancel={jest.fn()} />,
    );
    expect(screen.getByText('Create Dataset')).toBeInTheDocument();
    expect(screen.getByText(/Name/)).toBeInTheDocument();
    expect(screen.getByText(/Owner/)).toBeInTheDocument();
    expect(screen.getByText(/Rows/)).toBeInTheDocument();
  });

  it('shows edit title when a record is supplied', () => {
    render(
      <CustomEntityForm
        descriptor={descriptor}
        record={
          {
            name: 'ds1',
            description: '',
            visibility: 'private',
            allowed_groups: [],
            tags: [],
            attributes: {},
          } as any
        }
        onSave={jest.fn()}
        onCancel={jest.fn()}
      />,
    );
    expect(screen.getByText('Edit Dataset')).toBeInTheDocument();
    expect(screen.getByDisplayValue('ds1')).toBeInTheDocument();
  });

  it('validates that name is required before saving', async () => {
    const onSave = jest.fn();
    render(
      <CustomEntityForm descriptor={descriptor} onSave={onSave} onCancel={jest.fn()} />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.getByText('Name is required')).toBeInTheDocument());
    expect(onSave).not.toHaveBeenCalled();
  });

  it('reveals the allowed-groups chip input for group-restricted visibility', () => {
    render(
      <CustomEntityForm descriptor={descriptor} onSave={jest.fn()} onCancel={jest.fn()} />,
    );
    expect(screen.queryByText(/Allowed Groups/)).not.toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue('private'), {
      target: { value: 'group-restricted' },
    });
    expect(screen.getByText(/Allowed Groups/)).toBeInTheDocument();
  });

  it('submits a valid payload', async () => {
    const onSave = jest.fn().mockResolvedValue(undefined);
    render(
      <CustomEntityForm descriptor={descriptor} onSave={onSave} onCancel={jest.fn()} />,
    );
    // Name is the first text input in the envelope.
    const nameInput = screen.getAllByRole('textbox')[0];
    fireEvent.change(nameInput, { target: { value: 'my-dataset' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'my-dataset', visibility: 'private' }),
        // Second arg: the upstream-header rows (empty when not proxied).
        expect.anything(),
      ),
    );
  });

  it('calls onCancel from the cancel button', () => {
    const onCancel = jest.fn();
    render(
      <CustomEntityForm descriptor={descriptor} onSave={jest.fn()} onCancel={onCancel} />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCancel).toHaveBeenCalled();
  });

  it('folds the proxy opt-in into the attributes bag on submit', async () => {
    const onSave = jest.fn().mockResolvedValue(undefined);
    render(
      <CustomEntityForm descriptor={descriptor} onSave={onSave} onCancel={jest.fn()} />,
    );
    const nameInput = screen.getAllByRole('textbox')[0];
    fireEvent.change(nameInput, { target: { value: 'proxied-ds' } });
    // Enable proxying, which reveals the backend URL input.
    fireEvent.click(screen.getByRole('checkbox'));
    const urlInput = screen.getByPlaceholderText('https://backend.example.com/');
    fireEvent.change(urlInput, { target: { value: 'https://backend.example.com/' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    // Proxy fields are TOP-LEVEL on the wire (mixin fields), NOT in attributes.
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          is_proxied: true,
          proxy_target_url: 'https://backend.example.com/',
        }),
        expect.anything(),
      ),
    );
    // And they must NOT leak into the attributes bag.
    const saved = onSave.mock.calls[0][0];
    expect(saved.attributes).not.toHaveProperty('is_proxied');
    expect(saved.attributes).not.toHaveProperty('proxy_target_url');
  });

  it('blocks submit when proxying is on but the target URL is blank', async () => {
    const onSave = jest.fn();
    render(
      <CustomEntityForm descriptor={descriptor} onSave={onSave} onCancel={jest.fn()} />,
    );
    const nameInput = screen.getAllByRole('textbox')[0];
    fireEvent.change(nameInput, { target: { value: 'proxied-ds' } });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() =>
      expect(
        screen.getByText(/Proxy target URL is required when proxying is enabled/),
      ).toBeInTheDocument(),
    );
    expect(onSave).not.toHaveBeenCalled();
  });

  it('blocks submit when a proxied entity has an invalid upstream header row', async () => {
    const onSave = jest.fn();
    render(
      <CustomEntityForm descriptor={descriptor} onSave={onSave} onCancel={jest.fn()} />,
    );
    const nameInput = screen.getAllByRole('textbox')[0];
    fireEvent.change(nameInput, { target: { value: 'proxied-ds' } });
    // Enable proxying (single checkbox at this point), give a valid target.
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.change(screen.getByPlaceholderText('https://backend.example.com/'), {
      target: { value: 'https://backend.example.com/' },
    });
    // Add a fixed Authorization row — invalid unless caller-overridable.
    fireEvent.click(screen.getByText('+ Add header'));
    fireEvent.change(screen.getByLabelText('Header name (row 1)'), {
      target: { value: 'Authorization' },
    });
    fireEvent.change(screen.getByLabelText('Header value (row 1)'), {
      target: { value: 'Bearer x' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    // The row error is surfaced both inline on the row and in the form's error
    // banner (two nodes) and gates save.
    await waitFor(() =>
      expect(screen.getAllByText(/caller-overridable/).length).toBeGreaterThan(0),
    );
    expect(onSave).not.toHaveBeenCalled();
  });

  it('trims a non-blank upstream header value before submit', async () => {
    const onSave = jest.fn().mockResolvedValue(undefined);
    render(
      <CustomEntityForm descriptor={descriptor} onSave={onSave} onCancel={jest.fn()} />,
    );
    fireEvent.change(screen.getAllByRole('textbox')[0], { target: { value: 'proxied-ds' } });
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.change(screen.getByPlaceholderText('https://backend.example.com/'), {
      target: { value: 'https://backend.example.com/' },
    });
    fireEvent.click(screen.getByText('+ Add header'));
    fireEvent.change(screen.getByLabelText('Header name (row 1)'), {
      target: { value: 'X-Api-Key' },
    });
    fireEvent.change(screen.getByLabelText('Header value (row 1)'), {
      target: { value: '  secret  ' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() =>
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({ is_proxied: true }),
        [{ name: 'X-Api-Key', value: 'secret', overridable: false }],
      ),
    );
  });

  it('does not render a raw widget for a descriptor field colliding with a reserved proxy key', () => {
    const collidingDescriptor = {
      name: 'dataset',
      display_name: 'Dataset',
      fields: [
        { name: 'owner', datatype: 'string', required: true },
        // Admin mis-configuration: a descriptor field named like a reserved key.
        { name: 'is_proxied', datatype: 'bool', required: false },
      ],
    } as CustomTypeDescriptor;
    render(
      <CustomEntityForm descriptor={collidingDescriptor} onSave={jest.fn()} onCancel={jest.fn()} />,
    );
    // Exactly one proxy control (the ProxyField's checkbox), not two — the
    // colliding descriptor field must not spawn a second raw bool widget.
    expect(screen.getAllByRole('checkbox')).toHaveLength(1);
  });
});
