import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import VirtualServerCard from '../VirtualServerCard';
import { VirtualServerInfo } from '../../types/virtualServer';

// ServerConfigModal pulls in connection logic we don't need here; stub it.
jest.mock('../ServerConfigModal', () => () => null);
// StarRatingWidget makes network calls on mount; stub it to a sentinel.
jest.mock('../StarRatingWidget', () => () => <div data-testid="stars" />);

const baseServer = {
  path: '/my-vserver',
  server_name: 'My Virtual Server',
  description: 'A composed virtual server',
  is_enabled: true,
  tags: ['alpha', 'beta'],
  tool_count: 0,
  num_stars: 0,
  rating_details: [],
} as unknown as VirtualServerInfo;

function renderCard(overrides: Partial<VirtualServerInfo> = {}, props = {}) {
  const onToggle = jest.fn();
  const onEdit = jest.fn();
  const onDelete = jest.fn();
  render(
    <VirtualServerCard
      virtualServer={{ ...baseServer, ...overrides }}
      canModify
      onToggle={onToggle}
      onEdit={onEdit}
      onDelete={onDelete}
      {...props}
    />,
  );
  return { onToggle, onEdit, onDelete };
}

describe('VirtualServerCard (composed from card primitives)', () => {
  it('renders the name, path, and VIRTUAL badge', () => {
    renderCard();
    expect(screen.getByText('My Virtual Server')).toBeInTheDocument();
    expect(screen.getByText('/my-vserver')).toBeInTheDocument();
    expect(screen.getByText('VIRTUAL')).toBeInTheDocument();
  });

  it('renders the description and tags', () => {
    renderCard();
    expect(screen.getByText('A composed virtual server')).toBeInTheDocument();
    expect(screen.getByText('#alpha')).toBeInTheDocument();
    expect(screen.getByText('#beta')).toBeInTheDocument();
  });

  it('shows the enabled status and a virtual-accented toggle', () => {
    const { container } = render(
      <VirtualServerCard
        virtualServer={baseServer}
        canModify
        onToggle={() => {}}
        onEdit={() => {}}
        onDelete={() => {}}
      />,
    );
    expect(screen.getByText('Enabled')).toBeInTheDocument();
    const track = container.querySelector('div.w-12');
    expect(track?.className).toContain('bg-virtual-600');
  });

  it('calls onToggle when the switch is flipped', () => {
    const { onToggle } = renderCard();
    fireEvent.click(
      screen.getByRole('checkbox', { name: 'Enable My Virtual Server' }),
    );
    expect(onToggle).toHaveBeenCalledWith('/my-vserver', false);
  });

  it('calls onEdit and onDelete from the header actions', () => {
    const { onEdit, onDelete } = renderCard();
    fireEvent.click(screen.getByTitle('Edit virtual server'));
    expect(onEdit).toHaveBeenCalled();
    fireEvent.click(screen.getByTitle('Delete virtual server'));
    expect(onDelete).toHaveBeenCalledWith('/my-vserver');
  });

  it('hides edit/delete when canModify is false', () => {
    render(
      <VirtualServerCard
        virtualServer={baseServer}
        canModify={false}
        onToggle={() => {}}
        onEdit={() => {}}
        onDelete={() => {}}
      />,
    );
    expect(screen.queryByTitle('Edit virtual server')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Delete virtual server')).not.toBeInTheDocument();
    expect(
      screen.queryByRole('checkbox', { name: 'Enable My Virtual Server' }),
    ).not.toBeInTheDocument();
  });
});
