import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import DuplicateCheckModal from '../DuplicateCheckModal';
import type { ExistingEntity } from '../../types/duplicateCheck';

const collisionEntity: ExistingEntity = {
  entity_type: 'mcp_server',
  path: '/servers/foo',
  name: 'Foo Server',
  owner: 'alice',
  registered_at: '2026-05-01T00:00:00Z',
  relevance_score: null,
  match_reason: 'URL match',
};

const advisoryEntity: ExistingEntity = {
  entity_type: 'a2a_agent',
  path: '/agents/bar',
  name: 'Bar Agent',
  owner: 'bob',
  registered_at: '2026-05-01T00:00:00Z',
  relevance_score: 0.92,
  match_reason: 'Similar name',
};

const redactedEntity: ExistingEntity = {
  entity_type: 'skill',
  path: '',
  name: '',
  owner: null,
  registered_at: null,
  relevance_score: null,
  match_reason: null,
};

describe('DuplicateCheckModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: jest.fn(),
    onProceed: jest.fn(),
    onPickExisting: jest.fn(),
    collisionWith: [],
    advisoryMatches: [],
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders collision section with prominent warning', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[collisionEntity]}
      />,
    );

    expect(screen.getByText(/A matching entry already exists/i)).toBeInTheDocument();
    expect(screen.getByText('Foo Server')).toBeInTheDocument();
    expect(screen.getByText('/servers/foo')).toBeInTheDocument();
    expect(screen.getByText(/Owner: alice/)).toBeInTheDocument();
  });

  test('renders advisory section with secondary tone', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        advisoryMatches={[advisoryEntity]}
      />,
    );

    expect(screen.getByText(/Similar 1 agent found/i)).toBeInTheDocument();
    expect(screen.getByText('Bar Agent')).toBeInTheDocument();
    expect(screen.getByText(/Relevance: 0.92/)).toBeInTheDocument();
  });

  test('renders redacted entry as placeholder', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[redactedEntity]}
      />,
    );

    expect(
      screen.getByText(/already.*registered.*don.*t have permission/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /view/i })).not.toBeInTheDocument();
  });

  test('proceed button calls onProceed', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[collisionEntity]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Register anyway/i }));
    expect(defaultProps.onProceed).toHaveBeenCalledTimes(1);
  });

  test('close button calls onClose', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[collisionEntity]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Edit my entry/i }));
    expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
  });

  test('view button calls onPickExisting with entity', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[collisionEntity]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /View/i }));
    expect(defaultProps.onPickExisting).toHaveBeenCalledWith(collisionEntity);
  });

  test('disables buttons while loading', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[collisionEntity]}
        isLoading
      />,
    );

    expect(screen.getByRole('button', { name: /Edit my entry/i })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Registering/i })).toBeDisabled();
  });

  test('renders both sections simultaneously', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        collisionWith={[collisionEntity]}
        advisoryMatches={[advisoryEntity]}
      />,
    );

    expect(screen.getByText('Foo Server')).toBeInTheDocument();
    expect(screen.getByText('Bar Agent')).toBeInTheDocument();
    expect(
      screen.getByText(/may also want to consider these similar entries/i),
    ).toBeInTheDocument();
  });

  test('title pluralizes from the actual matches when all share a type', () => {
    // advisoryEntity is an a2a_agent; a list of one yields "1 agent".
    render(
      <DuplicateCheckModal
        {...defaultProps}
        advisoryMatches={[advisoryEntity]}
      />,
    );

    expect(screen.getByText(/Similar 1 agent found/i)).toBeInTheDocument();
  });

  test('title falls back to entity-agnostic copy on heterogeneous matches', () => {
    // The dedup checks are cross-entity: an advisory list can mix
    // server / agent / skill hits. When the list isn't all one type,
    // we don't lie about it by picking the registering type's label.
    const skillEntity: ExistingEntity = {
      entity_type: 'skill',
      path: '/skills/baz',
      name: 'Baz Skill',
      owner: 'carol',
      registered_at: '2026-05-01T00:00:00Z',
      relevance_score: 0.88,
      match_reason: 'Similar name',
    };

    render(
      <DuplicateCheckModal
        {...defaultProps}
        advisoryMatches={[advisoryEntity, skillEntity]}
      />,
    );

    expect(screen.getByText(/Similar 2 entries found/i)).toBeInTheDocument();
  });

  test('does not render when isOpen is false', () => {
    render(
      <DuplicateCheckModal
        {...defaultProps}
        isOpen={false}
        collisionWith={[collisionEntity]}
      />,
    );

    expect(screen.queryByText('Foo Server')).not.toBeInTheDocument();
  });
});
