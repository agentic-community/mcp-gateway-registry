/**
 * DOM snapshot safety net for the five entity cards.
 *
 * Purpose: lock in main's current rendered markup BEFORE refactoring the cards
 * to compose the shared cards/ primitives. The refactor must keep these
 * snapshots byte-identical, proving the look does not change while the
 * duplication is removed.
 *
 * Heavy children (modals, rating widget, network-touching panels) are stubbed
 * to stable sentinels so the snapshot captures the card's own structure, not
 * its children's internals.
 */
import React from 'react';
import { render } from '@testing-library/react';

import ServerCard, { Server } from '../ServerCard';
import AgentCard, { Agent } from '../AgentCard';
import SkillCard from '../SkillCard';
import VirtualServerCard from '../VirtualServerCard';
import CustomEntityCard from '../CustomEntityCard';
import type { Skill } from '../../types/skill';
import type { VirtualServerInfo } from '../../types/virtualServer';
import type {
  CustomTypeDescriptor,
  CustomEntityRecord,
} from '../../types/customEntity';

// react-markdown / remark-gfm are ESM-only; mock them (matches SkillResources.test
// and SemanticSearchResults.test).
jest.mock('react-markdown', () => {
  return { __esModule: true, default: ({ children }: { children?: React.ReactNode }) => <>{children}</> };
});
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));

// ServerCard reads auth context; provide a stable admin user.
jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: {
      username: 'admin',
      is_admin: true,
      can_modify_servers: true,
      ui_permissions: {},
    },
  }),
}));

// Stub every heavy child so the snapshot is deterministic and child-internal
// changes never churn these snapshots.
jest.mock('axios');
jest.mock('../ServerConfigModal', () => () => null);
jest.mock('../SecurityScanModal', () => () => null);
jest.mock('../VersionSelectorModal', () => () => null);
jest.mock('../ServerDetailsModal', () => () => null);
jest.mock('../AgentDetailsModal', () => () => null);
jest.mock('../PullCardPreviewModal', () => () => null);
jest.mock('../SkillResources', () => () => null);
jest.mock('../ResourceBoundTokenButton', () => () => null);
jest.mock('../StarRatingWidget', () => () => <div data-testid="stars" />);

const noop = () => {};
const asyncNoop = async () => {};


/**
 * Serialize a rendered card to HTML with each element's class tokens sorted.
 *
 * Tailwind class ORDER in the className attribute does not affect appearance
 * (the cascade is decided by the generated CSS, not attribute order), so the
 * snapshot must compare the SET of classes, not the exact string. This lets a
 * card refactored to compose primitives (which emit classes in a different
 * order via clsx) match main's inline markup as long as the classes are the
 * same. Real visual differences (different/added/removed classes, changed
 * structure) still fail the snapshot.
 */
function normalizedHtml(container: HTMLElement): string {
  const clone = container.cloneNode(true) as HTMLElement;
  clone.querySelectorAll('[class]').forEach((el) => {
    const sorted = Array.from(el.classList).sort().join(' ');
    el.setAttribute('class', sorted);
  });
  return clone.innerHTML;
}


const server = {
  server_name: 'Snapshot Server',
  path: '/snapshot-server',
  description: 'A server used for snapshot testing',
  proxy_pass_url: 'http://localhost:9000',
  is_enabled: true,
  enabled: true,
  tags: ['alpha', 'beta', 'gamma'],
  num_tools: 3,
  num_stars: 0,
  rating_details: [],
  health_status: 'healthy',
} as unknown as Server;


const agent = {
  name: 'Snapshot Agent',
  path: '/snapshot-agent',
  description: 'An agent used for snapshot testing',
  status: 'healthy',
  enabled: true,
  tags: ['alpha', 'beta'],
  num_stars: 0,
  rating_details: [],
  supportedProtocol: 'a2a',
  visibility: 'public',
} as unknown as Agent;


const skill = {
  name: 'Snapshot Skill',
  path: '/snapshot-skill',
  description: 'A skill used for snapshot testing',
  skill_md_url: 'http://localhost/skill.md',
  visibility: 'public',
  is_enabled: true,
  tags: ['alpha', 'beta'],
  num_stars: 0,
  rating_details: [],
} as unknown as Skill;


const virtualServer = {
  path: '/snapshot-vserver',
  server_name: 'Snapshot Virtual Server',
  description: 'A virtual server used for snapshot testing',
  is_enabled: true,
  tags: ['alpha', 'beta'],
  tool_count: 2,
  num_stars: 0,
  rating_details: [],
} as unknown as VirtualServerInfo;


const customDescriptor = {
  name: 'gadget',
  display_name: 'Gadget',
  description: 'A custom type',
  fields: [],
  schema_version: 1,
  created_at: '2026-01-01T00:00:00Z',
} as CustomTypeDescriptor;


const customRecord = {
  path: '/gadget/snapshot',
  entity_type: 'gadget',
  name: 'Snapshot Gadget',
  description: 'A custom record used for snapshot testing',
  visibility: 'public',
  allowed_groups: [],
  tags: ['alpha', 'beta'],
  is_enabled: true,
  num_stars: 0,
  rating_details: [],
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
  attributes: {},
} as CustomEntityRecord;


describe('Entity card DOM snapshots (lock main markup before refactor)', () => {
  it('ServerCard markup is stable', () => {
    const { container } = render(
      <ServerCard
        server={server}
        onToggle={noop}
        onEdit={noop}
        canModify
        canHealthCheck
        canToggle
        canDelete
        onDelete={asyncNoop}
      />,
    );
    expect(normalizedHtml(container)).toMatchSnapshot();
  });

  it('AgentCard markup is stable', () => {
    const { container } = render(
      <AgentCard
        agent={agent}
        onToggle={noop}
        onEdit={noop}
        canModify
        canHealthCheck
        canToggle
        canDelete
        onDelete={asyncNoop}
      />,
    );
    expect(normalizedHtml(container)).toMatchSnapshot();
  });

  it('SkillCard markup is stable', () => {
    const { container } = render(
      <SkillCard
        skill={skill}
        onToggle={noop}
        onEdit={noop}
        onDelete={noop}
        canModify
        canToggle
        canHealthCheck
      />,
    );
    expect(normalizedHtml(container)).toMatchSnapshot();
  });

  it('VirtualServerCard markup is stable', () => {
    const { container } = render(
      <VirtualServerCard
        virtualServer={virtualServer}
        canModify
        onToggle={noop}
        onEdit={noop}
        onDelete={noop}
      />,
    );
    expect(normalizedHtml(container)).toMatchSnapshot();
  });

  it('CustomEntityCard markup is stable', () => {
    const { container } = render(
      <CustomEntityCard
        descriptor={customDescriptor}
        record={customRecord}
        canModify
        onView={noop}
        onEdit={noop}
        onDelete={noop}
      />,
    );
    expect(normalizedHtml(container)).toMatchSnapshot();
  });
});
