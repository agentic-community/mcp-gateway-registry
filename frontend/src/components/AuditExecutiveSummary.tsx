import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  ArrowPathIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  ServerStackIcon,
  UsersIcon,
  CpuChipIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

interface GovernanceScope {
  mcp_servers_governed: number;
  tools_under_policy: number;
  identities_active: number;
}

interface RegisteredAssets {
  servers: number;
  tools: number;
  agents: number;
  skills: number;
  custom_entities: number;
}

interface ActiveUsers {
  dau: number;
  wau: number;
  mau: number;
  wau_available: boolean;
  mau_available: boolean;
}

interface ActiveAgents {
  daa: number;
  waa: number;
  maa: number;
  waa_available: boolean;
  maa_available: boolean;
}

interface TrafficSplit {
  human_events: number;
  agent_events: number;
  human_pct: number;
  agent_pct: number;
}

interface AdoptionMomentum {
  events_current: number;
  events_prior: number;
  events_wow_pct: number | null;
  active_identities_current: number;
  active_identities_prior: number;
  active_agents_current: number;
  active_agents_prior: number;
  has_prior_data: boolean;
}

interface ExecutiveSummaryResponse {
  window_days: number;
  retention_days: number;
  governance: GovernanceScope;
  registered_assets: RegisteredAssets;
  active_users: ActiveUsers;
  active_agents: ActiveAgents;
  traffic_split: TrafficSplit;
  momentum: AdoptionMomentum;
}

const SUMMARY_DAYS = 7;

// Format a count with its singular or plural noun: 1 identity, 2 identities.
function _plural(count: number, singular: string, plural: string): string {
  const noun = count === 1 ? singular : plural;
  return `${count.toLocaleString()} ${noun}`;
}

// Build a plain-English summary sentence from the live metrics, so a reader
// can follow the band without decoding each tile. Returns null when there is
// no governed activity to describe.
function _buildNarrative(data: ExecutiveSummaryResponse): string | null {
  const g = data.governance;
  const days = data.window_days;
  if (g.identities_active === 0 && g.mcp_servers_governed === 0) {
    return null;
  }
  const identities = _plural(g.identities_active, 'identity', 'identities');
  const agents = _plural(data.active_agents.waa, 'agent', 'agents');
  const servers = _plural(g.mcp_servers_governed, 'MCP server', 'MCP servers');
  const tools = _plural(g.tools_under_policy, 'tool', 'tools');
  const calls = data.momentum.events_current.toLocaleString();
  const agentPct = data.traffic_split.agent_pct;
  return (
    `In the last ${days} days, ${identities} (including ${agents}) accessed ${servers} ` +
    `exposing ${tools} under policy, generating ${calls} governed calls ` +
    `(${agentPct}% of those calls came from agents).`
  );
}

// A single hero tile: large number, uppercase label, optional sub-line.
const HeroTile: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  subLine?: React.ReactNode;
  title?: string;
}> = ({ icon, label, value, subLine, title }) => {
  return (
    <div
      className="border border-gray-100 dark:border-gray-700 rounded-lg p-3"
      title={title}
    >
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-gray-400 dark:text-gray-500">{icon}</span>
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
          {label}
        </span>
      </div>
      <div className="text-2xl font-bold text-gray-900 dark:text-white">{value}</div>
      {subLine ? (
        <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{subLine}</div>
      ) : null}
    </div>
  );
};

// One side of the Active tile: a weekly headline count with daily and monthly
// sub-counts. Monthly is shown only when retention covers the 30-day window.
const ActiveCountColumn: React.FC<{
  label: string;
  weekly: number;
  daily: number;
  monthly: number;
  monthlyAvailable: boolean;
  dailyAbbr: string;
  monthlyAbbr: string;
  retentionDays: number;
}> = ({
  label,
  weekly,
  daily,
  monthly,
  monthlyAvailable,
  dailyAbbr,
  monthlyAbbr,
  retentionDays,
}) => {
  return (
    <div
      title={
        monthlyAvailable
          ? `Distinct ${label.toLowerCase()} active per day / week / month`
          : `${monthlyAbbr} hidden: audit retention is ${retentionDays}d (needs 30d).`
      }
    >
      <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-0.5">
        {label}
      </div>
      <div className="text-2xl font-bold text-gray-900 dark:text-white">
        {weekly.toLocaleString()}
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
        {dailyAbbr} {daily.toLocaleString()}
        {monthlyAvailable ? (
          <>
            {' '}
            &middot; {monthlyAbbr} {monthly.toLocaleString()}
          </>
        ) : (
          <span className="text-gray-400 dark:text-gray-500 italic">
            {' '}
            &middot; {monthlyAbbr} needs 30d
          </span>
        )}
      </div>
    </div>
  );
};

// Colored WoW delta chip: green up for positive, red down for negative.
const DeltaChip: React.FC<{ pct: number }> = ({ pct }) => {
  const isUp = pct >= 0;
  const colorClasses = isUp
    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
    : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
  const sign = isUp ? '+' : '';

  return (
    <span
      className={`inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-xs font-medium ${colorClasses}`}
    >
      {isUp ? (
        <ArrowTrendingUpIcon className="h-3 w-3" />
      ) : (
        <ArrowTrendingDownIcon className="h-3 w-3" />
      )}
      {sign}
      {pct.toFixed(0)}% WoW
    </span>
  );
};

// Thin two-segment bar: agent (purple) vs human (blue).
const TrafficSplitBar: React.FC<{ split: TrafficSplit }> = ({ split }) => {
  const total = split.human_events + split.agent_events;
  if (total === 0) {
    return (
      <span className="text-base font-medium text-gray-400 dark:text-gray-500 italic">
        No governed traffic yet
      </span>
    );
  }

  return (
    <div>
      <div className="text-2xl font-bold text-gray-900 dark:text-white">
        {split.agent_pct}%
        <span className="text-sm font-normal text-gray-400 dark:text-gray-500">
          {' '}
          of calls
        </span>
      </div>
      <div className="flex h-2.5 rounded-full overflow-hidden bg-gray-100 dark:bg-gray-700 my-1.5">
        {split.agent_events > 0 ? (
          <div
            className="bg-purple-500"
            style={{ width: `${split.agent_pct}%` }}
            title={`Agent: ${split.agent_events.toLocaleString()} (${split.agent_pct}%)`}
          />
        ) : null}
        {split.human_events > 0 ? (
          <div
            className="bg-blue-500"
            style={{ width: `${split.human_pct}%` }}
            title={`Human: ${split.human_events.toLocaleString()} (${split.human_pct}%)`}
          />
        ) : null}
      </div>
      <div className="text-xs text-gray-500 dark:text-gray-400">
        {split.agent_pct}% of calls from agents &middot; {split.human_pct}% from humans
      </div>
    </div>
  );
};


const AuditExecutiveSummary: React.FC = () => {
  const [data, setData] = useState<ExecutiveSummaryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSummary = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get('/api/audit/executive-summary', {
        params: { days: SUMMARY_DAYS },
      });
      setData(res.data);
    } catch (err) {
      console.error('Failed to fetch executive summary:', err);
      setError('Failed to load executive summary');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 mb-6">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-2">
          <ShieldCheckIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Executive Summary
          </h3>
          <span className="text-xs text-gray-400 ml-2">Last {SUMMARY_DAYS} days</span>
        </div>
        <button
          onClick={fetchSummary}
          disabled={loading}
          className="p-1.5 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors disabled:opacity-50"
          title="Refresh executive summary"
        >
          <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Content */}
      <div className="px-4 pb-4">
        {loading && !data ? (
          <div className="flex items-center justify-center py-8">
            <ArrowPathIcon className="h-6 w-6 text-gray-400 animate-spin" />
            <span className="ml-2 text-sm text-gray-400">Loading executive summary...</span>
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <p className="text-sm text-red-500">{error}</p>
            <button
              onClick={fetchSummary}
              className="mt-2 text-sm text-blue-500 hover:text-blue-600"
            >
              Retry
            </button>
          </div>
        ) : data ? (
          <>
            {/* Plain-English summary line so the tiles read as a story */}
            {_buildNarrative(data) ? (
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
                {_buildNarrative(data)}
              </p>
            ) : null}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {/* Tile 1: Registered assets (total catalog inventory, not
                activity-scoped). Headline = servers, with the rest of the
                inventory on the sub-line. */}
            <HeroTile
              icon={<ServerStackIcon className="h-4 w-4" />}
              label="Registered Assets"
              value={
                <span className="whitespace-nowrap">
                  {data.registered_assets.servers.toLocaleString()}
                  <span className="text-sm font-normal text-gray-400 dark:text-gray-500">
                    {' '}
                    servers
                  </span>
                </span>
              }
              subLine={
                <span>
                  {data.registered_assets.tools.toLocaleString()} tools &middot;{' '}
                  {data.registered_assets.agents.toLocaleString()} agents &middot;{' '}
                  {data.registered_assets.skills.toLocaleString()} skills &middot;{' '}
                  {data.registered_assets.custom_entities.toLocaleString()} custom
                </span>
              }
              title="Total registered inventory in the catalog: servers, the tools they expose, agents, skills, and custom entities"
            />

            {/* Tile 2: Weekly Active Users and Agents, side by side with a
                divider. Headline number is weekly; daily and monthly sit below.
                Monthly is gated on 30-day retention. */}
            <div className="border border-gray-100 dark:border-gray-700 rounded-lg p-3">
              <div className="flex items-center gap-1.5 mb-2">
                <span className="text-gray-400 dark:text-gray-500">
                  <UsersIcon className="h-4 w-4" />
                </span>
                <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Weekly Active
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 divide-x divide-gray-100 dark:divide-gray-700">
                <ActiveCountColumn
                  label="People"
                  weekly={data.active_users.wau}
                  daily={data.active_users.dau}
                  monthly={data.active_users.mau}
                  monthlyAvailable={data.active_users.mau_available}
                  dailyAbbr="DAU"
                  monthlyAbbr="MAU"
                  retentionDays={data.retention_days}
                />
                <div className="pl-3">
                  <ActiveCountColumn
                    label="Agents"
                    weekly={data.active_agents.waa}
                    daily={data.active_agents.daa}
                    monthly={data.active_agents.maa}
                    monthlyAvailable={data.active_agents.maa_available}
                    dailyAbbr="DAA"
                    monthlyAbbr="MAA"
                    retentionDays={data.retention_days}
                  />
                </div>
              </div>
            </div>

            {/* Tile 3: Adoption momentum */}
            <HeroTile
              icon={<CpuChipIcon className="h-4 w-4" />}
              label="Events (7d)"
              value={
                <span className="flex items-center gap-2 flex-wrap">
                  {data.momentum.events_current.toLocaleString()}
                  {data.momentum.has_prior_data && data.momentum.events_wow_pct !== null ? (
                    <DeltaChip pct={data.momentum.events_wow_pct} />
                  ) : null}
                </span>
              }
              subLine={
                <span>
                  {data.momentum.has_prior_data ? (
                    <>
                      Active agents: {data.momentum.active_agents_current.toLocaleString()} (up from{' '}
                      {data.momentum.active_agents_prior.toLocaleString()})
                    </>
                  ) : (
                    <>
                      {data.momentum.active_agents_current.toLocaleString()} active agents
                      <span className="text-gray-400 dark:text-gray-500 italic">
                        {' '}
                        &middot; no prior-week data
                      </span>
                    </>
                  )}
                </span>
              }
            />

            {/* Tile 4: Human vs Agent split. "Agent" = non-interactive,
                token-based callers (bearer token); "Human" = interactive web
                UI sessions (session cookie). Anonymous traffic is excluded. */}
            <HeroTile
              icon={<CpuChipIcon className="h-4 w-4" />}
              label="Agent vs Human"
              value={<TrafficSplitBar split={data.traffic_split} />}
              title="Share of governed CALLS (requests), not identities. Agent = token-based (non-interactive) callers; Human = interactive web UI sessions. Anonymous traffic excluded."
            />
            </div>

            {/* Legend: make the agent vs human definition explicit so no one
                misreads the split. Any non-web-UI access counts as an agent. */}
            <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400 dark:text-gray-500 border-t border-gray-100 dark:border-gray-700 pt-3">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                People = interactive web UI sessions
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
                Agents = all non-web-UI access (API / token callers, incl. CLI and scripts)
              </span>
              <span>Anonymous traffic (health checks, probes) is excluded.</span>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
};

export default AuditExecutiveSummary;
