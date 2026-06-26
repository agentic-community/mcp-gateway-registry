import React from 'react';
import EntityGrid from '../EntityGrid';

interface ExternalRegistriesSectionProps {
  /** Source tab keys (e.g. 'anthropic', 'aws_registry'). Empty hides the tabs. */
  availableSources: string[];
  /** Human labels for source keys. */
  sourceLabels: Record<string, string>;
  /**
   * Source keys that are ARD (ai-catalog) registries — i.e. dynamic sources
   * derived from a federated peer id rather than the built-in
   * anthropic/asor/aws_registry sources. These get an "ARD" pill in their tab.
   */
  ardSources?: string[];
  activeSource: string | null;
  onSelectSource: (source: string) => void;
  /** Filtered lists actually rendered. */
  servers: Array<{ path: string }>;
  agents: Array<{ path: string }>;
  skills: Array<{ path: string }>;
  /** Raw (pre-filter) counts, to distinguish "none configured" from "none match". */
  hasAnyExternal: boolean;
  /** Card renderers — the Dashboard owns the exact per-card props. */
  renderServerCard: (server: any) => React.ReactNode;
  renderAgentCard: (agent: any) => React.ReactNode;
  renderSkillCard: (skill: any) => React.ReactNode;
}

/**
 * The "External Registries" Dashboard collection: federated source tabs plus
 * server/agent/skill subsections, each a labeled EntityGrid. Card rendering is
 * delegated to the Dashboard via render callbacks so the per-card prop wiring
 * (permissions, handlers) stays where the state lives.
 */
const ExternalRegistriesSection: React.FC<ExternalRegistriesSectionProps> = ({
  availableSources,
  sourceLabels,
  ardSources = [],
  activeSource,
  onSelectSource,
  servers,
  agents,
  skills,
  hasAnyExternal,
  renderServerCard,
  renderAgentCard,
  renderSkillCard,
}) => {
  const nothingToShow =
    servers.length === 0 && agents.length === 0 && skills.length === 0;
  const ardSourceSet = new Set(ardSources);

  return (
    <div className="mb-8">
      <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
        External Registries
      </h2>

      {/* Source tabs - only show when there are sources */}
      {availableSources.length > 0 && (
        <div className="flex border-b border-gray-200 dark:border-gray-700 mb-6">
          {availableSources.map((source) => (
            <button
              key={source}
              onClick={() => onSelectSource(source)}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors flex items-center gap-1.5 ${
                activeSource === source
                  ? 'border-green-500 text-green-600 dark:text-green-400'
                  : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
              }`}
            >
              {sourceLabels[source] || source}
              {/* ARD pill — distinguishes ai-catalog (ARD) registries from the
                  built-in Anthropic/ASOR/AWS sources. */}
              {ardSourceSet.has(source) && (
                <span
                  className="px-1.5 py-0.5 text-[10px] font-semibold rounded-full bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
                  title="ARD (ai-catalog.json) registry"
                >
                  ARD
                </span>
              )}
            </button>
          ))}
        </div>
      )}

      {nothingToShow ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
          <div className="text-gray-400 text-lg mb-2">
            {hasAnyExternal ? 'No Results Found' : 'No External Registries Available'}
          </div>
          <p className="text-gray-500 dark:text-gray-300 text-sm max-w-md mx-auto">
            {hasAnyExternal
              ? 'Press Enter in the search bar to search semantically'
              : 'External registry integrations (Anthropic, AWS Agents, and more) will appear here when configured'}
          </p>
        </div>
      ) : (
        <div>
          {servers.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                Servers
              </h3>
              <EntityGrid>{servers.map((s) => renderServerCard(s))}</EntityGrid>
            </div>
          )}

          {agents.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                Agents
              </h3>
              <EntityGrid>{agents.map((a) => renderAgentCard(a))}</EntityGrid>
            </div>
          )}

          {skills.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                Skills
              </h3>
              <EntityGrid>{skills.map((s) => renderSkillCard(s))}</EntityGrid>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ExternalRegistriesSection;
