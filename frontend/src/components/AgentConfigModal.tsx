import React, { useCallback, useState } from 'react';
import {
  ClipboardDocumentIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import type { Agent } from './AgentCard';

type IDE = 'vscode' | 'cursor' | 'cline' | 'claude-code';
type CliCommand =
  | 'register'
  | 'list'
  | 'get'
  | 'test'
  | 'test-all'
  | 'search'
  | 'update'
  | 'toggle'
  | 'delete';

interface AgentConfigModalProps {
  agent: Agent & { [key: string]: any };
  isOpen: boolean;
  onClose: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}

const AgentConfigModal: React.FC<AgentConfigModalProps> = ({
  agent,
  isOpen,
  onClose,
  onShowToast,
}) => {
  const [selectedIDE, setSelectedIDE] = useState<IDE>('vscode');
  const [selectedCliCommand, setSelectedCliCommand] = useState<CliCommand>('list');

  const generateCliCommand = useCallback((): string => {
    const agentPath = agent.path;
    const agentName = agent.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    switch (selectedCliCommand) {
      case 'list':
        return 'uv run python cli/agent_mgmt.py list';
      case 'get':
        return `uv run python cli/agent_mgmt.py get ${agentPath}`;
      case 'test':
        return `uv run python cli/agent_mgmt.py test ${agentPath}`;
      case 'test-all':
        return 'uv run python cli/agent_mgmt.py test-all';
      case 'search':
        return `uv run python cli/agent_mgmt.py search \"${agent.name}\"`;
      case 'update':
        return `uv run python cli/agent_mgmt.py update ${agentPath} cli/examples/${agentName}_agent.json`;
      case 'toggle':
        return (
          `uv run python cli/agent_mgmt.py toggle ${agentPath} true   # Enable\n` +
          `uv run python cli/agent_mgmt.py toggle ${agentPath} false  # Disable`
        );
      case 'delete':
        return `uv run python cli/agent_mgmt.py delete ${agentPath}`;
      case 'register':
      default:
        return `uv run python cli/agent_mgmt.py register cli/examples/${agentName}_agent.json`;
    }
  }, [selectedCliCommand, agent.path, agent.name]);

  const getCliCommandDescription = useCallback((): string => {
    switch (selectedCliCommand) {
      case 'list':
        return 'List all agents (filtered by your permissions). Shows agent summaries with basic metadata.';
      case 'get':
        return 'Retrieve the complete agent card for a specific agent, including all metadata, skills, and configuration.';
      case 'test':
        return 'Verify agent registration in registry and test endpoint accessibility. Checks health status.';
      case 'test-all':
        return 'Test all registered agents to verify they are accessible and healthy.';
      case 'search':
        return 'Perform semantic search to find agents by capability. Searches by name, description, and tags.';
      case 'update':
        return "Update an agent's metadata and configuration. Requires the agent JSON file with updated values.";
      case 'toggle':
        return 'Enable or disable an agent in the registry without deleting it. Useful for maintenance.';
      case 'delete':
        return 'Remove an agent from the registry. This action is permanent.';
      case 'register':
      default:
        return 'Register a new agent with the registry. Requires a properly formatted agent JSON file with metadata, skills, and security schemes.';
    }
  }, [selectedCliCommand]);

  const generateAgentConfig = useCallback(() => {
    const agentName = agent.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    const currentUrl = new URL(window.location.origin);
    const baseUrl = `${currentUrl.protocol}//${currentUrl.hostname}`;

    const cleanPath = agent.path.replace(/\/+$/, '').replace(/^\/+/, '/');
    const url = `${baseUrl}${cleanPath}/a2a`;

    switch (selectedIDE) {
      case 'vscode':
        return {
          agents: {
            [agentName]: {
              type: 'a2a',
              url,
              version: agent.version || '1.0.0',
              trust_level: agent.trust_level || 'community',
              headers: {
                Authorization: 'Bearer [YOUR_AUTH_TOKEN]',
              },
            },
          },
          inputs: [
            {
              type: 'promptString',
              id: 'auth-token',
              description: 'Gateway Authentication Token',
            },
          ],
        };
      case 'cursor':
        return {
          a2aAgents: {
            [agentName]: {
              url,
              version: agent.version || '1.0.0',
              trust_level: agent.trust_level || 'community',
              headers: {
                Authorization: 'Bearer [YOUR_AUTH_TOKEN]',
              },
            },
          },
        };
      case 'cline':
        return {
          a2aAgents: {
            [agentName]: {
              type: 'a2a',
              url,
              version: agent.version || '1.0.0',
              trust_level: agent.trust_level || 'community',
              disabled: false,
              headers: {
                Authorization: 'Bearer [YOUR_AUTH_TOKEN]',
              },
            },
          },
        };
      case 'claude-code':
        return {
          a2aAgents: {
            [agentName]: {
              type: 'a2a',
              url,
              version: agent.version || '1.0.0',
              trust_level: agent.trust_level || 'community',
              headers: {
                Authorization: 'Bearer [YOUR_AUTH_TOKEN]',
              },
            },
          },
        };
      default:
        return {
          a2aAgents: {
            [agentName]: {
              type: 'a2a',
              url,
              version: agent.version || '1.0.0',
              trust_level: agent.trust_level || 'community',
              headers: {
                Authorization: 'Bearer [YOUR_AUTH_TOKEN]',
              },
            },
          },
        };
    }
  }, [agent.name, agent.path, agent.version, agent.trust_level, selectedIDE]);

  const copyConfigToClipboard = useCallback(async () => {
    try {
      const config = generateAgentConfig();
      const configText = JSON.stringify(config, null, 2);
      await navigator.clipboard.writeText(configText);
      onShowToast?.('Agent configuration copied to clipboard!', 'success');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      onShowToast?.('Failed to copy configuration', 'error');
    }
  }, [generateAgentConfig, onShowToast]);

  const copyCliCommandToClipboard = useCallback(async () => {
    try {
      const command = generateCliCommand();
      await navigator.clipboard.writeText(command);
      onShowToast?.('CLI command copied to clipboard!', 'success');
    } catch (error) {
      console.error('Failed to copy CLI command:', error);
      onShowToast?.('Failed to copy CLI command', 'error');
    }
  }, [generateCliCommand, onShowToast]);

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-3xl w-full mx-4 max-h-[80vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            A2A Agent Configuration for {agent.name}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          <div className="bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-800 rounded-lg p-4">
            <h4 className="font-medium text-cyan-900 dark:text-cyan-100 mb-2">How to use this configuration:</h4>
            <ol className="text-sm text-cyan-800 dark:text-cyan-200 space-y-1 list-decimal list-inside">
              <li>Copy the configuration below</li>
              <li>Paste it into your agent configuration file</li>
              <li>
                Replace <code className="bg-cyan-100 dark:bg-cyan-800 px-1 rounded">[YOUR_AUTH_TOKEN]</code> with your
                gateway authentication token
              </li>
              <li>Restart your AI coding assistant to load the new agent</li>
            </ol>
          </div>

          <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
            <h4 className="font-medium text-amber-900 dark:text-amber-100 mb-2">Authentication Required</h4>
            <p className="text-sm text-amber-800 dark:text-amber-200">
              This configuration requires gateway authentication tokens. The tokens authenticate your AI assistant with
              the MCP Gateway, not the individual agent. Visit the authentication documentation for setup instructions.
            </p>
          </div>

          <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
            <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-3">CLI Agent Management Commands</h4>
            <p className="text-sm text-purple-800 dark:text-purple-200 mb-4">
              Manage agents from the command line using the A2A CLI. Reference the{' '}
              <a
                href="https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/a2a-agent-management.md"
                target="_blank"
                rel="noopener noreferrer"
                className="font-semibold hover:underline"
              >
                A2A Agent Management Guide
              </a>{' '}
              for complete documentation.
            </p>

            <div className="mb-3">
              <label className="block text-sm font-medium text-purple-900 dark:text-purple-100 mb-2">
                Select Command:
              </label>
              <select
                value={selectedCliCommand}
                onChange={(e) => setSelectedCliCommand(e.target.value as CliCommand)}
                className="w-full px-3 py-2 border border-purple-300 dark:border-purple-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              >
                <option value="list">List agents</option>
                <option value="get">Get agent details</option>
                <option value="test">Test specific agent</option>
                <option value="test-all">Test all agents</option>
                <option value="search">Search agents</option>
                <option value="update">Update agent from file</option>
                <option value="toggle">Enable/Disable agent</option>
                <option value="delete">Delete agent</option>
                <option value="register">Register agent from file</option>
              </select>
            </div>

            <div className="flex items-start gap-3 bg-purple-100 dark:bg-purple-900/40 rounded-lg p-3">
              <InformationCircleIcon className="h-5 w-5 text-purple-700 dark:text-purple-200" />
              <p className="text-sm text-purple-800 dark:text-purple-200">
                Commands run from the repository root. Make sure you have authenticated with the gateway before executing
                CLI operations.
              </p>
            </div>

            <div className="mt-3">
              <div className="flex items-center justify-between mb-2">
                <h5 className="font-medium text-purple-900 dark:text-purple-100">Command Preview</h5>
                <button
                  onClick={copyCliCommandToClipboard}
                  className="flex items-center gap-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm"
                >
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  Copy Command
                </button>
              </div>
              <pre className="bg-gray-900 text-purple-100 p-3 rounded-lg text-xs overflow-x-auto">
                {generateCliCommand()}
              </pre>
              <p className="mt-2 text-sm text-purple-800 dark:text-purple-200">
                <strong>Description:</strong> {getCliCommandDescription()}
              </p>
            </div>
          </div>

          <div className="bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">Select your IDE/Tool:</h4>
            <div className="flex flex-wrap gap-2">
              {(['vscode', 'cursor', 'cline', 'claude-code'] as IDE[]).map((ide) => (
                <button
                  key={ide}
                  onClick={() => setSelectedIDE(ide)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedIDE === ide
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  {ide === 'vscode'
                    ? 'VS Code'
                    : ide === 'cursor'
                    ? 'Cursor'
                    : ide === 'cline'
                    ? 'Cline'
                    : 'Claude Code'}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              Configuration format optimized for{' '}
              {selectedIDE === 'vscode'
                ? 'VS Code'
                : selectedIDE === 'cursor'
                ? 'Cursor'
                : selectedIDE === 'cline'
                ? 'Cline'
                : 'Claude Code'}{' '}
              integration
            </p>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-gray-900 dark:text-white">Configuration JSON:</h4>
              <button
                onClick={copyConfigToClipboard}
                className="flex items-center gap-2 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors duration-200"
              >
                <ClipboardDocumentIcon className="h-4 w-4" />
                Copy to Clipboard
              </button>
            </div>
            <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto">
              {JSON.stringify(generateAgentConfig(), null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentConfigModal;
