/**
 * AllowedToolsBuilder - Component for building allowed_tools list for agents
 * Allows multi-server tool selection with duplicate warnings
 */

import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { ToolPicker } from '@/components/common/ToolPicker';
import { ServerWithTools } from '@/features/servers/hooks';
import { XMarkIcon } from '@heroicons/react/24/outline';

export interface AllowedToolsBuilderProps {
  serversWithTools: ServerWithTools[];
  initialTools?: string[];
  onApply?: (tools: string[]) => void;
  showApplyButton?: boolean;
}

/**
 * AllowedToolsBuilder component
 */
export function AllowedToolsBuilder({
  serversWithTools,
  initialTools = [],
  onApply,
  showApplyButton = true,
}: AllowedToolsBuilderProps) {
  const navigate = useNavigate();
  const [selectedTools, setSelectedTools] = useState<string[]>(initialTools);

  const handleApply = () => {
    if (onApply) {
      onApply(selectedTools);
    } else {
      // Navigate to agent create page with pre-filled tools
      navigate('/agents/new', { state: { allowedTools: selectedTools } });
    }
  };

  const handleRemoveTool = (toolName: string) => {
    setSelectedTools(selectedTools.filter((t) => t !== toolName));
  };

  const handleClearAll = () => {
    setSelectedTools([]);
  };

  return (
    <div className="space-y-6">
      {/* Tool Picker */}
      <Card>
        <div className="p-4">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
            Select Tools
          </h3>
          <ToolPicker
            serversWithTools={serversWithTools}
            selectedTools={selectedTools}
            onChange={setSelectedTools}
            showDuplicateWarning={true}
          />
        </div>
      </Card>

      {/* Selected tools summary */}
      {selectedTools.length > 0 && (
        <Card>
          <div className="p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                Selected Tools ({selectedTools.length})
              </h3>
              <Button variant="ghost" size="sm" onClick={handleClearAll}>
                Clear All
              </Button>
            </div>

            <div className="flex flex-wrap gap-2">
              {selectedTools.map((toolName) => {
                // Find which server(s) have this tool
                const serversWithThisTool = serversWithTools.filter((swt) =>
                  swt.tools?.some((t) => t.name === toolName)
                );

                const isDuplicate = serversWithThisTool.length > 1;

                return (
                  <div
                    key={toolName}
                    className="flex items-center gap-2 px-3 py-2 bg-gray-100 dark:bg-gray-800 rounded-lg"
                  >
                    <span className="text-sm font-mono text-gray-900 dark:text-gray-100">
                      {toolName}
                    </span>
                    {isDuplicate && (
                      <Badge variant="warning" size="sm">
                        {serversWithThisTool.length} servers
                      </Badge>
                    )}
                    <button
                      type="button"
                      onClick={() => handleRemoveTool(toolName)}
                      className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    >
                      <XMarkIcon className="h-4 w-4" />
                    </button>
                  </div>
                );
              })}
            </div>

            {showApplyButton && (
              <div className="mt-4 flex justify-end">
                <Button
                  variant="primary"
                  onClick={handleApply}
                  disabled={selectedTools.length === 0}
                >
                  Apply to Agent
                </Button>
              </div>
            )}
          </div>
        </Card>
      )}

      {/* Help text */}
      <div className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
        <p>
          <strong>What are allowed_tools?</strong> This list restricts which tools an agent can
          access. If empty or null, the agent can access all tools permitted by their scopes.
        </p>
        <p>
          <strong>Duplicate tool names:</strong> If the same tool name appears in multiple
          servers, the agent may access any of them. Consider using unique naming or server-specific
          scopes to control access.
        </p>
      </div>
    </div>
  );
}
