import React from "react";
import {Box, Text, useInput} from "ink";

export interface ToolConfirmationRequest {
  toolName: string;
  summary: string;
}

interface ToolConfirmationProps {
  request: ToolConfirmationRequest;
  onDecision: (approved: boolean) => void;
}

/**
 * Modal prompt shown before a mutating or shell tool call runs. The operator must
 * explicitly approve (y) or decline (n/Esc). Any other key is ignored so an
 * accidental keystroke cannot approve a destructive action.
 */
export function ToolConfirmation({request, onDecision}: ToolConfirmationProps) {
  useInput((input, key) => {
    const answer = input.toLowerCase();
    if (answer === "y") {
      onDecision(true);
    } else if (answer === "n" || key.escape) {
      onDecision(false);
    }
  });

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>
      <Text bold color="yellow">
        Confirm action requested by the assistant
      </Text>
      <Text>
        <Text dimColor>Tool: </Text>
        {request.toolName}
      </Text>
      <Text>
        <Text dimColor>Action: </Text>
        {request.summary}
      </Text>
      <Text color="cyan">Approve? (y = run, n / Esc = decline)</Text>
    </Box>
  );
}
