import React, { createContext, useContext } from 'react';
import type { EgressCardState } from '../utils/egressAuth';

/**
 * Shared per-user egress connect state for server cards, so every place that
 * renders a ServerCard (dashboard grid, discover/search rows) shows the same
 * "connect your account" affordance without threading props through each render
 * path. ServerCard reads this context by server path; the provider (Dashboard)
 * owns the fetch + refresh.
 */
interface EgressConnectContextValue {
  // Per-server-path egress state; empty map when the feature is off or the
  // caller is not a per-user principal.
  stateByPath: Map<string, EgressCardState>;
  // Re-fetch the state after a connect/disconnect (e.g. from the modal callout).
  reload: () => void;
}

const EgressConnectContext = createContext<EgressConnectContextValue>({
  stateByPath: new Map(),
  reload: () => {},
});


export const EgressConnectProvider = EgressConnectContext.Provider;


/** Read the shared egress connect state (state map + reload callback). */
export function useEgressConnect(): EgressConnectContextValue {
  return useContext(EgressConnectContext);
}
