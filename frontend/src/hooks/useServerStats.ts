// Server stats are now backed by a shared context provider so that the
// triple `limit=2000` fetch runs once per dashboard view instead of once
// per consumer (Layout + Dashboard previously each ran their own fetch).
// The hook is re-exported here to keep existing import paths working.
export { useServerStats, ServerStatsProvider } from '../contexts/ServerStatsContext';
