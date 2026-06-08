import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import type {
  DuplicateCheckResult,
  ExistingEntity,
  EntityType,
} from '../types/duplicateCheck';
import { useRegistryConfig } from './useRegistryConfig';


/**
 * Payload shape per entity type. The backend exposes one identity URL
 * field per entity (proxy_pass_url for servers, url for agents,
 * skill_md_url for skills); the rest of the payload is shared.
 */
interface CommonCheckFields {
  name: string;
  description: string | null;
  self_path: string | null;
}

export interface ServerCheckPayload extends CommonCheckFields {
  proxy_pass_url: string | null;
}

export interface AgentCheckPayload extends CommonCheckFields {
  url: string | null;
}

export interface SkillCheckPayload extends CommonCheckFields {
  skill_md_url: string | null;
}

export type DuplicateCheckPayload =
  | { entityType: 'mcp_server'; payload: ServerCheckPayload }
  | { entityType: 'a2a_agent'; payload: AgentCheckPayload }
  | { entityType: 'skill'; payload: SkillCheckPayload };


/**
 * Outcome returned by `runCheck`. The caller decides what to do next:
 * - 'show-modal': hook has populated state and mounted the modal.
 *   Caller should return early and let React render.
 * - 'proceed': caller invokes the registration POST. Covers the
 *   no-collision-no-advisory case, the hint-flag-off case, and any
 *   transient network failure (advisory must never block registration).
 *   `notice` is set when the user should see a soft toast.
 * - 'cancelled': another runCheck superseded this one OR the
 *   component unmounted. Caller MUST NOT proceed — a stale callback
 *   would otherwise POST /register after the user has navigated away.
 */
export type DuplicateCheckOutcome =
  | { kind: 'show-modal' }
  | { kind: 'proceed'; notice?: string }
  | { kind: 'cancelled' };


export interface UseDuplicateCheckResult {
  /** URL-collision matches; only non-empty while modal is open. */
  collisionWith: ExistingEntity[];
  /** Similarity-based advisory matches; only non-empty while modal is open. */
  advisoryMatches: ExistingEntity[];
  /** Whether to render the DuplicateCheckModal. */
  showModal: boolean;
  /**
   * Run the duplicate check against the entity's /check-duplicates
   * endpoint. Cancels any prior in-flight call. Always non-blocking:
   * any network/HTTP error resolves to 'proceed'.
   */
  runCheck: (request: DuplicateCheckPayload) => Promise<DuplicateCheckOutcome>;
  /** Close the modal and clear matches. */
  closeModal: () => void;
}


const ENDPOINT_BY_ENTITY: Record<EntityType, string> = {
  mcp_server: '/api/servers/check-duplicates',
  a2a_agent: '/api/agents/check-duplicates',
  skill: '/api/skills/check-duplicates',
};


/**
 * Entity-agnostic advisory duplicate check for registration forms.
 *
 * Server, agent, and skill registration forms all use this hook to
 * pre-flight /api/{entity}/check-duplicates before submitting. The
 * modal renders on a hit; the form submits to /register regardless
 * of what the user picks (proceed / cancel / view existing) — the
 * check is advisory, never a hard block.
 *
 * The hook short-circuits when ``dedup_registration_hint_enabled``
 * is false on /api/config: no network call, no modal, immediate
 * 'proceed' outcome. That setting is the operator's switch for
 * disabling the UI surface entirely.
 */
export function useDuplicateCheck(): UseDuplicateCheckResult {
  const { config } = useRegistryConfig();
  const hintEnabled = config?.dedup_registration_hint_enabled ?? false;

  const [collisionWith, setCollisionWith] = useState<ExistingEntity[]>([]);
  const [advisoryMatches, setAdvisoryMatches] = useState<ExistingEntity[]>([]);
  const [showModal, setShowModal] = useState<boolean>(false);

  const abortRef = useRef<AbortController | null>(null);

  // Abort any in-flight check on unmount so a Promise resolution
  // arriving after navigation cannot mistakenly proceed with
  // registration.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const runCheck = useCallback(
    async (request: DuplicateCheckPayload): Promise<DuplicateCheckOutcome> => {
      abortRef.current?.abort();

      // Operator opt-out: skip the network round trip and the modal
      // entirely. Caller proceeds straight to /register.
      if (!hintEnabled) {
        return { kind: 'proceed' };
      }

      const endpoint = ENDPOINT_BY_ENTITY[request.entityType];
      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const response = await axios.post<DuplicateCheckResult>(
          endpoint,
          request.payload,
          { signal: controller.signal },
        );
        const data = response.data;

        if (!data.similarity_search_available) {
          // The exact-match check still ran, but the embedding-backed
          // similarity check was skipped. Surface a soft notice so
          // the user knows the result is partial; still proceed.
          if (
            data.collision_with.length === 0 &&
            data.advisory_matches.length === 0
          ) {
            return {
              kind: 'proceed',
              notice:
                'Similarity check unavailable; proceeded with URL match only.',
            };
          }
        }

        if (
          data.collision_with.length === 0 &&
          data.advisory_matches.length === 0
        ) {
          return { kind: 'proceed' };
        }

        setCollisionWith(data.collision_with);
        setAdvisoryMatches(data.advisory_matches);
        setShowModal(true);
        return { kind: 'show-modal' };
      } catch (error: unknown) {
        const isCancel =
          axios.isCancel(error) ||
          (error as { name?: string })?.name === 'CanceledError' ||
          (error as { name?: string })?.name === 'AbortError';
        if (isCancel) {
          return { kind: 'cancelled' };
        }
        // eslint-disable-next-line no-console
        console.warn('Duplicate check failed; proceeding with registration', error);
        return { kind: 'proceed' };
      }
    },
    [hintEnabled],
  );

  const closeModal = useCallback(() => {
    setShowModal(false);
    setCollisionWith([]);
    setAdvisoryMatches([]);
  }, []);

  return {
    collisionWith,
    advisoryMatches,
    showModal,
    runCheck,
    closeModal,
  };
}
