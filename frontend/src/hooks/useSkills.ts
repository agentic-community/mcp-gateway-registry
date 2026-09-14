import { useState, useEffect, useCallback } from 'react';
import { Skill } from '../types/skill';
import { fetchAllPages } from '../utils/fetchAllPages';

export type { Skill } from '../types/skill';

interface UseSkillsReturn {
  skills: Skill[];
  setSkills: React.Dispatch<React.SetStateAction<Skill[]>>;
  loading: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
}

export const useSkills = (): UseSkillsReturn => {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Issue #880: page through /api/skills (max 2000 per request)
      const skillsList = await fetchAllPages<any>({
        url: '/api/skills',
        itemsKey: 'skills',
        params: { include_disabled: true },
      });

      console.log(`Skills returned from API: ${skillsList.length}`);

      // Transform skill data from backend format to frontend format
      const transformedSkills: Skill[] = skillsList.map((skillInfo: any) => ({
        name: skillInfo.name || 'Unknown Skill',
        path: skillInfo.path,
        description: skillInfo.description || '',
        skill_md_url: skillInfo.skill_md_url || '',
        skill_md_raw_url: skillInfo.skill_md_raw_url || '',
        repository_url: skillInfo.repository_url,
        version: skillInfo.version,
        author: skillInfo.author,
        visibility: skillInfo.visibility || 'public',
        is_enabled: skillInfo.is_enabled !== undefined ? skillInfo.is_enabled : true,
        tags: skillInfo.tags || [],
        owner: skillInfo.owner,
        registry_name: skillInfo.registry_name || 'local',
        target_agents: skillInfo.target_agents || [],
        allowed_tools: skillInfo.allowed_tools || [],
        requirements: skillInfo.requirements || [],
        metadata: skillInfo.metadata || null,
        auth_scheme: skillInfo.auth_scheme || 'none',
        auth_header_name: skillInfo.auth_header_name || undefined,
        num_stars: skillInfo.num_stars || 0,
        rating_details: skillInfo.rating_details || [],
        security_scan: skillInfo.security_scan ?? null,
        status: skillInfo.status || 'active',
        health_status: skillInfo.health_status || 'unknown',
        last_checked_time: skillInfo.last_checked_time,
        created_at: skillInfo.created_at,
        updated_at: skillInfo.updated_at,
        // ARD discovery imports: preserve the read-only marker and source
        // descriptor URL so the SkillCard can render the discovery treatment.
        is_read_only: skillInfo.is_read_only ?? false,
        ard_source_url: skillInfo.ard_source_url,
        // Gateway-proxy opt-in. Carry through so the card badge and edit modal
        // reflect the stored state. proxy_client_url is the read-only,
        // auto-derived client path; proxy_target_url is the origin/backend URL.
        is_proxied: skillInfo.is_proxied ?? false,
        proxy_target_url: skillInfo.proxy_target_url ?? undefined,
        proxy_client_url: skillInfo.proxy_client_url ?? undefined,
      }));

      setSkills(transformedSkills);
    } catch (err: any) {
      console.error('Failed to fetch skills data:', err);
      setError(err.response?.data?.detail || 'Failed to fetch skills');
      setSkills([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    skills,
    setSkills,
    loading,
    error,
    refreshData: fetchData,
  };
};
