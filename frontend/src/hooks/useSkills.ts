import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

export interface Skill {
  name: string;
  path: string;
  description?: string;
  skill_md_url: string;
  version?: string;
  author?: string;
  visibility: 'public' | 'private' | 'group';
  is_enabled: boolean;
  tags?: string[];
  owner?: string;
  registry_name?: string;
  target_agents?: string[];
  allowed_tools?: Array<{
    tool_name: string;
    server_path?: string;
    capabilities?: string[];
  }>;
  requirements?: Array<{
    type: string;
    target: string;
    min_version?: string;
    required?: boolean;
  }>;
  created_at?: string;
  updated_at?: string;
}

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

      const response = await axios.get('/api/skills');

      // The API returns {"skills": [...]}
      const responseData = response.data || {};
      const skillsList = responseData.skills || [];

      console.log(`Skills returned from API: ${skillsList.length}`);

      // Transform skill data from backend format to frontend format
      const transformedSkills: Skill[] = skillsList.map((skillInfo: any) => ({
        name: skillInfo.name || 'Unknown Skill',
        path: skillInfo.path,
        description: skillInfo.description || '',
        skill_md_url: skillInfo.skill_md_url || '',
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
        created_at: skillInfo.created_at,
        updated_at: skillInfo.updated_at,
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
