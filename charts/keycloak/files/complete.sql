\set ON_ERROR_STOP on
SELECT :'mode' = 'restore' AS restoring \gset
\if :restoring
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.realm) OR NOT EXISTS (SELECT 1 FROM public.databasechangelog)
     OR to_regclass('public.user_entity') IS NULL OR to_regclass('public.client') IS NULL THEN
    RAISE EXCEPTION 'Snapshot does not contain initialized Keycloak schema/data';
  END IF;
END $$;
\endif
CREATE SCHEMA _mcp_bootstrap;
REVOKE ALL ON SCHEMA _mcp_bootstrap FROM PUBLIC;
CREATE TABLE _mcp_bootstrap.receipt (intent text NOT NULL);
INSERT INTO _mcp_bootstrap.receipt VALUES (:'intent');
