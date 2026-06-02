-- ============================================================
-- RBAC - Control de Acceso Basado en Roles (PostgreSQL)
-- JHM Pneumonia Detection System
-- Ejecutar como superusuario en Render PostgreSQL
-- ============================================================

-- 1. ROL DE SOLO LECTURA (para dashboards, Grafana, analytics)
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'jhm_readonly') THEN
    CREATE ROLE jhm_readonly;
  END IF;
END $$;

GRANT CONNECT ON DATABASE postgres TO jhm_readonly;
GRANT USAGE ON SCHEMA public TO jhm_readonly;
GRANT SELECT ON auditorias_diagnostico TO jhm_readonly;
GRANT SELECT ON model_registry TO jhm_readonly;


-- 2. ROL DE APLICACION (la API de FastAPI usa este rol - solo lo que necesita)
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'jhm_app') THEN
    CREATE ROLE jhm_app;
  END IF;
END $$;

GRANT CONNECT ON DATABASE postgres TO jhm_app;
GRANT USAGE ON SCHEMA public TO jhm_app;
GRANT SELECT, INSERT, UPDATE ON auditorias_diagnostico TO jhm_app;
GRANT SELECT, INSERT, UPDATE ON model_registry TO jhm_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO jhm_app;


-- 3. ROL DE ADMINISTRADOR ML (para cientificos de datos - puede borrar y actualizar modelos)
DO $$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'jhm_ml_admin') THEN
    CREATE ROLE jhm_ml_admin;
  END IF;
END $$;

GRANT CONNECT ON DATABASE postgres TO jhm_ml_admin;
GRANT USAGE ON SCHEMA public TO jhm_ml_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON auditorias_diagnostico TO jhm_ml_admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON model_registry TO jhm_ml_admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO jhm_ml_admin;


-- 4. POLITICAS DE SEGURIDAD A NIVEL DE FILA (Row Level Security)
-- Previene acceso a datos de otras organizaciones si se escala a multi-tenant
ALTER TABLE auditorias_diagnostico ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_registry ENABLE ROW LEVEL SECURITY;

-- Politica: jhm_app solo puede ver sus propios registros (preparado para multi-tenant)
CREATE POLICY IF NOT EXISTS "app_own_data" ON auditorias_diagnostico
  FOR ALL TO jhm_app USING (true);  -- por ahora permite todo, listo para restringir

CREATE POLICY IF NOT EXISTS "app_model_registry" ON model_registry
  FOR ALL TO jhm_app USING (true);

CREATE POLICY IF NOT EXISTS "readonly_audit" ON auditorias_diagnostico
  FOR SELECT TO jhm_readonly USING (true);

CREATE POLICY IF NOT EXISTS "readonly_models" ON model_registry
  FOR SELECT TO jhm_readonly USING (true);


-- 5. VERIFICACION - Ver roles y permisos configurados
SELECT grantee, table_name, privilege_type
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
ORDER BY grantee, table_name;
