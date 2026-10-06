PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
  global_run_id TEXT PRIMARY KEY,
  dataset_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  participant_id TEXT NOT NULL,
  trial_id TEXT NOT NULL,
  training_year REAL,
  source_manifest_hash TEXT,
  created_at TEXT NOT NULL,
  UNIQUE(dataset_id, run_id)
);

CREATE TABLE IF NOT EXISTS source_files (
  source_file_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  source_type TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  size_bytes INTEGER,
  modified_at TEXT,
  read_only INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS generated_artifacts (
  artifact_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  artifact_type TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  generator_version TEXT NOT NULL,
  git_commit TEXT NOT NULL,
  config_hash TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS validation_references (
  reference_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  reference_type TEXT NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT,
  notes TEXT
);

CREATE TABLE IF NOT EXISTS evidence_sources (
  source_node_id TEXT PRIMARY KEY,
  source_family TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
  entity_id TEXT PRIMARY KEY,
  entity_type TEXT NOT NULL,
  label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS phases (
  phase_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  phase_name TEXT NOT NULL,
  legacy_phase_name TEXT,
  t_start_s REAL NOT NULL,
  t_end_s REAL NOT NULL,
  annotation_source_id TEXT REFERENCES source_files(source_file_id),
  CHECK(t_start_s >= 0 AND t_end_s >= t_start_s)
);

CREATE TABLE IF NOT EXISTS frames (
  frame_evidence_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  frame_idx INTEGER NOT NULL,
  t_video_s REAL NOT NULL,
  phase_id TEXT REFERENCES phases(phase_id),
  source_file_id TEXT NOT NULL REFERENCES source_files(source_file_id),
  source_row INTEGER,
  observation_json TEXT NOT NULL,
  UNIQUE(global_run_id, frame_idx)
);

CREATE TABLE IF NOT EXISTS strokes (
  stroke_evidence_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  stroke_id TEXT NOT NULL,
  t_start_s REAL NOT NULL,
  t_end_s REAL NOT NULL,
  source_file_id TEXT REFERENCES source_files(source_file_id),
  generated_artifact_id TEXT REFERENCES generated_artifacts(artifact_id),
  source_row INTEGER,
  metrics_json TEXT NOT NULL,
  CHECK(source_file_id IS NOT NULL OR generated_artifact_id IS NOT NULL),
  CHECK(t_start_s >= 0 AND t_end_s >= t_start_s)
);

CREATE TABLE IF NOT EXISTS motion_contexts (
  motion_context_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  event_id TEXT NOT NULL REFERENCES events(event_id),
  t_start_s REAL NOT NULL,
  t_end_s REAL NOT NULL,
  context_json TEXT NOT NULL,
  CHECK(t_start_s >= 0 AND t_end_s >= t_start_s)
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  event_type TEXT NOT NULL,
  t_start_s REAL NOT NULL,
  t_end_s REAL NOT NULL,
  phase_id TEXT REFERENCES phases(phase_id),
  anatomy TEXT,
  evidence_status TEXT NOT NULL,
  lifecycle_status TEXT NOT NULL,
  rule_version TEXT NOT NULL,
  attributes_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  CHECK(t_start_s >= 0 AND t_end_s >= t_start_s),
  CHECK(evidence_status IN ('candidate', 'simulator_supported', 'human_confirmed', 'human_rejected')),
  CHECK(lifecycle_status IN ('open', 'updated', 'closed'))
);

CREATE TABLE IF NOT EXISTS event_evidence (
  event_id TEXT NOT NULL REFERENCES events(event_id),
  evidence_id TEXT NOT NULL,
  evidence_type TEXT NOT NULL,
  PRIMARY KEY(event_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS relations (
  relation_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  source_id TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  construction_method TEXT NOT NULL,
  provisional INTEGER NOT NULL DEFAULT 0,
  rule_version TEXT NOT NULL,
  attributes_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS review_clips (
  clip_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  event_id TEXT REFERENCES events(event_id),
  t_start_s REAL NOT NULL,
  t_end_s REAL NOT NULL,
  media_path TEXT,
  media_hash TEXT,
  status TEXT NOT NULL,
  CHECK(t_start_s >= 0 AND t_end_s >= t_start_s)
);

CREATE TABLE IF NOT EXISTS knowledge_rules (
  rule_id TEXT PRIMARY KEY,
  rule_status TEXT NOT NULL,
  rule_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  CHECK(rule_status IN ('draft', 'expert_validated', 'retired'))
);

CREATE TABLE IF NOT EXISTS feedback_claims (
  claim_id TEXT PRIMARY KEY,
  global_run_id TEXT NOT NULL REFERENCES runs(global_run_id),
  claim_type TEXT NOT NULL,
  claim_text TEXT NOT NULL,
  certainty TEXT NOT NULL,
  feedback_level TEXT,
  knowledge_rule_id TEXT REFERENCES knowledge_rules(rule_id),
  validation_status TEXT NOT NULL,
  validation_messages_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claim_evidence (
  claim_id TEXT NOT NULL REFERENCES feedback_claims(claim_id),
  evidence_id TEXT NOT NULL,
  PRIMARY KEY(claim_id, evidence_id)
);

CREATE TABLE IF NOT EXISTS human_reviews (
  review_id TEXT PRIMARY KEY,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  decision TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  comment TEXT,
  created_at TEXT NOT NULL,
  CHECK(decision IN ('confirmed', 'rejected', 'uncertain', 'accept', 'revise'))
);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
  tool_call_id TEXT PRIMARY KEY,
  global_run_id TEXT REFERENCES runs(global_run_id),
  session_id TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  arguments_json TEXT NOT NULL,
  result_evidence_ids_json TEXT NOT NULL,
  called_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_frames_run_time ON frames(global_run_id, t_video_s);
CREATE INDEX IF NOT EXISTS idx_events_run_time ON events(global_run_id, t_start_s, t_end_s);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id, relation_type);
CREATE INDEX IF NOT EXISTS idx_event_evidence_event ON event_evidence(event_id, evidence_type);
CREATE INDEX IF NOT EXISTS idx_motion_context_event ON motion_contexts(event_id);

