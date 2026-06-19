-- Startup & Skills Intelligence Agents — Postgres schema
-- Append-only everywhere that matters: snapshots and extractions are never overwritten.
-- Raw ATS payloads live in JSONB; all queryable fields are promoted to typed columns.

-- ---------------------------------------------------------------------------
-- companies
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS companies (
    slug            TEXT        PRIMARY KEY,           -- watchlist key, e.g. "anthropic"
    name            TEXT        NOT NULL,
    ats             TEXT        NOT NULL,              -- 'greenhouse' | 'lever' | 'ashby'
    ats_slug        TEXT        NOT NULL,              -- slug used on that ATS board (often == slug)
    board_url       TEXT,                              -- resolved careers board URL, cached
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- postings
-- ---------------------------------------------------------------------------
-- One row per unique job posting. Updated in-place when the ATS reports a
-- change (updated_at advances); the snapshot table records point-in-time counts.
CREATE TABLE IF NOT EXISTS postings (
    ats             TEXT        NOT NULL,              -- 'greenhouse' | 'lever' | 'ashby'
    id              TEXT        NOT NULL,              -- ATS-native posting ID
    company_slug    TEXT        NOT NULL REFERENCES companies(slug),
    title           TEXT        NOT NULL,
    url             TEXT,
    department      TEXT,
    team            TEXT,
    location        TEXT,
    remote          BOOLEAN,
    employment_type TEXT,                              -- full-time, part-time, contract, intern
    seniority       TEXT,                             -- senior, staff, principal, junior, ic, manager
    description_html TEXT,
    description_text TEXT,
    compensation_min        INTEGER,                  -- in currency_minor units (cents) or raw if no minor
    compensation_max        INTEGER,
    compensation_currency   TEXT,                     -- ISO 4217, e.g. 'USD'
    compensation_interval   TEXT,                     -- 'annual' | 'hourly' | 'monthly'
    posted_at       TIMESTAMPTZ,
    updated_at      TIMESTAMPTZ,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),-- updated each successful fetch
    raw             JSONB       NOT NULL,              -- verbatim ATS payload
    PRIMARY KEY (ats, id)
);

CREATE INDEX IF NOT EXISTS postings_company_slug_idx   ON postings (company_slug);
CREATE INDEX IF NOT EXISTS postings_updated_at_idx     ON postings (updated_at);
CREATE INDEX IF NOT EXISTS postings_department_idx     ON postings (department);
CREATE INDEX IF NOT EXISTS postings_seniority_idx      ON postings (seniority);
-- GIN index enables skill extraction queries over the raw JSONB
CREATE INDEX IF NOT EXISTS postings_raw_gin_idx        ON postings USING GIN (raw);

-- ---------------------------------------------------------------------------
-- snapshots
-- Append-only — never UPDATE or DELETE. Each scheduled run appends one row
-- per company. The time series is the product; history must be preserved.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS snapshots (
    id              BIGSERIAL   PRIMARY KEY,
    company_slug    TEXT        NOT NULL REFERENCES companies(slug),
    snapshot_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    posting_count   INTEGER     NOT NULL,
    eng_count       INTEGER,                          -- eng/product/data postings (hiring-velocity proxy)
    new_ids         TEXT[]      NOT NULL DEFAULT '{}', -- posting IDs added since last snapshot
    removed_ids     TEXT[]      NOT NULL DEFAULT '{}', -- posting IDs no longer live
    summary         JSONB       NOT NULL DEFAULT '{}'  -- arbitrary per-run metadata
);

CREATE INDEX IF NOT EXISTS snapshots_company_slug_at_idx ON snapshots (company_slug, snapshot_at DESC);

-- ---------------------------------------------------------------------------
-- extractions
-- LLM extraction output per posting. Append-only; one row per
-- (posting, model, run). Downstream aggregation always reads the latest
-- extraction per posting (keyed on extracted_at DESC).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS extractions (
    id              BIGSERIAL   PRIMARY KEY,
    ats             TEXT        NOT NULL,
    posting_id      TEXT        NOT NULL,
    -- denormalised FK: REFERENCES postings(ats, id)
    FOREIGN KEY (ats, posting_id) REFERENCES postings (ats, id),
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model           TEXT        NOT NULL,             -- e.g. 'claude-sonnet-4-6'
    skills          TEXT[]      NOT NULL DEFAULT '{}',
    platforms       TEXT[]      NOT NULL DEFAULT '{}',
    seniority_signal TEXT,                            -- normalised signal after taxonomy pass
    comp_min        INTEGER,
    comp_max        INTEGER,
    comp_currency   TEXT,
    comp_interval   TEXT,
    raw             JSONB       NOT NULL              -- full structured LLM output (Pydantic dump)
);

CREATE INDEX IF NOT EXISTS extractions_posting_idx  ON extractions (ats, posting_id, extracted_at DESC);
CREATE INDEX IF NOT EXISTS extractions_skills_gin   ON extractions USING GIN (skills);
CREATE INDEX IF NOT EXISTS extractions_platforms_gin ON extractions USING GIN (platforms);

-- ---------------------------------------------------------------------------
-- watermarks
-- One row per (company_slug, ats). Updated after each successful ingestion
-- run. Agents filter postings WHERE updated_at > watermark to get only deltas.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS watermarks (
    company_slug    TEXT        NOT NULL REFERENCES companies(slug),
    ats             TEXT        NOT NULL,
    last_fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (company_slug, ats)
);
