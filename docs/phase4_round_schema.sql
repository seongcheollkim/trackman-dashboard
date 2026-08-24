-- DODOS Golf Solution / Phase 4 Round Record
-- Run once in the Supabase SQL Editor before saving round records.

create table if not exists public.dodos_rounds (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.dodos_users(id) on delete cascade,
    round_date date not null,
    course_name text not null,
    tee_name text,
    course_par integer,
    scoring_mode text,
    playing_hcp integer,
    total_score integer,
    total_putts integer,
    fir_pct numeric(5,1),
    gir_pct numeric(5,1),
    penalties integer not null default 0,
    source text not null default 'manual',
    source_round_id text,
    source_url text,
    best_shot text not null default '',
    weakness text not null default '',
    next_goal text not null default '',
    notes text not null default '',
    duration_minutes integer,
    distance_km numeric(8,3),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create unique index if not exists dodos_rounds_source_unique
    on public.dodos_rounds(user_id, source, source_round_id)
    where source_round_id is not null;

create index if not exists dodos_rounds_user_date_idx
    on public.dodos_rounds(user_id, round_date desc);

create table if not exists public.dodos_round_holes (
    id uuid primary key default gen_random_uuid(),
    round_id uuid not null references public.dodos_rounds(id) on delete cascade,
    user_id uuid not null references public.dodos_users(id) on delete cascade,
    hole_number integer not null,
    par integer,
    stroke_index integer,
    distance_m numeric(8,2),
    score integer,
    putts integer,
    fir text,
    gir boolean,
    sand_shots integer not null default 0,
    penalties integer not null default 0,
    extra_strokes integer,
    net_score integer,
    stableford_points integer,
    created_at timestamptz not null default now(),
    unique(round_id, hole_number)
);

create index if not exists dodos_round_holes_round_idx
    on public.dodos_round_holes(round_id, hole_number);

-- Existing projects use the service-role key from the Streamlit app, so no
-- application-side RLS policy is required for the current architecture.
-- If client-side Supabase access is added later, enable RLS and add policies
-- scoped to auth.uid() / user_id at that time.
