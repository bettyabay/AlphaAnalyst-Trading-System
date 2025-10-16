-- AlphaAnalyst Trading System v2.1 — Phase 1 Supabase Schema
-- How to apply:
-- 1) In Supabase SQL Editor, paste and run this entire script; OR
-- 2) psql against your Postgres: psql "$DATABASE_URL" -f scripts/supabase_schema_phase1.sql
-- Note: uses pgcrypto for gen_random_uuid(). Enable if not present.

begin;

create extension if not exists pgcrypto;

-- =============
-- Core Tables
-- =============

-- Users: Stores user account data and access levels
create table if not exists public.users (
  id uuid primary key default gen_random_uuid(),
  email text not null unique,
  role text not null default 'user',
  created_at timestamptz not null default now()
);
create index if not exists idx_users_email on public.users (email);

-- Market data: Historical and live OHLCV
create table if not exists public.market_data (
  id bigserial primary key,
  symbol text not null,
  date date not null,
  open numeric,
  high numeric,
  low numeric,
  close numeric,
  volume bigint,
  source text, -- polygon/yfinance/demo
  created_at timestamptz not null default now()
);
create unique index if not exists ux_market_data_symbol_date on public.market_data (symbol, date);
create index if not exists idx_market_data_symbol_date_desc on public.market_data (symbol, date desc);

-- Trade signals: AI-generated recommendations
create table if not exists public.trade_signals (
  id uuid primary key default gen_random_uuid(),
  symbol text not null,
  signal_type text not null, -- buy/sell/hold
  confidence numeric,
  details jsonb, -- optional metadata
  timestamp timestamptz not null default now()
);
create index if not exists idx_trade_signals_symbol_time on public.trade_signals (symbol, timestamp desc);

-- Portfolio: Tracks active holdings and performance
create table if not exists public.portfolio (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.users(id) on delete cascade,
  symbol text not null,
  quantity numeric not null,
  avg_price numeric not null,
  pnl numeric default 0,
  updated_at timestamptz not null default now()
);
create index if not exists idx_portfolio_user_symbol on public.portfolio (user_id, symbol);

-- System logs: Log trading decisions and AI events
create table if not exists public.system_logs (
  id bigserial primary key,
  event text not null,
  details jsonb,
  timestamp timestamptz not null default now()
);
create index if not exists idx_system_logs_time on public.system_logs (timestamp desc);

-- =============
-- Phase 1 helper tables (from app)
-- =============
-- These align with phase1 app functionality. Keep here for convenience; in prod, manage via migrations.

create table if not exists public.instrument_master_data (
  symbol text primary key,
  data_fetch_status jsonb,
  last_updated timestamptz,
  health_score numeric
);

create table if not exists public.research_documents (
  id uuid primary key default gen_random_uuid(),
  symbol text,
  file_name text,
  file_content text,
  embedding_vector jsonb,
  uploaded_at timestamptz not null default now()
);
create index if not exists idx_research_documents_symbol_time on public.research_documents (symbol, uploaded_at desc);

commit;

-- Grant public read if desired (adjust to your security model)
-- grant select on table public.market_data to anon, authenticated;
-- grant select on table public.trade_signals to anon, authenticated;
-- grant select on table public.research_documents to anon, authenticated;
-- For writes, prefer RLS with policies tailored to your needs.


