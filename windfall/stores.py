"""The Aiven data layer. Every store degrades gracefully, so the product keeps working if one is missing,
and /api/health says exactly which ones are live.

  PostgreSQL + pgvector  source of truth (site, assets, learned models) and the agents' memory of past days
  ClickHouse             time series: market history, telemetry, backtest results
  Valkey                 live state of the site and the event bus to the dashboard (pub/sub)
  Kafka                  the telemetry and decision streams
  OpenSearch             searchable audit log of every agent decision (CSRD evidence)
"""
from __future__ import annotations

import json
import os
import ssl
import tempfile
import threading
import time
from urllib.parse import urlparse

STATUS: dict[str, dict] = {}


def _env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return None


def _ok(name: str, detail: str = "") -> None:
    STATUS[name] = {"live": True, "detail": detail, "at": time.time()}


def _fail(name: str, err: Exception | str) -> None:
    STATUS[name] = {"live": False, "detail": str(err)[:300], "at": time.time()}
    print(f"[stores] {name} unavailable: {err}")


# ------------------------------------------------------------------ PostgreSQL


class PG:
    def __init__(self) -> None:
        self.conn = None
        self.vector = False
        url = _env("DATABASE_URL", "PG_URI", "POSTGRES_URL", "PG_SERVICE_URI")
        if not url:
            _fail("postgres", "no DATABASE_URL")
            return
        try:
            import psycopg
            self.conn = psycopg.connect(url, autocommit=True, connect_timeout=10)
            self.vector = True
            try:
                self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception as e:  # noqa: BLE001
                self.vector = False
                print(f"[pg] pgvector unavailable: {e}")
            self.conn.execute("""
              CREATE TABLE IF NOT EXISTS assets (
                id SERIAL PRIMARY KEY, site TEXT, kind TEXT, name TEXT UNIQUE, rated_kw REAL,
                tau_hat REAL, tau_true REAL, updated_at TIMESTAMPTZ DEFAULT now());
              CREATE TABLE IF NOT EXISTS model_history (
                id SERIAL PRIMARY KEY, asset TEXT, day DATE, tau_hat REAL, error_pct REAL);
              CREATE TABLE IF NOT EXISTS decisions (
                id SERIAL PRIMARY KEY, at TIMESTAMPTZ DEFAULT now(), agent TEXT, kind TEXT,
                summary TEXT, payload JSONB);
            """)
            if self.vector:
                self.conn.execute("""
                  CREATE TABLE IF NOT EXISTS day_memory (
                    day DATE PRIMARY KEY, embedding vector(24), avg_price REAL, avg_co2 REAL,
                    prices JSONB, co2 JSONB, saved_eur REAL, note TEXT)""")
            else:
                self.conn.execute("""
                  CREATE TABLE IF NOT EXISTS day_memory (
                    day DATE PRIMARY KEY, embedding JSONB, avg_price REAL, avg_co2 REAL,
                    prices JSONB, co2 JSONB, saved_eur REAL, note TEXT)""")
            _ok("postgres", "pgvector on" if self.vector else "pgvector off (JSON fallback)")
        except Exception as e:  # noqa: BLE001
            self.conn = None
            _fail("postgres", e)

    @property
    def live(self) -> bool:
        return self.conn is not None

    def q(self, sql: str, params: tuple | list | None = None) -> list[tuple]:
        if not self.conn:
            return []
        try:
            cur = self.conn.execute(sql, params)
            return cur.fetchall() if cur.description else []
        except Exception as e:  # noqa: BLE001
            print(f"[pg] {e}")
            return []

    def upsert_asset(self, site: str, kind: str, name: str, kw: float, tau_hat: float | None, tau_true: float | None) -> None:
        self.q("""INSERT INTO assets(site,kind,name,rated_kw,tau_hat,tau_true) VALUES (%s,%s,%s,%s,%s,%s)
                  ON CONFLICT (name) DO UPDATE SET tau_hat=EXCLUDED.tau_hat, updated_at=now()""",
               (site, kind, name, kw, tau_hat, tau_true))

    def remember_day(self, day, vec: list[float], prices: list[float], co2: list[float], saved: float) -> None:
        emb = "[" + ",".join(f"{v:.4f}" for v in vec) + "]" if self.vector else json.dumps(vec)
        self.q("""INSERT INTO day_memory(day, embedding, avg_price, avg_co2, prices, co2, saved_eur)
                  VALUES (%s, %s, %s, %s, %s, %s, %s)
                  ON CONFLICT (day) DO UPDATE SET embedding=EXCLUDED.embedding, prices=EXCLUDED.prices,
                  co2=EXCLUDED.co2, saved_eur=EXCLUDED.saved_eur""",
               (day, emb, sum(prices) / len(prices), sum(co2) / len(co2), json.dumps(prices), json.dumps(co2), saved))

    def similar_days(self, vec: list[float], before, k: int = 5) -> list[dict]:
        if not self.conn or not self.vector:
            return []
        emb = "[" + ",".join(f"{v:.4f}" for v in vec) + "]"
        rows = self.q("""SELECT day, embedding <-> %s::vector AS dist, prices, co2, avg_price, avg_co2, saved_eur
                         FROM day_memory WHERE day < %s ORDER BY embedding <-> %s::vector LIMIT %s""",
                      (emb, before, emb, k))
        return [{"day": r[0].isoformat(), "distance": round(float(r[1]), 3), "prices": r[2], "co2": r[3],
                 "avg_price": r[4], "avg_co2": r[5], "saved_eur": r[6]} for r in rows]

    def log_decision(self, agent: str, kind: str, summary: str, payload: dict) -> None:
        self.q("INSERT INTO decisions(agent, kind, summary, payload) VALUES (%s,%s,%s,%s)",
               (agent, kind, summary, json.dumps(payload, default=str)))


# ------------------------------------------------------------------ ClickHouse


class CH:
    def __init__(self) -> None:
        self.client = None
        url = _env("CLICKHOUSE_URL", "CLICKHOUSE_URI", "CLICKHOUSE_HTTPS_URI", "CH_URI")
        if not url:
            host = _env("CLICKHOUSE_HOST")
            if host:
                url = f"https://{_env('CLICKHOUSE_USER') or 'avnadmin'}:{_env('CLICKHOUSE_PASSWORD') or ''}@{host}:{_env('CLICKHOUSE_PORT') or '443'}"
        if not url:
            _fail("clickhouse", "no CLICKHOUSE_URL")
            return
        try:
            import clickhouse_connect
            u = urlparse(url)
            self.client = clickhouse_connect.get_client(
                host=u.hostname, port=u.port or 443, username=u.username or "avnadmin", password=u.password or "",
                secure=u.scheme in ("https", "clickhouses"), connect_timeout=15)
            self.client.command("""CREATE TABLE IF NOT EXISTS market_hourly (
                ts DateTime('Europe/Paris'), price Float32, co2 Float32, temp Float32, source LowCardinality(String))
                ENGINE = ReplacingMergeTree ORDER BY ts""")
            self.client.command("""CREATE TABLE IF NOT EXISTS telemetry (
                ts DateTime64(3), sim_ts DateTime('Europe/Paris'), asset LowCardinality(String), metric LowCardinality(String), value Float64)
                ENGINE = MergeTree ORDER BY (asset, metric, ts)""")
            self.client.command("""CREATE TABLE IF NOT EXISTS backtest_daily (
                run_id String, day Date, baseline_cost Float64, windfall_cost Float64, baseline_co2 Float64, windfall_co2 Float64,
                baseline_peak Float64, windfall_peak Float64, safety_overrides UInt16, tau_error_pct Float32)
                ENGINE = MergeTree ORDER BY (run_id, day)""")
            _ok("clickhouse")
        except Exception as e:  # noqa: BLE001
            self.client = None
            _fail("clickhouse", e)

    @property
    def live(self) -> bool:
        return self.client is not None

    def insert(self, table: str, rows: list[list], cols: list[str]) -> None:
        if not self.client or not rows:
            return
        try:
            self.client.insert(table, rows, column_names=cols)
        except Exception as e:  # noqa: BLE001
            print(f"[ch] insert {table}: {e}")

    def query(self, sql: str, params: dict | None = None) -> list[tuple]:
        if not self.client:
            return []
        try:
            return list(self.client.query(sql, parameters=params or {}).result_rows)
        except Exception as e:  # noqa: BLE001
            print(f"[ch] {e}")
            return []


# ------------------------------------------------------------------ Valkey


class VK:
    def __init__(self) -> None:
        self.r = None
        url = _env("VALKEY_URL", "VALKEY_URI", "REDIS_URL", "VALKEY_SERVICE_URI")
        if not url:
            _fail("valkey", "no VALKEY_URL")
            return
        try:
            import redis
            self.r = redis.Redis.from_url(url, decode_responses=True, socket_timeout=10, ssl_cert_reqs=None) if url.startswith("rediss") \
                else redis.Redis.from_url(url, decode_responses=True, socket_timeout=10)
            self.r.ping()
            _ok("valkey")
        except Exception as e:  # noqa: BLE001
            self.r = None
            _fail("valkey", e)

    @property
    def live(self) -> bool:
        return self.r is not None

    def set_state(self, key: str, mapping: dict) -> None:
        if self.r:
            try:
                self.r.hset(f"windfall:{key}", mapping={k: json.dumps(v) for k, v in mapping.items()})
            except Exception as e:  # noqa: BLE001
                print(f"[valkey] {e}")

    def get_state(self, key: str) -> dict:
        if not self.r:
            return {}
        try:
            return {k: json.loads(v) for k, v in self.r.hgetall(f"windfall:{key}").items()}
        except Exception:  # noqa: BLE001
            return {}

    def incr(self, key: str, by: int = 1) -> int:
        if not self.r:
            return 0
        try:
            return int(self.r.incrby(f"windfall:count:{key}", by))
        except Exception:  # noqa: BLE001
            return 0

    def publish(self, event: dict) -> None:
        if self.r:
            try:
                self.r.publish("windfall:events", json.dumps(event, default=str))
            except Exception as e:  # noqa: BLE001
                print(f"[valkey] publish {e}")


# ------------------------------------------------------------------ Kafka


class KF:
    TOPICS = ("site.telemetry", "agent.decisions")

    def __init__(self) -> None:
        self.producer = None
        self.cfg: dict = {}
        boot = _env("KAFKA_BOOTSTRAP_SERVER", "KAFKA_BOOTSTRAP_SERVERS", "KAFKA_SERVICE_URI", "KAFKA_URI")
        if not boot:
            _fail("kafka", "no KAFKA_BOOTSTRAP_SERVER")
            return
        try:
            from kafka import KafkaProducer
            from kafka.admin import KafkaAdminClient, NewTopic
            cfg: dict = {"bootstrap_servers": boot.replace("kafka://", "").split(",")}
            ca, cert, key = _env("KAFKA_CA_CERT"), _env("KAFKA_ACCESS_CERT"), _env("KAFKA_ACCESS_KEY")
            user, pwd = _env("KAFKA_USERNAME", "KAFKA_USER"), _env("KAFKA_PASSWORD")
            tmp = tempfile.mkdtemp()
            def wr(name: str, content: str) -> str:
                p = os.path.join(tmp, name)
                with open(p, "w") as f:
                    f.write(content.replace("\\n", "\n"))
                return p
            if user and pwd:
                cfg.update(security_protocol="SASL_SSL", sasl_mechanism="SCRAM-SHA-256", sasl_plain_username=user, sasl_plain_password=pwd)
                if ca:
                    cfg["ssl_cafile"] = wr("ca.pem", ca)
            elif cert and key:
                cfg.update(security_protocol="SSL", ssl_cafile=wr("ca.pem", ca) if ca else None,
                           ssl_certfile=wr("cert.pem", cert), ssl_keyfile=wr("key.pem", key))
            self.cfg = cfg
            try:
                admin = KafkaAdminClient(**cfg, request_timeout_ms=15000)
                existing = set(admin.list_topics())
                new = [NewTopic(t, num_partitions=2, replication_factor=-1) for t in self.TOPICS if t not in existing]
                if new:
                    try:
                        admin.create_topics(new)
                    except Exception:  # noqa: BLE001
                        admin.create_topics([NewTopic(t.name, num_partitions=2, replication_factor=2) for t in new])
                admin.close()
            except Exception as e:  # noqa: BLE001
                print(f"[kafka] topic setup: {e} (create topics site.telemetry and agent.decisions in the console)")
            self.producer = KafkaProducer(**cfg, value_serializer=lambda v: json.dumps(v, default=str).encode(), linger_ms=50)
            _ok("kafka", ",".join(self.TOPICS))
        except Exception as e:  # noqa: BLE001
            self.producer = None
            _fail("kafka", e)

    @property
    def live(self) -> bool:
        return self.producer is not None

    def send(self, topic: str, value: dict) -> None:
        if self.producer:
            try:
                self.producer.send(topic, value)
            except Exception as e:  # noqa: BLE001
                print(f"[kafka] send {e}")

    def consume(self, topic: str, handler, group: str = "windfall-watcher") -> None:
        """Run a consumer in a background thread; handler(dict) per message."""
        if not self.cfg:
            return
        def loop() -> None:
            from kafka import KafkaConsumer
            while True:
                try:
                    c = KafkaConsumer(topic, **self.cfg, group_id=group, auto_offset_reset="latest",
                                      value_deserializer=lambda b: json.loads(b.decode()))
                    for msg in c:
                        try:
                            handler(msg.value)
                        except Exception as e:  # noqa: BLE001
                            print(f"[kafka] handler {e}")
                except Exception as e:  # noqa: BLE001
                    print(f"[kafka] consumer {e}; retrying")
                    time.sleep(5)
        threading.Thread(target=loop, daemon=True).start()


# ------------------------------------------------------------------ OpenSearch


class OS:
    INDEX = "windfall-decisions"

    def __init__(self) -> None:
        self.c = None
        url = _env("OPENSEARCH_URL", "OPENSEARCH_URI", "OPENSEARCH_SERVICE_URI")
        if not url:
            _fail("opensearch", "no OPENSEARCH_URL")
            return
        try:
            from opensearchpy import OpenSearch
            u = urlparse(url)
            self.c = OpenSearch(hosts=[{"host": u.hostname, "port": u.port or 443}], http_auth=(u.username, u.password) if u.username else None,
                                use_ssl=u.scheme == "https", verify_certs=True, ssl_show_warn=False, timeout=15)
            if not self.c.indices.exists(index=self.INDEX):
                self.c.indices.create(index=self.INDEX, body={"mappings": {"properties": {
                    "at": {"type": "date"}, "agent": {"type": "keyword"}, "kind": {"type": "keyword"},
                    "asset": {"type": "keyword"}, "summary": {"type": "text"}, "explanation": {"type": "text"},
                    "eur": {"type": "float"}, "co2_kg": {"type": "float"}}}})
            _ok("opensearch")
        except Exception as e:  # noqa: BLE001
            self.c = None
            _fail("opensearch", e)

    @property
    def live(self) -> bool:
        return self.c is not None

    def index(self, doc: dict) -> None:
        if self.c:
            try:
                self.c.index(index=self.INDEX, body=doc)
            except Exception as e:  # noqa: BLE001
                print(f"[os] {e}")

    def search(self, text: str, size: int = 20) -> list[dict]:
        if not self.c:
            return []
        try:
            body = {"size": size, "sort": [{"at": "desc"}],
                    "query": {"multi_match": {"query": text, "fields": ["summary", "explanation", "asset", "agent"]}} if text else {"match_all": {}}}
            return [h["_source"] for h in self.c.search(index=self.INDEX, body=body)["hits"]["hits"]]
        except Exception as e:  # noqa: BLE001
            print(f"[os] {e}")
            return []

    def count(self) -> int:
        if not self.c:
            return 0
        try:
            return int(self.c.count(index=self.INDEX)["count"])
        except Exception:  # noqa: BLE001
            return 0
