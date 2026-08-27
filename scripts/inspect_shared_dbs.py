"""Read-only snapshot of the shared PG/Neo4j before MemGraphRAG writes anything."""

import asyncio
from dotenv import dotenv_values

LR = "/home/sanda/Desktop/project/lightrag/cf_lightrag/.env"
cfg = dotenv_values(LR)

PG = dict(
    host=cfg.get("POSTGRES_HOST", "").split(":")[0] or "192.168.6.2",
    port=int(
        (
            cfg.get("POSTGRES_PORT")
            or (
                cfg.get("POSTGRES_HOST", "").split(":")[1]
                if ":" in cfg.get("POSTGRES_HOST", "")
                else "5432"
            )
        )
    ),
    user=cfg.get("POSTGRES_USER"),
    password=cfg.get("POSTGRES_PASSWORD"),
    database=cfg.get("POSTGRES_DATABASE"),
)
NEO = dict(
    uri=cfg.get("NEO4J_URI"), user=cfg.get("NEO4J_USERNAME"), password=cfg.get("NEO4J_PASSWORD")
)


async def pg():
    import asyncpg

    conn = await asyncpg.connect(
        host=PG["host"],
        port=PG["port"],
        user=PG["user"],
        password=PG["password"],
        database=PG["database"],
    )
    try:
        rows = await conn.fetch("""
            SELECT tablename FROM pg_tables
            WHERE schemaname NOT IN ('pg_catalog','information_schema') ORDER BY tablename""")
        names = [r["tablename"] for r in rows]
        print(f"PG base={PG['database']} host={PG['host']}:{PG['port']}  tables={len(names)}")
        lr = [n for n in names if n.lower().startswith("lightrag")]
        mgr = [n for n in names if n.lower().startswith("mgr_")]
        print(f"  LIGHTRAG_* : {len(lr)} -> {lr}")
        print(f"  mgr_*      : {len(mgr)} -> {mgr or '(aucune)'}")
        for t in lr:
            c = await conn.fetchval(f'SELECT count(*) FROM "{t}"')
            print(f"    {t:32} {c:>8} lignes")
    finally:
        await conn.close()


def neo():
    from neo4j import GraphDatabase

    d = GraphDatabase.driver(NEO["uri"], auth=(NEO["user"], NEO["password"]))
    try:
        with d.session() as s:
            labels = [
                r["label"]
                for r in s.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            ]
            print(f"\nNeo4j {NEO['uri']}  labels={labels}")
            for lab in labels:
                n = s.run(f"MATCH (n:`{lab}`) RETURN count(n) AS c").single()["c"]
                e = s.run(
                    f"MATCH (:`{lab}`)-[r]-(:`{lab}`) RETURN count(DISTINCT r) AS c"
                ).single()["c"]
                print(f"    label {lab:20} noeuds={n:>7}  aretes={e:>7}")
            # An empty label is harmless: CREATE INDEX registers it in db.labels()
            # without any node. What matters is whether it already holds data.
            if "rfe_mgr" in labels:
                n = s.run("MATCH (n:`rfe_mgr`) RETURN count(n) AS c").single()["c"]
                print(f"    'rfe_mgr' : {n} noeuds -> {'DANGER' if n else 'vide (OK)'}")
            else:
                print("    'rfe_mgr' : absent (OK)")
    finally:
        d.close()


asyncio.run(pg())
neo()
