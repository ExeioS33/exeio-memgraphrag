"""🎮 Playful Streamlit UI for MemGraphRAG.

Run:
    uv run --extra client streamlit run memgraphrag/client/app.py
"""

from __future__ import annotations

import json
import os
from functools import reduce
from operator import mul
from typing import Any

import pandas as pd
import streamlit as st

from memgraphrag.client.http import MemGraphRAGClient
from memgraphrag.client.optimize import run_optimize
from memgraphrag.client.params import (
    PRESETS,
    QUERY_PARAMS,
    SUPPORTED_EXTENSIONS,
    clean_params,
    default_sweep_grid,
    defaults,
)

st.set_page_config(
    page_title="MemGraphRAG Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_EMOJI = {
    "pending": "⏳",
    "parsing": "🔍",
    "processing": "⚙️",
    "processed": "✅",
    "failed": "❌",
}


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    if "query_params" not in st.session_state:
        st.session_state.query_params = defaults()
    if "server_url" not in st.session_state:
        st.session_state.server_url = os.environ.get(
            "MEMGRAPHRAG_SERVER_URL", "http://localhost:9621"
        )
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("MEMGRAPHRAG_API_KEY", "")
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "opt_report" not in st.session_state:
        st.session_state.opt_report = None
    if "connected_once" not in st.session_state:
        st.session_state.connected_once = False


def get_client() -> MemGraphRAGClient:
    return MemGraphRAGClient(
        base_url=st.session_state.server_url or None,
        api_key=st.session_state.api_key or None,
    )


def apply_preset(name: str) -> None:
    st.session_state.query_params = {
        **st.session_state.query_params,
        **PRESETS[name],
    }


# --------------------------------------------------------------------------- #
# Sidebar connection + param knobs
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧠 MemGraphRAG")
        st.caption("Playful client for the Docker API container")
        st.session_state.server_url = st.text_input(
            "🌐 Server URL", value=st.session_state.server_url
        )
        st.session_state.api_key = st.text_input(
            "🔑 API key", value=st.session_state.api_key, type="password"
        )

        st.markdown("---")
        st.markdown("### ✨ Presets")
        cols = st.columns(2)
        for i, name in enumerate(PRESETS):
            if cols[i % 2].button(name, use_container_width=True):
                apply_preset(name)
                st.toast(f"Applied {name}", icon="✨")

        st.markdown("### 🎛️ Query params")
        params = dict(st.session_state.query_params)
        for spec in QUERY_PARAMS:
            key = spec.name
            label = f"{spec.emoji} {spec.name}"
            if spec.kind == "choice":
                choices = list(spec.choices or ())
                current = params.get(key, spec.default)
                idx = choices.index(current) if current in choices else 0
                params[key] = st.selectbox(label, choices, index=idx, help=spec.help)
            elif spec.kind == "int":
                params[key] = st.slider(
                    label,
                    min_value=int(spec.min or 1),
                    max_value=int(spec.max or 50),
                    value=int(params.get(key, spec.default)),
                    step=int(spec.step or 1),
                    help=spec.help,
                )
            elif spec.kind == "float":
                params[key] = st.slider(
                    label,
                    min_value=float(spec.min or 0.0),
                    max_value=float(spec.max or 1.0),
                    value=float(params.get(key, spec.default)),
                    step=float(spec.step or 0.01),
                    help=spec.help,
                )
            elif spec.kind == "bool":
                params[key] = st.toggle(
                    label,
                    value=bool(params.get(key, spec.default)),
                    help=spec.help,
                )
            elif spec.kind == "str":
                params[key] = st.text_area(
                    label,
                    value=params.get(key) or "",
                    help=spec.help,
                    height=80,
                )
        st.session_state.query_params = clean_params(params)

        st.markdown("---")
        st.caption("Tip: apply optimizer winners with ✨ Apply best params")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_home() -> None:
    st.markdown("## 🏠 Home base")
    st.write(
        "Welcome to the **MemGraphRAG Playground**! Connect to your container, "
        "ingest docs, tweak knobs, and watch Personalized PageRank do its magic. 🪄"
    )
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🔌 Connect / Refresh health", type="primary"):
            try:
                with get_client() as c:
                    health = c.health()
                st.session_state.last_health = health
                if not st.session_state.connected_once:
                    st.balloons()
                    st.session_state.connected_once = True
                st.success("Connected! 🎉")
            except Exception as exc:  # noqa: BLE001
                st.error(f"💥 Cannot reach server: {exc}")

    health = st.session_state.get("last_health")
    if not health:
        st.info("👆 Hit **Connect** to ping `/health`.")
        return

    busy = bool(health.get("pipeline_busy"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("❤️ Status", health.get("status", "?"))
    m2.metric("🧩 Core", str(health.get("core_version", "?")))
    m3.metric("🚀 API", str(health.get("api_version", "?")))
    m4.metric("🏭 Pipeline", "🟡 BUSY" if busy else "🟢 IDLE")

    st.json(health)


def tab_query() -> None:
    st.markdown("## 💬 Ask away")
    question = st.text_area(
        "Your question",
        placeholder="What is MemGraphRAG and how does PPR retrieval work?",
        height=100,
    )
    c1, c2, c3 = st.columns(3)
    data_only = c1.toggle("📚 Context only (`/query/data`)", value=False)
    use_stream = c2.toggle("📡 Stream (`/query/stream`)", value=False)
    show_raw = c3.toggle("🧾 Raw JSON", value=False)

    if st.button("🚀 Launch query", type="primary", disabled=not question.strip()):
        params = dict(st.session_state.query_params)
        try:
            with get_client() as c, st.spinner("🧠 Thinking really hard…"):
                if use_stream and not data_only:
                    chunks: list[str] = []
                    placeholder = st.empty()
                    for payload in c.query_stream(question, **params):
                        if payload == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload)
                            text = obj.get("response") or obj.get("error") or payload
                        except json.JSONDecodeError:
                            text = payload
                        chunks.append(str(text))
                        placeholder.markdown("".join(chunks))
                    st.session_state.last_answer = {
                        "answer": "".join(chunks),
                        "docs": [],
                        "doc_scores": [],
                    }
                elif data_only:
                    payload = c.query_data(question, **params)
                    st.session_state.last_answer = payload.get("data") or payload
                else:
                    st.session_state.last_answer = c.query(question, **params)
            st.toast("Answer ready!", icon="✨")
        except Exception as exc:  # noqa: BLE001
            st.error(f"💥 Query failed: {exc}")

    ans = st.session_state.last_answer
    if not ans:
        return

    if show_raw:
        st.json(ans)
        return

    answer = ans.get("answer") or ans.get("response")
    if answer:
        st.markdown("### ✨ Answer")
        st.markdown(answer)

    docs = ans.get("docs") or []
    scores = ans.get("doc_scores") or []
    if docs:
        st.markdown("### 📎 Retrieved evidence")
        for i, doc in enumerate(docs):
            score = scores[i] if i < len(scores) else "?"
            with st.expander(f"#{i + 1}  ·  score={score}", expanded=i == 0):
                st.write(doc)


def _status_rows(statuses: Any) -> pd.DataFrame:
    rows = []
    if isinstance(statuses, dict):
        for doc_id, meta in statuses.items():
            if isinstance(meta, dict):
                st_name = str(meta.get("status") or meta.get("doc_status") or "?")
                path = str(meta.get("file_path") or "")
            else:
                st_name, path = str(meta), ""
            rows.append(
                {
                    "emoji": STATUS_EMOJI.get(st_name.lower(), "📄"),
                    "doc_id": str(doc_id),
                    "status": st_name,
                    "file": path,
                }
            )
    return pd.DataFrame(rows)


def tab_ingest() -> None:
    st.markdown("## 📥 Feed the graph")
    st.write(
        "Upload files, point at a local directory, drop a URL, paste text, "
        "or ask the server to rescan its inbox. 📬"
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### 📤 Upload files")
        uploads = st.file_uploader(
            "Drop one or more docs",
            accept_multiple_files=True,
            type=[e.lstrip(".") for e in sorted(SUPPORTED_EXTENSIONS)],
        )
        if st.button("🚀 Upload selected", disabled=not uploads):
            try:
                with get_client() as c, st.spinner("📤 Shipping files…"):
                    for uf in uploads:
                        c.upload_bytes(uf.getvalue(), uf.name)
                st.success(f"✅ Queued {len(uploads)} file(s) — indexing in background")
                st.balloons()
            except Exception as exc:  # noqa: BLE001
                st.error(f"💥 Upload failed: {exc}")

        st.markdown("### 📁 Local directory")
        dir_path = st.text_input("Directory path on this machine", placeholder="/data/docs")
        recursive = st.checkbox("Recursive", value=True)
        if st.button("📁 Upload directory", disabled=not dir_path.strip()):
            try:
                with get_client() as c, st.spinner("📁 Walking directory…"):
                    results = c.upload_directory(dir_path, recursive=recursive)
                st.success(f"✅ Uploaded {len(results)} file(s)")
            except Exception as exc:  # noqa: BLE001
                st.error(f"💥 {exc}")

    with right:
        st.markdown("### 🌐 From URL")
        url = st.text_input("Document URL", placeholder="https://arxiv.org/pdf/2606.00610")
        url_name = st.text_input("Optional filename", placeholder="paper.pdf")
        if st.button("🌐 Download & ingest", disabled=not url.strip()):
            try:
                with get_client() as c, st.spinner("🌐 Fetching…"):
                    data = c.upload_url(url, filename=url_name or None)
                st.success(f"✅ Queued {data.get('filename')} → `{data.get('doc_id')}`")
            except Exception as exc:  # noqa: BLE001
                st.error(f"💥 {exc}")

        st.markdown("### 📝 Inline text")
        text = st.text_area("Paste text to index", height=120)
        if st.button("📝 Index text", disabled=not text.strip()):
            try:
                with get_client() as c, st.spinner("📝 Indexing…"):
                    data = c.insert_text(text)
                st.success(f"✅ Indexed `{data.get('doc_id')}`")
            except Exception as exc:  # noqa: BLE001
                st.error(f"💥 {exc}")

        st.markdown("### 🔎 Server inbox")
        if st.button("🔎 Rescan server input dir"):
            try:
                with get_client() as c:
                    data = c.scan_input_dir()
                st.info(
                    f"Scan queued — found **{data.get('files_found')}** file(s) in "
                    f"`{data.get('input_dir')}`"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"💥 {exc}")

    st.markdown("---")
    st.markdown("### 📋 Document status")
    refresh = st.button("🔄 Refresh status")
    if refresh or "doc_statuses" not in st.session_state:
        try:
            with get_client() as c:
                st.session_state.doc_statuses = c.list_documents()
                st.session_state.doc_statuses_error = None
        except Exception as exc:  # noqa: BLE001
            st.session_state.doc_statuses_error = str(exc)

    err = st.session_state.get("doc_statuses_error")
    if err:
        st.warning(f"Could not list documents: {err}")
        return

    data = st.session_state.get("doc_statuses") or {}
    df = _status_rows(data.get("statuses"))
    if df.empty:
        st.info("No documents yet — feed me! 🍽️")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
        busy_like = df["status"].str.lower().isin(["pending", "parsing", "processing"])
        if busy_like.any():
            st.caption("⏳ Indexing in progress — hit Refresh to update.")


def tab_optimize() -> None:
    st.markdown("## 🧪 Param lab")
    st.write(
        "Phase 1 scores every combo with **retrieval metrics** on `/query/data`. "
        "Phase 2 LLM-judges the top-N full answers (hybrid). 🔬"
    )

    question = st.text_area(
        "Evaluation question",
        value="What is the three-layer memory in MemGraphRAG?",
        height=80,
    )
    extra = st.text_input(
        "Extra questions (comma-separated, optional)",
        placeholder="How does PPR work?, What is conflict resolution?",
    )
    top_n = st.slider("🏆 Top-N for LLM judge", 1, 10, 3)
    use_judge = st.toggle("🧑‍⚖️ Run LLM judge (phase 2)", value=True)

    st.markdown("### 🎛️ Sweep grid")
    base_grid = default_sweep_grid()
    grid: dict[str, list[Any]] = {}
    for spec in QUERY_PARAMS:
        if not spec.sweepable or not spec.grid:
            continue
        default_vals = list(base_grid.get(spec.name, list(spec.grid)))
        if spec.kind == "choice":
            grid[spec.name] = st.multiselect(
                f"{spec.emoji} {spec.name}",
                options=list(spec.choices or spec.grid),
                default=default_vals,
            )
        elif spec.kind == "bool":
            grid[spec.name] = st.multiselect(
                f"{spec.emoji} {spec.name}",
                options=[False, True],
                default=default_vals,
            )
        else:
            # Free-form CSV for numeric grids
            raw = st.text_input(
                f"{spec.emoji} {spec.name} (comma-separated)",
                value=",".join(str(v) for v in default_vals),
            )
            cast = int if spec.kind == "int" else float
            try:
                grid[spec.name] = [cast(x.strip()) for x in raw.split(",") if x.strip()]
            except ValueError:
                st.error(f"Invalid values for {spec.name}")
                grid[spec.name] = default_vals

    axes = [len(v) for v in grid.values() if v]
    n_combos = reduce(mul, axes, 1) if axes else 0
    st.caption(f"Roughly **{n_combos}** combinations in phase 1.")

    if st.button("🧪 Run optimization", type="primary", disabled=not question.strip()):
        extras = [q.strip() for q in extra.split(",") if q.strip()] if extra else []
        progress = st.progress(0.0, text="Starting…")
        status = st.empty()

        def on_progress(phase: str, i: int, total: int) -> None:
            frac = i / max(1, total)
            # Phase 1 occupies 0–0.7, phase 2 0.7–1.0
            if phase == "phase1":
                progress.progress(min(0.7, frac * 0.7), text=f"Phase 1 retrieval {i}/{total}")
            else:
                progress.progress(0.7 + frac * 0.3, text=f"Phase 2 judge {i}/{total}")
            status.write(f"⏳ {phase}: {i}/{total}")

        try:
            with get_client() as c:
                report = run_optimize(
                    c,
                    question,
                    grid=grid,
                    questions=extras or None,
                    top_n=top_n,
                    judge=use_judge,
                    progress=on_progress,
                )
            st.session_state.opt_report = report
            progress.progress(1.0, text="Done!")
            st.success("🧪 Sweep complete!")
            st.balloons()
        except Exception as exc:  # noqa: BLE001
            st.error(f"💥 Optimize failed: {exc}")

    report = st.session_state.opt_report
    if not report:
        return

    rows = [
        {
            "final": r.final_score,
            "retrieval": r.retrieval_score,
            "judge": r.judge_score,
            "mean_doc": r.mean_doc_score,
            "n_docs": r.n_docs,
            "params": json.dumps(r.params),
        }
        for r in report.results
    ]
    df = pd.DataFrame(rows)
    st.markdown("### 🏆 Leaderboard")
    st.dataframe(df, use_container_width=True, hide_index=True)
    if not df.empty:
        st.bar_chart(df.head(15).set_index("params")["final"])

    st.markdown("### ✨ Recommended")
    st.code(json.dumps(report.recommended, indent=2), language="json")
    if st.button("✨ Apply best params to Query tab"):
        st.session_state.query_params = {
            **st.session_state.query_params,
            **report.recommended,
        }
        st.toast("Applied recommended params!", icon="✨")

    st.download_button(
        "💾 Download JSON report",
        data=report.to_json(),
        file_name="memgraphrag_optimize.json",
        mime="application/json",
    )


def tab_graph() -> None:
    st.markdown("## 🕸️ Graph explorer")
    try:
        with get_client() as c:
            labels_payload = c.list_labels()
        labels = labels_payload.get("labels") or []
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load labels: {exc}")
        labels = []

    label = st.selectbox("🏷️ Label / layer filter", options=["(all)"] + list(labels))
    limit = st.slider("Max nodes", 10, 500, 100)
    if st.button("🔍 Explore"):
        try:
            with get_client() as c, st.spinner("🕸️ Fetching graph…"):
                data = c.explore_graph(
                    label=None if label == "(all)" else label, limit=limit
                )
            nodes = data.get("nodes") or []
            edges = data.get("edges") or []
            st.metric("Nodes", len(nodes))
            st.metric("Edges", len(edges))
            st.markdown("### Nodes")
            st.dataframe(pd.DataFrame(nodes), use_container_width=True)
            st.markdown("### Edges")
            st.dataframe(pd.DataFrame(edges), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.error(f"💥 {exc}")


def main() -> None:
    _init_state()
    render_sidebar()
    st.markdown("# 🧠✨ MemGraphRAG Playground")
    st.caption("Query · Ingest · Optimize — with vibes")

    t_home, t_query, t_ingest, t_opt, t_graph = st.tabs(
        ["🏠 Home", "💬 Query", "📥 Ingest", "🧪 Optimize", "🕸️ Graph"]
    )
    with t_home:
        tab_home()
    with t_query:
        tab_query()
    with t_ingest:
        tab_ingest()
    with t_opt:
        tab_optimize()
    with t_graph:
        tab_graph()


if __name__ == "__main__":
    main()
