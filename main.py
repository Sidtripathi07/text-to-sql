"""
main.py — Schema-Mind UI v2 (FIXED)
Run: streamlit run main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import SMALL_MODEL, LARGE_MODEL, HYBRID_ASSIGNMENT
from database.db_executor import DBExecutor
from cache.schema_cache import SchemaCache
from evaluation.spider_loader import SpiderLoader
from evaluation.evaluator import Evaluator
from pipeline.baseline import run_baseline
from pipeline.full_8b import run_full_8b
from pipeline.hybrid import run_hybrid
from taxonomy.error_taxonomy import TAXONOMY

st.set_page_config(page_title="Schema-Mind", page_icon="",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .main .block-container{padding-top:1.2rem}
  .agent-card{background:#1e1e2e;border-radius:10px;padding:12px 16px;
    margin:5px 0;border-left:4px solid #7c3aed}
  .agent-card.cached{border-left-color:#059669}
  .agent-card.err{border-left-color:#dc2626}
  .live-ea{font-size:2rem;font-weight:700;color:#22c55e}
  .warn{color:#f59e0b;font-size:0.85rem}
</style>
""", unsafe_allow_html=True)

# Session state
for k, v in {"eval_results": {}, "query_result": None, "api_key": "",
              "spider_path": "./spider"}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.title("Schema-Mind")
    st.caption("Multi-Agent Text-to-SQL v2")
    st.divider()

    api_key = st.text_input("Groq API Key", type="password",
                             value=st.session_state.api_key)
    st.session_state.api_key = api_key

    spider_path = st.text_input("Spider Dataset Path",
                                 value=st.session_state.spider_path)
    st.session_state.spider_path = spider_path

    st.divider()
    st.subheader("Pipeline")
    pipeline_choice = st.selectbox("Pipeline", ["hybrid", "full_8b", "baseline"],
        format_func=lambda x: {
            "hybrid"  : "Hybrid 70B/8B",
            "full_8b" : "Full 8B",
            "baseline": "Baseline",
        }[x])

    max_corr  = st.slider("Max Corrections", 0, 3, 2)
    use_cache = st.toggle("Schema Cache", True)

    st.divider()
    st.subheader("Evaluation")
    eval_samples = st.slider("Samples", 10, 200, 100)
    eval_dataset = st.selectbox("Dataset", ["Spider Dev", "Spider-Realistic", "Spider-SYN"])
    resume_eval  = st.toggle("Resume from checkpoint", True,
                              help="Skip already-evaluated samples if run was interrupted")

    st.divider()
    # Token budget monitor
    st.subheader("Token Budget")
    st.caption(f"70B daily: 100K | 8B daily: 500K")
    with st.expander("Hybrid assignment"):
        for agent, model in HYBRID_ASSIGNMENT.items():
            label = "70B" if model == LARGE_MODEL else "8B"
            st.caption(f"{label} -> {agent}")

    if use_cache:
        cache = SchemaCache()
        stats = cache.stats()
        st.metric("Cached linkings", stats["cached_linkings"])


# =============================================================
# HELPERS
# =============================================================

def get_pipeline_fn(name: str):
    def _baseline(question, db_id, schema_str, full_schema, db_executor,
                  api_key, gold_sql="", schema_cache=None, **kw):
        return run_baseline(question, db_id, schema_str, db_executor,
                            api_key, gold_sql)
    def _full_8b(question, db_id, schema_str, full_schema, db_executor,
                 api_key, gold_sql="", schema_cache=None, **kw):
        return run_full_8b(question, db_id, schema_str, full_schema,
                           db_executor, api_key, gold_sql, max_corr, schema_cache)
    def _hybrid(question, db_id, schema_str, full_schema, db_executor,
                api_key, gold_sql="", schema_cache=None, **kw):
        return run_hybrid(question, db_id, schema_str, full_schema,
                          db_executor, api_key, gold_sql, max_corr, schema_cache)
    return {"baseline": _baseline, "full_8b": _full_8b, "hybrid": _hybrid}[name]


def render_trace(trace: dict, idx: int):
    agent  = trace.get("agent", "?")
    model  = trace.get("model", SMALL_MODEL)
    cached = trace.get("cached", False)
    error  = trace.get("error")
    ti, to = trace.get("tokens_in", 0), trace.get("tokens_out", 0)
    lat    = trace.get("latency_ms", 0)
    parsed = trace.get("parsed", {})

    is70   = model == LARGE_MODEL
    bg     = "#3b1e6e" if is70 else "#1e3a5f"
    lbl    = "70B" if is70 else "8B"
    icons  = {"schema_linking": "", "subproblem": "", "query_plan": "",
               "sql": "", "baseline_sql": "", "correction_plan": "",
               "correction_sql": ""}
    icon   = icons.get(agent, "")
    cls    = "cached" if cached else ("err" if error else "")

    st.markdown(
        f'<div class="agent-card {cls}">'
        f'<b>{icon} {agent.replace("_"," ").title()}</b> '
        f'<span style="background:{bg};color:white;border-radius:4px;'
        f'padding:1px 6px;font-size:0.72rem">{lbl}</span>'
        f'{"&nbsp; cached" if cached else ""}'
        f'{"&nbsp; " + str(error)[:60] if error else ""}'
        f'<br><span style="color:#9ca3af;font-size:0.74rem">'
        f'in:{ti} out:{to} lat:{lat}ms</span></div>',
        unsafe_allow_html=True,
    )
    if not error and not cached:
        with st.expander(f"{agent} detail", expanded=(idx == 0)):
            if agent in ("sql", "correction_sql", "baseline_sql"):
                st.code(str(parsed) or "—", language="sql")
            elif isinstance(parsed, dict):
                st.json(parsed)
            else:
                st.text(str(parsed)[:800])
            prompt = trace.get("prompt", "")
            if prompt:
                with st.expander("Raw prompt"):
                    st.text(prompt[:3000])
            raw = trace.get("raw_response", "")
            if raw:
                with st.expander("Raw LLM response"):
                    st.text(raw[:2000])


def render_exec_result(exec_res: dict):
    if not exec_res:
        return
    if exec_res.get("success"):
        rows = exec_res.get("results", [])
        cols = exec_res.get("columns", [])
        if rows:
            df = pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption(f"{len(rows)} row(s)")
        else:
            st.info("Query ran successfully — no rows returned.")
    else:
        st.error(f"SQL error: {exec_res.get('error','?')}")


# =============================================================
# TABS
# =============================================================

tab_q, tab_ev, tab_cmp, tab_arch, tab_tax, tab_diag = st.tabs([
    "Query", "Evaluate", "Compare", "Architecture",
    "Taxonomy", "Diagnosis",
])

# -------------------- TAB 1: QUERY --------------------------
with tab_q:
    st.header("Natural Language -> SQL")
    st.caption("Watch every agent reason step-by-step. Perfect for viva demo.")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_area("Question",
            placeholder="e.g. How many singers do we have?", height=80)
    with col2:
        st.metric("Pipeline", pipeline_choice)
        st.metric("Models", "70B+8B" if pipeline_choice == "hybrid" else "8B")

    db_opts = ["concert_singer", "car_1", "pets_1", "flight_2",
               "museum_visit", "world_1", "network_1"]
    try:
        loader  = SpiderLoader(spider_path)
        db_opts = list(loader.load_tables().keys())
    except Exception:
        pass

    c_db, c_btn = st.columns([2, 1])
    with c_db:
        db_id = st.selectbox("Database", db_opts)
    with c_btn:
        st.write("")
        run_q = st.button("Run", type="primary", use_container_width=True)

    if run_q:
        if not api_key:
            st.error("Enter Groq API key in sidebar.")
        elif not question.strip():
            st.error("Enter a question.")
        else:
            with st.spinner("Agents working..."):
                try:
                    dbe  = DBExecutor(spider_path)
                    fs   = dbe.get_schema_raw(db_id)
                    ss   = dbe.schema_to_string(fs)
                    cch  = SchemaCache() if use_cache else None
                    fn   = get_pipeline_fn(pipeline_choice)
                    res  = fn(question=question, db_id=db_id, schema_str=ss,
                              full_schema=fs, db_executor=dbe, api_key=api_key,
                              schema_cache=cch)
                    st.session_state.query_result = res
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.query_result:
        res = st.session_state.query_result
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tokens",      getattr(res, "total_tokens", "—"))
        c2.metric("Latency",     f"{getattr(res,'total_latency',0)/1000:.1f}s")
        c3.metric("Corrections", getattr(res, "corrections", 0))
        c4.metric("Valid SQL",   "Yes" if getattr(res,"valid_sql",False) else "No")

        st.subheader("Generated SQL")
        st.code(getattr(res, "final_sql", "—"), language="sql")

        st.subheader("Execution Result")
        render_exec_result(getattr(res, "exec_result", {}))

        st.subheader("Agent Traces — Proof of Multi-Agent Design")
        st.caption("Every agent's prompt and response is logged below.")
        for i, tr in enumerate(getattr(res, "traces", [])):
            render_trace(tr, i)

        traces = getattr(res, "traces", [])
        if len(traces) > 1:
            tok_df = pd.DataFrame([
                {"Agent": t.get("agent","?"),
                 "Tokens": t.get("tokens_in",0)+t.get("tokens_out",0),
                 "Model": "70B" if t.get("model")==LARGE_MODEL else "8B"}
                for t in traces if not t.get("cached") and not t.get("error")
            ])
            if not tok_df.empty:
                fig = px.bar(tok_df, x="Agent", y="Tokens", color="Model",
                             color_discrete_map={"70B":"#7c3aed","8B":"#2563eb"},
                             title="Token usage per agent")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)


# -------------------- TAB 2: EVALUATE -----------------------
with tab_ev:
    st.header("Benchmark Evaluation")

    if not api_key:
        st.warning("Set API key in sidebar.")

    # Warn about token budget
    est_tok_per_q = {"baseline": 350, "full_8b": 700, "hybrid": 800}[pipeline_choice]
    est_70b_calls = {"baseline": 0, "full_8b": 0, "hybrid": 3}[pipeline_choice]
    total_70b     = est_70b_calls * eval_samples
    st.info(
        f"Estimated: ~{est_tok_per_q * eval_samples:,} total tokens | "
        f"~{total_70b} 70B calls ({total_70b} of 1000/day limit). "
        f"{'Close to 70B daily limit!' if total_70b > 700 else 'Within limits.'}"
    )

    run_ev = st.button("Run Evaluation", type="primary",
                        disabled=not api_key or not spider_path)

    if run_ev:
        try:
            loader = SpiderLoader(spider_path)
            if "Realistic" in eval_dataset:
                dataset = loader.load_realistic(eval_samples)
            elif "SYN" in eval_dataset:
                dataset = loader.load_syn(eval_samples)
            else:
                dataset = loader.load_dev(eval_samples)

            evaluator = Evaluator(spider_path, api_key, use_cache)
            fn        = get_pipeline_fn(pipeline_choice)

            prog   = st.progress(0)
            status = st.empty()
            live   = st.empty()

            def cb(done, total, last, running_ea):
                prog.progress(done / total)
                status.text(
                    f"[{done}/{total}] {last['question'][:55]}... "
                    f"| EA: {running_ea:.1f}% "
                    f"| {'Pass' if last['exec_accurate'] else 'Fail'}"
                )
                live.markdown(
                    f'<span class="live-ea">{running_ea:.1f}% EA</span> '
                    f'after {done} samples',
                    unsafe_allow_html=True,
                )

            with st.spinner("Evaluating..."):
                metrics = evaluator.run(
                    pipeline_fn   = fn,
                    pipeline_name = pipeline_choice,
                    dataset       = dataset,
                    max_samples   = eval_samples,
                    progress_cb   = cb,
                    resume        = resume_eval,
                )

            st.session_state.eval_results[pipeline_choice] = metrics
            prog.progress(1.0)

        except Exception as e:
            st.error(f"Evaluation error: {e}")
            import traceback; st.text(traceback.format_exc())

    if pipeline_choice in st.session_state.eval_results:
        m = st.session_state.eval_results[pipeline_choice]
        st.divider()

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Execution Accuracy", f"{m.ea:.1f}%")
        c2.metric("Valid SQL %",        f"{m.valid_pct:.1f}%")
        c3.metric("Total Samples",       m.total)
        c4.metric("Avg Tokens",         f"{m.avg_tokens:.0f}")
        c5.metric("Corrections",         m.corrections_made)

        if m.per_difficulty:
            dd = [{"Difficulty": k,
                   "EA (%)": round(v["exec_accurate"]/v["total"]*100,1) if v["total"] else 0,
                   "N": v["total"]}
                  for k, v in m.per_difficulty.items()]
            fig = px.bar(pd.DataFrame(dd), x="Difficulty", y="EA (%)", text="N",
                         title="EA by difficulty", color="EA (%)",
                         color_continuous_scale="viridis")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

        if m.error_codes:
            ed = sorted(m.error_codes.items(), key=lambda x:-x[1])[:15]
            fig2 = px.bar(pd.DataFrame(ed, columns=["Code","Count"]),
                          x="Code", y="Count", title="Correction error codes",
                          color="Count", color_continuous_scale="reds")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

        df_res = pd.DataFrame([{
            "#": r["index"], "Question": r["question"][:55]+"...",
            "DB": r["db_id"], "EA": "Pass" if r["exec_accurate"] else "Fail",
            "Valid": "Yes" if r["valid_sql"] else "No",
            "Tokens": r["tokens"], "Corr.": r["corrections"],
        } for r in m.results])
        st.dataframe(df_res, use_container_width=True, height=380)
        st.download_button("Download CSV", df_res.to_csv(index=False),
                           f"eval_{pipeline_choice}_v2.csv")


# -------------------- TAB 3: COMPARE ------------------------
with tab_cmp:
    st.header("Pipeline Comparison")
    stored = st.session_state.eval_results

    if len(stored) < 2:
        st.info("Run at least 2 pipelines from the Evaluate tab.")
    else:
        names = list(stored.keys())
        eas   = [stored[n].ea for n in names]
        vlds  = [stored[n].valid_pct for n in names]
        toks  = [stored[n].avg_tokens for n in names]

        fig = go.Figure(data=[go.Bar(
            x=names, y=eas,
            text=[f"{v:.1f}%" for v in eas], textposition="outside",
            marker_color=["#dc2626","#2563eb","#16a34a"][:len(names)],
        )])
        fig.update_layout(title="Execution Accuracy Comparison",
                          yaxis_title="EA (%)", yaxis_range=[0,100],
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        df_cmp = pd.DataFrame({
            "Pipeline"          : names,
            "EA (%)"            : eas,
            "Valid SQL (%)"     : vlds,
            "Avg Tokens/Query"  : toks,
            "Total Corrections" : [stored[n].corrections_made for n in names],
        }).set_index("Pipeline")
        st.dataframe(df_cmp.style.format({
            "EA (%)": "{:.1f}", "Valid SQL (%)": "{:.1f}",
            "Avg Tokens/Query": "{:.0f}",
        }), use_container_width=True)

        if "baseline" in stored and "hybrid" in stored:
            g = stored["hybrid"].ea - stored["baseline"].ea
            color = "success" if g > 0 else "error"
            getattr(st, color)(
                f"{'Pass' if g>0 else 'Fail'} Hybrid vs Baseline: "
                f"{'+'  if g>0 else ''}{g:.1f}% EA"
            )
        if "full_8b" in stored and "hybrid" in stored:
            g2 = stored["hybrid"].ea - stored["full_8b"].ea
            getattr(st, "success" if g2>0 else "warning")(
                f"{'Pass' if g2>0 else 'Warning'} Hybrid vs Full-8B: "
                f"{'+'  if g2>0 else ''}{g2:.1f}% EA"
            )


# -------------------- TAB 4: ARCHITECTURE -------------------
with tab_arch:
    st.header("Agent Architecture")
    st.caption("Multi-agent pipeline — 6 distinct LLM calls per query, each with a specialized role.")

    arch_data = [
        {"Stage": "1", "Agent": "Schema Linking", "Model": "70B",
         "Input": "Question + Full schema",
         "Output": "Relevant tables/columns JSON",
         "Purpose": "Reduce schema to only what's needed; join keys identified"},
        {"Stage": "2", "Agent": "Subproblem", "Model": "8B",
         "Input": "Question + Linked schema",
         "Output": "SQL clause dict (WHERE, GROUP BY, ...)",
         "Purpose": "Decompose question into SQL clause-level sub-tasks"},
        {"Stage": "3", "Agent": "Query Plan", "Model": "70B",
         "Input": "Question + Subproblems",
         "Output": "Numbered step-by-step plan (no SQL)",
         "Purpose": "Chain-of-Thought planning before SQL is generated"},
        {"Stage": "4", "Agent": "SQL Agent", "Model": "8B",
         "Input": "Plan + Schema",
         "Output": "SQLite SQL query",
         "Purpose": "Mechanical SQL generation guided by the plan"},
        {"Stage": "5a", "Agent": "Correction Plan", "Model": "70B",
         "Input": "Failed SQL + Error + Taxonomy",
         "Output": "Error codes + fix steps",
         "Purpose": "Taxonomy-guided error diagnosis (not just execution feedback)"},
        {"Stage": "5b", "Agent": "Correction SQL", "Model": "8B",
         "Input": "Failed SQL + Correction plan",
         "Output": "Fixed SQL query",
         "Purpose": "Targeted SQL fix using diagnosis from stage 5a"},
    ]
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    st.subheader("Why this beats a single API call")
    st.markdown("""
| Concern | Single call | Schema-Mind |
|---|---|---|
| Schema comprehension | Entire schema in one prompt | Schema Linking narrows to relevant subset |
| Query planning | Implicit | Explicit CoT plan before SQL |
| Error correction | None or blind retry | Taxonomy classifies error type -> targeted fix |
| Model efficiency | One large expensive call | Cheap 8B for mechanical steps |
| Traceability | Black box | Every agent's prompt + response logged |
    """)


# -------------------- TAB 5: TAXONOMY -----------------------
with tab_tax:
    st.header("SQL Error Taxonomy")
    st.caption("9 categories, 31 error subtypes — drives Correction Plan Agent")
    for cat, info in TAXONOMY.items():
        with st.expander(f"**{cat.upper()}** — {info['description']}"):
            st.dataframe(pd.DataFrame([
                {"Code": c, "Description": d}
                for c, d in info["subtypes"].items()
            ]), use_container_width=True, hide_index=True)

    st.subheader("Key insight for viva")
    st.markdown("""
**95-99% of LLM-generated SQL is syntactically valid.**  
Execution-only feedback (used by DIN-SQL, DAIL-SQL) only catches the 1-5% that crash.  
The real errors are *semantic* — wrong join, missing GROUP BY, wrong column.  
The taxonomy names 31 of these specifically so the correction agent can fix them precisely.
    """)


# -------------------- TAB 6: DIAGNOSIS ----------------------
with tab_diag:
    st.header("Failure Diagnosis")
    st.caption("Understand exactly why the hybrid pipeline failed on certain questions.")

    stored = st.session_state.eval_results
    if not stored:
        st.info("Run evaluation first.")
    else:
        selected_pipeline = st.selectbox("Pipeline to inspect", list(stored.keys()))
        m = stored[selected_pipeline]

        failures = [r for r in m.results if not r["exec_accurate"]]
        success  = [r for r in m.results if r["exec_accurate"]]

        st.metric("Failed queries", len(failures))
        st.metric("Passed queries", len(success))

        # DB-level breakdown
        db_ea = {
            db: round(v["exec_accurate"]/v["total"]*100, 1)
            for db, v in m.per_db.items() if v["total"] > 0
        }
        fig = px.bar(
            pd.DataFrame({"DB": list(db_ea.keys()), "EA": list(db_ea.values())}),
            x="DB", y="EA", title="EA per database",
            color="EA", color_continuous_scale="RdYlGn",
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # Correction efficiency
        zero_tok = [r for r in failures if r.get("tokens", 1) == 0]
        if zero_tok:
            st.error(
                f"Warning: {len(zero_tok)} queries returned 0 tokens — likely hit TPM limit mid-run. "
                "The v2 rate limiter should prevent this. If it recurs, reduce eval_samples "
                "or run in two batches on different hours."
            )

        st.subheader("Failed query inspector")
        if failures:
            idx = st.selectbox("Select failure", range(len(failures)),
                               format_func=lambda i: f"#{failures[i]['index']} — {failures[i]['question'][:50]}")
            f = failures[idx]
            st.write(f"**Question:** {f['question']}")
            st.write(f"**DB:** {f['db_id']} | **Corrections made:** {f['corrections']}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Predicted SQL")
                st.code(f.get("pred_sql","—"), language="sql")
            with col2:
                st.subheader("Gold SQL")
                st.code(f.get("gold_sql","—"), language="sql")

            traces = f.get("traces", [])
            if traces:
                st.subheader("Agent traces for this failure")
                for i, tr in enumerate(traces):
                    render_trace(tr, i)