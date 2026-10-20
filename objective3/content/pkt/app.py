"""KSAP learning-material recommender: interactive replay demo.

Run from the repository root:
    streamlit run content/pkt/app.py

Pick a held-out learner and how many of their attempts the system has "seen".
The frozen ASTNN-Longformer checkpoint diagnoses concept mastery from that
prefix only; KSAP then expands/orders weak concepts with the mined knowledge
graph, filters by the learner's difficulty preference, ranks unseen problems
with DMF, and attaches learning material for each targeted concept.
"""

from dataclasses import replace
import json
from pathlib import Path
import sys

import altair as alt
import pandas as pd
import streamlit as st
import torch

PKT = Path(__file__).resolve().parent
ROOT = PKT.parents[1]
sys.path.insert(0, str(PKT))
from recommendation import v2  # noqa: E402
from recommendation.data import iter_histories  # noqa: E402
from recommendation.mastery import HFMastery  # noqa: E402
from recommendation.calibration import CalibratedDiagnosis, evidence_adjust  # noqa: E402
from recommendation import live_input  # noqa: E402
from recommendation.resources import (CF_PRACTICE, PROBLEM_CODES, concept_name, explain, practice_problems, problem_label,  # noqa: E402
                                      problem_url, record_text, resources_for, study_plan)  # noqa: E402

ARCHIVE = ROOT / "data" / "codeforces.zip"
BUNDLE = PKT / "results" / "ksap_hf_v1"
CALIBRATION = PKT / "results" / "ksap_calibrated" / "calibration.json"
MODES = {
    "KSAP v2-P (personalised; default)": replace(
        v2.Config(), name="v2p", graph="window", window=100, prereq_rule="unmastered"),
    "KSAP v2 (as in the reported evaluation)": replace(v2.Config(), name="v2", graph="window", window=100),
    "KSAP v1 (paper-faithful)": v2.Config(name="v1"),
    "KSAP v2 targeted (6 weakest concepts)": replace(v2.Config(), name="t", graph="window", window=100,
                                                    weak_mode="relative", weak_k=6),
    "KSAP v2-S (weak-focused; pairs with smoothed estimate)": replace(
        v2.Config(), name="v2s", graph="window", window=100, weak_mode="relative", weak_k=6,
        weak_min_candidates=5, top_k_per_concept=1),
    "KSAP v2-C (10 weakest; pairs with calibrated estimate)": replace(
        v2.Config(), name="v2c", graph="window", window=100, weak_mode="relative", weak_k=10,
        weak_min_candidates=5, top_k_per_concept=1),
}

st.set_page_config(page_title="KSAP Recommender", layout="wide")
torch.set_num_threads(2)


@st.cache_resource(show_spinner="Loading model and fitting KSAP components on training learners…")
def load_system():
    train = list(iter_histories(ARCHIVE, "train", include_asts=False))
    tm = json.loads((BUNDLE / "training_mastery.json").read_text())
    mean_m = [sum(v[i] for v in tm.values()) / len(tm) for i in range(37)]
    art = v2.fit(train, mean_m, dmf_checkpoint=str(BUNDLE / "dmf.pt"), windows=(100,))
    return art, HFMastery(BUNDLE / "mastery.pt", "cpu"), CalibratedDiagnosis(CALIBRATION)


@st.cache_resource(show_spinner="Reading learner histories…")
def load_split(split):
    return {h.learner_id: h for h in iter_histories(ARCHIVE, split)}


@st.cache_resource(show_spinner="Loading code converter…")
def load_converter():
    ok, msg = live_input.converter_available()
    if not ok:
        return None, msg
    return live_input.load_vocab(ARCHIVE), "ok"


SMOOTHING = {"Smoothed + own record (default)": "evidence",
             "Smoothed only": "ema",
             "Calibrated (most accurate concept ranking)": "calibrated",
             "Last step only (as in the reported evaluation)": "last"}


def predict(interactions, smoothing):
    _, model, cal = load_system()
    last, ema = model.predict_both(interactions)
    if smoothing == "calibrated":
        return cal.predict(interactions, last, ema)
    if smoothing == "evidence":
        return evidence_adjust(ema, interactions)
    return last if smoothing == "last" else ema


@st.cache_data(show_spinner=False)
def diagnose(split, learner, cutoff, smoothing):
    return predict(load_split(split)[learner].interactions[:cutoff], smoothing)


art = load_system()[0]
st.title("Cognitive-diagnosis-based learning material recommender")
st.caption("Objective 3 · ASTNN–Longformer diagnosis + KSAP (knowledge structure & attempting preferences) · "
           "Codeforces dataset, held-out learners only")

with st.sidebar:
    split = st.radio("Learner group", ["valid", "test"], horizontal=True,
                     help="Both groups are unseen by the fitted recommender (fitted on training learners).")
    histories = load_split(split)
    learner = st.selectbox("Learner", sorted(histories, key=str.lower))
    n = len(histories[learner].interactions)
    cutoff = st.slider("Attempts observed so far", 1, n - 1, max(1, int(n * 0.8)),
                       help="Only these attempts are used. Later attempts are hidden and used to check the list.")
    research = st.toggle("Research mode (all variants)", value=False,
                         help="Off: the final pipeline (personalised KSAP v2-P on the smoothed + own-record "
                              "diagnosis) and the configuration used in the reported evaluation. On: every "
                              "recommender and diagnosis variant studied (see the .md result files).")
    mode_options = list(MODES) if research else list(MODES)[:2]
    est_options = list(SMOOTHING) if research else [k for k in SMOOTHING if SMOOTHING[k] in ("evidence", "last")]
    mode = st.selectbox("Recommender", mode_options)
    smoothing = SMOOTHING[st.radio(
        "Mastery estimate", est_options,
        help="Last step: the model's prediction after the latest attempt only; one new submission can move "
             "unrelated concepts by ~0.16 (up to 0.54). Smoothed: moving average over recent steps (EMA, "
             "alpha 0.1); unrelated concepts move ~0.03, with the same predictive accuracy. "
             "Smoothed + own record: additionally shifts each concept by the learner's success on it relative "
             "to their overall level, so a solved attempt raises its own topics (right direction 94% vs 88%, "
             "same accuracy on test). "
             "Calibrated: logistic layer over the model outputs and the learner's record, fitted on training "
             "learners; ranks a learner's concepts best and its scores are real success probabilities, but it "
             "changes slowly (pair it with KSAP v2-C). "
             "Reported KSAP results use 'last step'. See MASTERY_SMOOTHING.md and DIAGNOSIS_CALIBRATION.md.")]
    show_future = st.toggle("Reveal what the learner attempted next", value=True)

h = histories[learner]
key = f"{split}|{learner}|{cutoff}"
added = st.session_state.setdefault("added", {}).setdefault(key, [])
soft = st.session_state.setdefault("soft", {}).setdefault(key, [])   # per added event: P(success) if verdict unknown


def score(events):
    """Mastery for a history whose last added event may have an unknown verdict (probability mix)."""
    p = soft[-1] if soft and len(events) == len(prefix) else None
    if p is None:
        return predict(events, smoothing)
    ok = predict(tuple(events[:-1]) + (replace(events[-1], correct=1),), smoothing)
    bad = predict(tuple(events[:-1]) + (replace(events[-1], correct=0),), smoothing)
    return [p * a + (1 - p) * b for a, b in zip(ok, bad)]


base_prefix = h.interactions[:cutoff]
prefix = tuple(base_prefix) + tuple(added)
base_mastery = diagnose(split, learner, cutoff, smoothing)
mastery = base_mastery if not added else score(prefix)
cfg = MODES[mode]
out = v2.recommend(art, cfg, prefix, mastery, limit=10)
recs = out["recommendations"]

with st.expander("0 · Add your own code submission (live input)", expanded=bool(added)):
    conv, conv_msg = load_converter()
    if conv is None:
        st.warning("Live code input needs `pip install libclang==13.0.0 pcpp`. Details: " + conv_msg)
    else:
        (k2i, unk) = conv
        catalog = art.catalogs[0.0, "paper"][0]
        options = sorted(int(p) for p in catalog)
        known = sorted((p for p in PROBLEM_CODES if str(p) in catalog), key=lambda p: PROBLEM_CODES[p])
        ca, cb = st.columns([3, 1])
        with ca:
            code = st.text_area("C++ source code", height=220,
                                placeholder="#include <bits/stdc++.h>\nusing namespace std;\nint main() { ... }")
        with cb:
            source = st.radio("Problem attempted", ["Another problem (pick its topics)",
                                                     "Identified Codeforces problem", "Catalogue problem (internal ID)"],
                              help="The diagnosis uses the problem's topic tags, the code and the verdict; "
                                   "the problem ID only matters for not recommending it again.")
            if source.startswith("Another"):
                tags = st.multiselect("Topics of the problem", list(range(1, 38)), format_func=concept_name,
                                      placeholder="e.g. greedy, math")
                pid = live_input.CUSTOM_PROBLEM_ID
            else:
                pool = known if source.startswith("Identified") else options
                pid = st.selectbox("Problem", pool, format_func=lambda p: (
                    f"{PROBLEM_CODES[p]} · " if p in PROBLEM_CODES else f"{p}: ") + ", ".join(
                    concept_name(c) for c in catalog[str(p)]["concept_ids"]))
                tags = catalog[str(pid)]["concept_ids"]
            verdict = st.radio("Verdict", ["Not known (estimate from history)", "Accepted", "Wrong / failed"],
                               help="The model cannot judge code: it forecasts success from the history, before "
                                    "seeing the code. If you do not know the verdict, the update averages both "
                                    "outcomes, weighted by that forecast. Only real judging (running tests) knows.")
            go = st.button("Add to this learner's history", type="primary", disabled=not code.strip() or not tags)
            if added and st.button("Remove added submissions"):
                added.clear()
                soft.clear()
                st.rerun()
        if go:
            try:
                roots, stats = live_input.code_to_roots(code, k2i, unk)
                p_ok = live_input.next_attempt_probability(load_system()[1].predict_both(prefix)[0], tags)
                known = not verdict.startswith("Not known")
                correct = verdict == "Accepted" if known else p_ok >= 0.5   # earlier unknown events keep the likelier verdict
                event = live_input.make_event(learner, prefix, pid, tags, correct, roots)
                added.append(event)
                soft.append(None if known else p_ok)
                st.session_state["last_forecast"] = p_ok
                st.session_state["last_stats"] = stats
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not convert this code: {exc}")
        if added:
            stats = st.session_state.get("last_stats", {})
            # Same last attempt, same topics and verdict, but with a minimal program instead of the code:
            # the difference is what the code itself contributed.
            if "minimal_roots" not in st.session_state:
                st.session_state["minimal_roots"] = live_input.code_to_roots("int main() { return 0; }", k2i, unk)[0]
            neutral = tuple(prefix[:-1]) + (replace(prefix[-1], ast_roots=st.session_state["minimal_roots"]),)
            from_code = [a - b for a, b in zip(mastery, score(neutral))]
            st.success(f"{len(added)} submission(s) added after attempt {cutoff}. Last one: {stats.get('blocks', '?')} statement "
                       f"blocks, {stats.get('unknown_tokens', 0)}/{stats.get('tokens', '?')} tokens outside the training vocabulary.")
            p_last = st.session_state.get("last_forecast")
            if soft and soft[-1] is not None:
                st.info(f"Verdict not given. From the history alone the model expected a **{soft[-1]:.0%}** chance of "
                        "success on a problem with these topics (it does not read the code to decide this); the update "
                        "below averages the accepted and failed outcomes with that weight.")
            elif p_last is not None:
                st.caption(f"For reference, the model's forecast before seeing this code was {p_last:.0%} success.")
            summ = stats.get("summary")
            if summ:
                seen, written = summ["loops"], summ["loops_in_source"]
                lines = [
                    f"**Functions:** {', '.join(summ['functions']) or 'none'}"
                    + (f" · recursive: {', '.join(summ['recursive_functions'])}" if summ["recursive_functions"] else ""),
                    f"**Loops:** {written} written, **{seen} visible to the model**"
                    + (f" (max nesting {summ['max_loop_nesting']})" if seen else "")
                    + (" — loops whose condition uses a library type (string, vector, map…) are dropped by the "
                       "dataset's preprocessing" if seen < written else ""),
                    f"**Conditions:** {summ['conditions']} · **array accesses:** {summ['array_accesses']} · "
                    f"**bit operations:** {summ['bit_operations']} · **modulo:** {summ['modulo']} · "
                    f"**calls to your own functions:** {summ['calls_to_own_functions']}",
                ]
                if summ["library_calls_not_visible"]:
                    lines.append("**Used in your code but invisible to the model:** "
                                 + ", ".join(f"`{x}`" for x in summ["library_calls_not_visible"])
                                 + " — the dataset's preprocessing parses code without the C++ standard library, so "
                                   "these calls (and statements built on them) never reach the model.")
                st.markdown("**What the model saw in this code**\n\n" + "\n".join(f"- {x}" for x in lines))
            practised = sorted({c - 1 for e in added for c in e.concept_ids})
            others = [i for i in sorted(range(37), key=lambda i: -abs(mastery[i] - base_mastery[i])) if i not in practised]
            delta = practised + others[:max(3, 6 - len(practised))]
            code_col = "from the code itself" if len(added) == 1 else "from the last code itself"
            st.dataframe(pd.DataFrame({"concept": [concept_name(i + 1) + ("  ← this problem" if i in practised else "")
                                                   for i in delta],
                                       "before": [round(base_mastery[i], 3) for i in delta],
                                       "after": [round(mastery[i], 3) for i in delta],
                                       "total change": [round(mastery[i] - base_mastery[i], 3) for i in delta],
                                       code_col: [round(from_code[i], 3) for i in delta]}), hide_index=True)
            st.caption("**Total change** is everything the new attempt implies: its verdict, its topics and its code. "
                       "**From the code itself** compares against the same attempt (same topics, same verdict) with a "
                       "minimal program instead of your code. Across 204 real attempts, the verdict moves the diagnosis "
                       "most, then the topics, then the code (about 40% of an attempt's effect). The concept names are "
                       "the problem's tags, not something read from the code. The code-to-tree conversion matches the "
                       "dataset exactly on 48 real Codeforces submissions (42 held out).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Attempts observed", len(prefix), f"of {n} recorded" + (f" + {len(added)} added" if added else ""), delta_color="off")
c2.metric("Success rate so far", f"{sum(e.correct for e in prefix) / len(prefix):.0%}")
c3.metric("Weak concepts diagnosed", len(out["weak_concepts"]))
c4.metric("Preferred difficulty", out["difficulty_preference"]["selected_bucket"].title())

left, right = st.columns([3, 2])
with left:
    st.subheader("1 · Cognitive diagnosis (concept mastery)")
    df = pd.DataFrame({"concept": [concept_name(i) for i in range(1, 38)], "mastery": mastery,
                       "status": ["weak" if i in out["weak_concepts"] else "ok" for i in range(1, 38)]})
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3, height=11).encode(
        x=alt.X("mastery:Q", scale=alt.Scale(domain=[0, 1]), title="estimated probability of success"),
        y=alt.Y("concept:N", sort="x", title=None, axis=alt.Axis(labelOverlap=False, labelLimit=160)),
        color=alt.Color("status:N", scale=alt.Scale(domain=["weak", "ok"], range=["#eb6834", "#a3a29c"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["concept", alt.Tooltip("mastery:Q", format=".2f"), "status"],
    ).properties(height=37 * 17)
    st.altair_chart(chart, width="stretch")
    st.caption("Scores come from the frozen ASTNN–Longformer using only the observed attempts. "
               "They are uncalibrated estimates, not exam grades.")
with right:
    st.subheader("2 · Knowledge-structure order")
    st.write("Weak concepts, expanded with association prerequisites and put in prerequisite order:")
    st.markdown(" → ".join(f"**{concept_name(c)}**" if c in out["weak_concepts"] else concept_name(c)
                           for c in out["ordered_concepts"][:12]) + (" → …" if len(out["ordered_concepts"]) > 12 else ""))
    st.caption("Bold = diagnosed weak; plain = prerequisite added by the knowledge graph.")
    st.subheader("3 · Attempting preference (DPM)")
    pref = out["difficulty_preference"]
    st.dataframe(pd.DataFrame({
        "bucket": list(pref["scores"]), "attempts": [pref["counts"][b] for b in pref["scores"]],
        "solved": [pref["correct_counts"][b] for b in pref["scores"]],
        "preference score": [round(v, 3) for v in pref["scores"].values()]}), hide_index=True)

st.subheader("4 · What to work on")
graph_used = art.graphs.get(f"window{cfg.window}" if cfg.graph == "window" else "learner")
plan = study_plan(recs, out["ordered_concepts"], out["weak_concepts"], mastery, prefix, graph_used)
if plan["weakest"]:
    step_of = {s["concept"]: i for i, s in enumerate(plan["steps"], 1)}
    attempts_on = {c: [e.correct for e in prefix if c in e.concept_ids] for c, _ in plan["weakest"]}
    st.markdown("**Your weakest concepts right now:**")
    rows = ["| Concept | Mastery | Your record | In the plan | Start with |", "|---|---|---|---|---|"]
    for c, m in plan["weakest"]:
        res = resources_for(c)
        start = " · ".join(f"[{x['title'].split(' (')[0]}]({x['url']})" for x in [r for r in res if r["type"] == "read"][:2]) \
            or f"[practice problems]({res[-1]['url']})"
        rows.append(f"| **{concept_name(c)}** | {m:.2f} | {record_text(len(attempts_on[c]), sum(attempts_on[c]))} | "
                    + (f"step {step_of[c]}" if c in step_of else "later (the graph orders other concepts first)") + f" | {start} |")
    st.markdown("\n".join(rows))
if plan["steps"]:
    st.markdown("**Study plan, in this order** (prerequisites first, so each step prepares the next):")
    rows = ["| Step | Concept | Mastery | Why | Your record | Read first | Practise |", "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(plan["steps"], 1):
        if s["role"] == "weak":
            why = "weak concept"
        elif s["leads_to"]:
            why = "prerequisite for " + ", ".join(concept_name(c) for c in s["leads_to"][:2])
        else:
            why = "prerequisite of your weak concepts"
        read = " · ".join(f"[{x['title'].split(' (')[0]}]({x['url']})" for x in s["reading"][:2]) or "—"
        rows.append(f"| {i} | **{concept_name(s['concept'])}** | {s['mastery']:.2f} | {why} | "
                    f"{record_text(s['attempts'], s['solved'])} | {read} | problems "
                    + ", ".join(f"#{k}" for k in s["ranks"]) + " below |")
    st.markdown("\n".join(rows))
    if plan["later"]:
        st.caption("Also weak, for after these: " + ", ".join(concept_name(c) for c in plan["later"][:8])
                   + (" …" if len(plan["later"]) > 8 else "") + ".")
st.caption("Weakest = lowest estimated mastery. The plan starts where the knowledge graph says to start, which "
           "is not always the single weakest concept: a weak concept's prerequisites come first.")

st.subheader("5 · Recommended problems and learning material")
future = {e.problem_id for e in h.interactions[cutoff:]} - {e.problem_id for e in prefix}
seen_codes = {PROBLEM_CODES[e.problem_id] for e in prefix if e.problem_id in PROBLEM_CODES}
if not recs:
    st.warning(out.get("empty_reason") or "No eligible problems.")
for rank, r in enumerate(recs, 1):
    hit = show_future and r["problem_id"] in future
    title = (f"#{rank} · {problem_label(r['problem_id'])} · target: {concept_name(r['target_concept'])} · "
             f"{r['difficulty_bucket']}" + ("  ✅ attempted later" if hit else ""))
    with st.expander(title, expanded=rank <= 2):
        st.write(explain(r, pref, mastery, weak=set(out["weak_concepts"])))
        st.write("Tags: " + ", ".join(concept_name(c) for c in r["concept_ids"]) +
                 f" · training learners solved {r['difficulty_training_correct']}/{r['difficulty_training_attempts']} attempts")
        if problem_url(r["problem_id"]):
            st.markdown(f"**Open the problem:** [{problem_label(r['problem_id'])}]({problem_url(r['problem_id'])})")
        st.markdown("**Learn first:** " + " · ".join(f"[{x['title']}]({x['url']})"
                                                     for x in resources_for(r["target_concept"], r["difficulty_bucket"])))
        extra = practice_problems(r["target_concept"], r["difficulty_bucket"], exclude=seen_codes)
        if extra:
            st.markdown("**More practice on Codeforces:** " + " · ".join(
                f"[{p['code']} {p['name']}]({p['url']}) ({p['rating']})" for p in extra))
if show_future and recs:
    hits = sum(r["problem_id"] in future for r in recs)
    st.info(f"Check against the hidden future: {hits} of these {len(recs)} problems were attempted later by this learner "
            f"(out of {len(future)} new problems they went on to try).")
st.caption("Problems are named by their Codeforces code where it was recovered (23 problems, via the Codeforces API); "
           "others show the dataset's internal ID, because the source data does not include that mapping. "
           "Practice links open Codeforces filtered to the target topic and a rating band matching the problem's "
           "difficulty (easy 800-1300, moderate 1300-1800, difficult 1800-2400)"
           + ("; extra problems are the most-solved Codeforces problems in that band." if CF_PRACTICE else "."))
