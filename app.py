"""
app.py — Streamlit web dashboard for the Hiver AI Suggested-Response System.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from src.dataset import build_index, load_emails
from src.classifier import classify_email
from src.evaluator import evaluate_reply, aggregate_results

load_dotenv()

# Set page config with high-end dark-friendly aesthetics
st.set_page_config(
    page_title="Hiver AI Suggested-Response System",
    page_icon="✉️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling
st.markdown("""
<style>
    .reportview-container {
        background: #0f1116;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stCard {
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #111827;
        margin-bottom: 1rem;
    }
    h1, h2, h3 {
        color: #f3f4f6;
    }
</style>
""", unsafe_allowed_code=True)


# Initialize OpenAI client and load index/dataset once
@st.cache_resource
def get_shared_resources():
    client = OpenAI()
    emails = load_emails("data/emails.json")
    index, client = build_index(emails=emails, client=client)
    return client, index, emails


try:
    client, index, emails_data = get_shared_resources()
except Exception as e:
    st.error(f"Error loading system resources: {e}")
    st.info("Make sure you have run `uv run python scripts/generate_dataset.py` to create the database, and set `OPENAI_API_KEY` in `.env`.")
    st.stop()


st.title("✉️ Hiver AI Suggested-Response System")
st.markdown("Generates RAG-grounded replies to support emails, with 2026 Power Prompting, Agentic Reasoning, and 5-Metric Evaluation.")

tab1, tab2, tab3 = st.tabs(["🚀 Reply Playground", "📊 Evaluation Dashboard", "🧪 Calibration & Metrics"])

# ---------------------------------------------------------------------------
# Tab 1: Reply Playground
# ---------------------------------------------------------------------------
with tab1:
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.subheader("1. Incoming Email")
        
        # Quick presets from the dataset
        preset_options = ["None (Custom Email)"] + [
            f"ID {em['id']}: {em['subject']} [{em['category']}]"
            for em in emails_data[:15]
        ]
        selected_preset = st.selectbox("Load preset email from dataset:", preset_options)
        
        preset_subject = ""
        preset_body = ""
        preset_reference = ""
        
        if selected_preset != "None (Custom Email)":
            preset_id = int(selected_preset.split(":")[0].split(" ")[1])
            selected_em = next(em for em in emails_data if em["id"] == preset_id)
            preset_subject = selected_em.get("subject", "")
            preset_body = selected_em.get("body", "")
            preset_reference = selected_em.get("reply", "")

        subject = st.text_input("Subject:", value=preset_subject or "Billing discrepancy on my invoice")
        body = st.text_area("Email Body:", value=preset_body or "Hi support, I noticed my card was billed $99 yesterday but I'm on the free plan. Can you please check and refund me? Thanks, Jane.", height=150)

        st.subheader("2. Generation Strategy")
        
        mode = st.selectbox(
            "Generation Mode:",
            ["standard", "refine", "moa", "debate"],
            help=(
                "standard: Single-pass RAG with Before-You-Answer steps\n"
                "refine: Self-refine loop (critique + revision)\n"
                "moa: Mixture-of-Agents (3 candidates merged by synthesizer)\n"
                "debate: Composer and Critic argue before Judge decides"
            )
        )
        
        retrieval_mode = st.selectbox(
            "Retrieval Mode:",
            ["hybrid", "hyde", "dense"],
            help="dense: FAISS embedding only. hybrid: BM25 keyword + FAISS fused. hyde: hypothetical reply search."
        )

        agentic_rag = st.checkbox("Enable Agentic RAG (re-query on low similarity)", value=True)
        classify = st.checkbox("Enable Sentiment/Urgency Pre-Classification", value=True)

        st.subheader("3. 2026 Power Prompting Calibration (Role+Stakes)")
        audience = st.text_input("Target Audience:", value="a frustrated customer with billing error", help="Calibrates vocabulary and tone.")
        stakes = st.text_input("Business Stakes:", value="customer churn prevention, high lifetime value", help="Calibrates urgency and precision.")

        generate_btn = st.button("Generate Suggested Reply", type="primary", use_container_width=True)

    with col_output:
        st.subheader("4. Suggestions & Internal Steps")
        
        if generate_btn:
            with st.spinner("Processing generation pipeline..."):
                # Classification
                classification = None
                if classify:
                    classification = classify_email(subject, body, client)
                    st.toast("Email sentiment & urgency classified!", icon="🏷️")
                    
                    st.markdown("### 🏷️ Sentiment & Urgency Analysis")
                    col_cls1, col_cls2, col_cls3 = st.columns(3)
                    with col_cls1:
                        st.metric("Sentiment", classification.sentiment.upper())
                    with col_cls2:
                        st.metric("Urgency", classification.urgency.upper())
                    with col_cls3:
                        st.metric("Escalation Risk", "⚠️ HIGH" if classification.escalation_risk else "🟢 Low")
                    
                    if classification.primary_issue:
                        st.caption(f"**Identified issue:** {classification.primary_issue}")

                # Build Role+Stakes context
                role_stakes_context = ""
                if audience or stakes:
                    parts = []
                    if audience:
                        parts.append(f"This reply will be read by: {audience}.")
                    if stakes:
                        parts.append(f"Stakes: {stakes}.")
                    parts.append("Factor this into the tone, precision, and depth of your reply.")
                    role_stakes_context = " ".join(parts)

                # RAG / Generation
                result = None
                if mode in ("standard", "refine"):
                    from src.generator import generate_reply
                    result = generate_reply(
                        subject=subject,
                        body=body,
                        index=index,
                        client=client,
                        top_k=3,
                        mode=mode,
                        agentic_rag=agentic_rag,
                        retrieval_mode=retrieval_mode,
                        classification=classification,
                        role_stakes_context=role_stakes_context,
                    )
                elif mode == "moa":
                    from src.moa_generator import moa_reply
                    # MoA uses shared logic
                    result = moa_reply(
                        subject=subject,
                        body=body,
                        index=index,
                        client=client,
                        n_candidates=3,
                        top_k=3,
                    )
                elif mode == "debate":
                    from src.debate_generator import debate_reply
                    result = debate_reply(
                        subject=subject,
                        body=body,
                        index=index,
                        client=client,
                        rounds=2,
                        top_k=3,
                        role_stakes_context=role_stakes_context,
                    )

                # Show Before-You-Answer details if available
                if "interpretation_note" in result and result["interpretation_note"]:
                    st.markdown("### 🧠 Technique 1: Before You Answer Chain")
                    st.info(result["interpretation_note"])

                # Show MoA Recommendation details if available
                if "recommendation_note" in result and result["recommendation_note"]:
                    st.markdown("### 🏆 Technique 2: Synthesizer Recommendation")
                    st.success(result["recommendation_note"])

                # Show Critique details in refine mode
                if mode == "refine" and result.get("critique"):
                    st.markdown("### 🔍 Self-Critique Loop")
                    st.warning(result["critique"])

                # Final reply
                st.markdown("### ✉️ Final Suggested Reply")
                st.code(result["generated_reply"], language="text")

                # Show retrieved RAG elements
                st.markdown("### 📚 Retrieved RAG References")
                retrieved_df = pd.DataFrame(result.get("retrieved_examples", []))
                if not retrieved_df.empty:
                    st.dataframe(retrieved_df[["id", "subject", "category", "similarity"]])
                else:
                    st.caption("No RAG examples retrieved.")

                # If we have preset reference, run evaluation on the fly!
                if preset_reference:
                    st.markdown("### 🎯 Live Ground-Truth Scorer")
                    eval_rec = {
                        "id": preset_id,
                        "subject": subject,
                        "body": body,
                        "reply": preset_reference,
                        "category": selected_em.get("category", "billing"),
                    }
                    eval_res = evaluate_reply(
                        email_record=eval_rec,
                        generated_reply=result["generated_reply"],
                        client=client,
                        retrieved_example_ids=[e["id"] for e in result.get("retrieved_examples", [])],
                        rag_context_text=str(result.get("retrieved_examples", "")),
                        classification=result.get("classification", {}),
                    )
                    
                    s = eval_res.scores
                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    col_m1.metric("Composite Score", f"{s.composite_score:.2f}")
                    col_m2.metric("Semantic Similarity", f"{s.semantic_similarity:.2f}")
                    col_m3.metric("ROUGE-L Recall", f"{s.rouge_recall:.2f}")
                    col_m4.metric("Faithfulness", f"{s.faithfulness_score:.2f}")
                    col_m5.metric("Guardrail Pass", "🟢 Yes" if s.guardrail_pass else "❌ No")
                    
                    if not s.guardrail_pass and s.guardrail_failures:
                        st.error(f"Guardrail failures detected: {', '.join(s.guardrail_failures)}")

# ---------------------------------------------------------------------------
# Tab 2: Evaluation Dashboard
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Performance Evaluation Metrics")
    
    # Try loading existing evaluation report
    report_file = Path("results/evaluation_report.json")
    if report_file.exists():
        try:
            with report_file.open() as f:
                report_data = json.load(f)
            
            st.success(f"Loaded evaluation report for mode: **{report_data.get('generation_mode', 'standard')}**")
            
            # Key statistics card
            overall = report_data.get("aggregate", {}).get("overall", {})
            
            col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
            col_stat1.metric("Mean Composite Score", f"{overall.get('composite_mean', 0.0):.4f}")
            col_stat2.metric("Semantic Similarity Mean", f"{overall.get('semantic_similarity_mean', 0.0):.4f}")
            col_stat3.metric("ROUGE-L Recall Mean", f"{overall.get('rouge_recall_mean', 0.0):.4f}")
            col_stat4.metric("Faithfulness Score Mean", f"{overall.get('faithfulness_mean', 0.0):.4f}")
            col_stat5.metric("Guardrail Pass Rate", f"{overall.get('guardrail_pass_rate', 1.0):.1%}")

            # Plot metric distributions
            per_email_list = report_data.get("per_email", [])
            if per_email_list:
                df_eval = pd.DataFrame([
                    {
                        "ID": r["email_id"],
                        "Category": r["category"],
                        "Composite": r["scores"]["composite_score"],
                        "Semantic": r["scores"]["semantic_similarity"],
                        "ROUGE-L": r["scores"]["rouge_recall"],
                        "Faithfulness": r["scores"].get("faithfulness_score", 1.0),
                        "Guardrail Pass": r["scores"].get("guardrail_pass", True)
                    }
                    for r in per_email_list
                ])

                # Charts
                st.markdown("### Performance Distributions")
                fig_box = px.box(
                    df_eval, 
                    y=["Composite", "Semantic", "ROUGE-L", "Faithfulness"],
                    title="Metric Distributions across Evaluated Emails",
                    color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
                )
                st.plotly_chart(fig_box, use_container_width=True)

                st.markdown("### Category Breakdown")
                df_cat = df_eval.groupby("Category")["Composite"].mean().reset_index()
                fig_bar = px.bar(
                    df_cat.sort_values(by="Composite", ascending=False),
                    x="Composite",
                    y="Category",
                    orientation="h",
                    title="Mean Composite Accuracy Score by Category",
                    color="Composite",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                st.markdown("### Per-Email Scores Detail Table")
                st.dataframe(df_eval)
        except Exception as e:
            st.error(f"Error reading evaluation report: {e}")
    else:
        st.info("No evaluation report found. Run `uv run python evaluate.py` to compile performance metrics first.")

# ---------------------------------------------------------------------------
# Tab 3: Calibration & Metrics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Metric Validation: Spearman Calibration")
    st.markdown("""
    To verify that our composite scorer accurately reflects real human quality rankings,
    we run correlation analysis (Spearman's rank correlation $\\rho$) against a human calibration dataset.
    """)

    if report_file.exists():
        try:
            with report_file.open() as f:
                report_data = json.load(f)
            
            cal = report_data.get("calibration", {})
            if cal.get("calibration_available"):
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    st.metric("Spearman Correlation ρ", f"{cal.get('spearman_rho', 0.0):.4f}")
                    st.caption(f"p-value: {cal.get('p_value', 0.0):.4f}")
                with col_c2:
                    st.subheader("Interpretation")
                    st.success(cal.get("interpretation", "Calibration data matches scorer well."))
                
                # Render calibration line
                y_val = cal.get("spearman_rho", 0.0)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = y_val,
                    title = {'text': "Spearman Rank Correlation (ρ)"},
                    gauge = {
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "#00CC96"},
                        'steps' : [
                            {'range': [-1, 0.3], 'color': "#EF553B"},
                            {'range': [0.3, 0.6], 'color': "#FECB52"},
                            {'range': [0.6, 1.0], 'color': "#00CC96"}
                        ],
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.info("Calibration calculation was skipped because evaluation subset did not intersect with the calibration records.")
        except Exception as e:
            st.error(f"Error parsing calibration: {e}")
    else:
        st.info("Run `uv run python evaluate.py` to execute calibration validation metrics.")
