"""
cli.py — Unified command-line interface for the email reply system.

Commands:
  generate   Generate a suggested reply for a single email.
  evaluate   Evaluate generated replies against the dataset.

Generation modes (--mode flag):
  standard   Single-pass RAG + LLM (default, fastest)
  refine     RAG + self-critique + revision (self-refine loop)
  moa        Mixture-of-Agents: N candidates → synthesizer
  debate     Agent-Agent Debate: Composer ↔ Critic ↔ Judge
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

console = Console()

_MODES = ["standard", "refine", "moa", "debate"]


@click.group()
def cli():
    """Hiver AI email suggested-reply system."""
    pass


# ---------------------------------------------------------------------------
# generate command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--subject", "-s", required=True, help="Subject line of the incoming email.")
@click.option("--body", "-b", required=True, help="Body of the incoming email.")
@click.option(
    "--mode", "-m",
    type=click.Choice(_MODES),
    default="standard",
    show_default=True,
    help=(
        "Generation strategy: "
        "standard=single-pass RAG, "
        "refine=self-refine loop, "
        "moa=Mixture-of-Agents, "
        "debate=Agent-Agent Debate."
    ),
)
@click.option("--top-k", "-k", default=3, show_default=True, help="Number of RAG examples to retrieve.")
@click.option("--moa-candidates", default=3, show_default=True, help="Candidates for MoA mode.")
@click.option("--debate-rounds", default=2, show_default=True, help="Rounds for debate mode.")
@click.option("--no-agentic-rag", is_flag=True, default=False, help="Disable Agentic RAG re-query.")
@click.option("--data-path", default="data/emails.json", show_default=True)
@click.option("--json-out", is_flag=True, default=False, help="Output raw JSON instead of formatted display.")
@click.option(
    "--audience", default="",
    help="Who will read this reply (e.g. 'an enterprise CFO', 'a first-time customer'). "
         "Calibrates tone and depth. [Role+Stakes Technique]"
)
@click.option(
    "--stakes", default="",
    help="What is riding on this reply (e.g. '$50k renewal', 'mass email to 20,000 users'). "
         "Calibrates urgency and precision. [Role+Stakes Technique]"
)
@click.option("--classify", is_flag=True, default=False,
              help="Run sentiment/urgency classifier before generation.")
def generate(
    subject: str,
    body: str,
    mode: str,
    top_k: int,
    moa_candidates: int,
    debate_rounds: int,
    no_agentic_rag: bool,
    data_path: str,
    json_out: bool,
    audience: str,
    stakes: str,
    classify: bool,
):
    """Generate a suggested reply for a new incoming email."""
    from src.dataset import build_index

    client = OpenAI()

    with console.status("Building index…"):
        index, client = build_index(client=client, data_path=data_path)

    # Optional: run sentiment/urgency classifier
    classification = None
    if classify:
        with console.status("Classifying email sentiment + urgency…"):
            from src.classifier import classify_email
            classification = classify_email(subject, body, client)
            console.print(
                f"[dim]Classified: [bold]{classification.sentiment}[/] sentiment, "
                f"[bold]{classification.urgency}[/] urgency"
                + (" | ⚠ Escalation risk" if classification.escalation_risk else "")
                + "[/]"
            )

    with console.status(f"Generating reply [{mode} mode]…"):
        result = _dispatch_generate(
            mode=mode,
            subject=subject,
            body=body,
            index=index,
            client=client,
            top_k=top_k,
            moa_candidates=moa_candidates,
            debate_rounds=debate_rounds,
            agentic_rag=not no_agentic_rag,
            classification=classification,
            audience=audience,
            stakes=stakes,
        )

    # Run reference-free grounding/guardrails audit on generated reply
    from src.evaluator import evaluate_reply_reference_free
    audit_res = evaluate_reply_reference_free(
        generated_reply=result["generated_reply"],
        rag_context_text=result.get("rag_context_text"),
        client=client,
    )
    result["audit"] = audit_res

    if json_out:
        print(json.dumps(result, indent=2))
        return

    _display_generate_result(subject, body, mode, result, classification)


def _dispatch_generate(
    mode: str,
    subject: str,
    body: str,
    index,
    client: OpenAI,
    top_k: int,
    moa_candidates: int,
    debate_rounds: int,
    agentic_rag: bool,
    classification=None,
    audience: str = "",
    stakes: str = "",
) -> dict:
    """Route to the correct generator based on mode."""
    # Build Role+Stakes extra context if provided
    role_stakes_context = ""
    if audience or stakes:
        parts = []
        if audience:
            parts.append(f"This reply will be read by: {audience}.")
        if stakes:
            parts.append(f"Stakes: {stakes}.")
        parts.append("Factor this into the tone, precision, and depth of your reply.")
        role_stakes_context = " ".join(parts)

    if mode in ("standard", "refine"):
        from src.generator import generate_reply
        return generate_reply(
            subject=subject,
            body=body,
            index=index,
            client=client,
            top_k=top_k,
            mode=mode,
            agentic_rag=agentic_rag,
            classification=classification,
            role_stakes_context=role_stakes_context,
        )
    elif mode == "moa":
        from src.moa_generator import moa_reply
        return moa_reply(
            subject=subject,
            body=body,
            index=index,
            client=client,
            n_candidates=moa_candidates,
            top_k=top_k,
        )
    elif mode == "debate":
        from src.debate_generator import debate_reply
        return debate_reply(
            subject=subject,
            body=body,
            index=index,
            client=client,
            rounds=debate_rounds,
            top_k=top_k,
            role_stakes_context=role_stakes_context,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _display_generate_result(
    subject: str, body: str, mode: str, result: dict,
    classification=None,
) -> None:
    """Pretty-print a generation result to the console."""
    mode_labels = {
        "standard": "Standard RAG",
        "refine": "Self-Refine",
        "moa": "Mixture-of-Agents",
        "debate": "Agent-Agent Debate",
    }
    mode_label = mode_labels.get(mode, mode)

    console.print(Panel(
        f"[bold cyan]Subject:[/] {subject}\n\n[dim]{body}[/]",
        title="[bold]Incoming Email[/]",
        border_style="blue",
    ))

    # Show classification if available
    if classification:
        sentiment_colors = {"frustrated": "red", "neutral": "yellow", "satisfied": "green"}
        urgency_colors = {"low": "dim", "medium": "yellow", "high": "red", "critical": "bold red"}
        sc = sentiment_colors.get(classification.sentiment, "white")
        uc = urgency_colors.get(classification.urgency, "white")
        lines = [
            f"Sentiment: [{sc}]{classification.sentiment.upper()}[/]   "
            f"Urgency: [{uc}]{classification.urgency.upper()}[/]"
        ]
        if classification.escalation_risk:
            lines.append("[bold red]⚠ Escalation risk detected[/]")
        if classification.primary_issue:
            lines.append(f"Issue: {classification.primary_issue}")
        console.print(Panel("\n".join(lines), title="[yellow]Email Classification[/]", border_style="yellow"))

    # Show Before-You-Answer interpretation note (standard/refine modes)
    if result.get("interpretation_note"):
        console.print(Panel(
            result["interpretation_note"],
            title="[magenta]🧠 Before You Answer — Interpretation[/]",
            border_style="magenta",
        ))

    # Show mode-specific extras
    if mode == "refine" and result.get("critique"):
        console.print(Panel(
            result["critique"],
            title="[yellow]Self-Critique[/]",
            border_style="yellow",
        ))

    if mode == "moa" and result.get("candidates"):
        for i, c in enumerate(result["candidates"], 1):
            console.print(Panel(
                c,
                title=f"[dim]Candidate {i}[/]",
                border_style="dim",
            ))
        # Show recommendation note
        if result.get("recommendation_note"):
            console.print(Panel(
                result["recommendation_note"],
                title="[cyan]🏆 Synthesizer Recommendation[/]",
                border_style="cyan",
            ))

    if mode == "debate" and result.get("debate_transcript"):
        for turn in result["debate_transcript"]:
            role = turn["role"].upper()
            rnd = turn["round"]
            color = "cyan" if role == "COMPOSER" else "red"
            console.print(Panel(
                turn["content"],
                title=f"[{color}]{role} — Round {rnd}[/]",
                border_style=color,
            ))

    # Reference-free Audit feedback
    audit = result.get("audit")
    if audit:
        passed = audit.get("passed", True)
        if not passed:
            warnings = []
            if not audit.get("guardrail_pass", True):
                failures = ", ".join(audit.get("guardrail_failures", []))
                warnings.append(f"[bold red]Guardrail Failures:[/] {failures}")
            if audit.get("faithfulness_score", 1.0) < 0.70:
                exp = audit.get("faithfulness_explanation", "")
                warnings.append(f"[bold red]Grounding Failure (Score: {audit.get('faithfulness_score'):.2f}):[/] {exp}")
            console.print(Panel(
                "\n".join(warnings),
                title="[bold yellow]⚠️ WARNING: Grounding & Guardrail Audit Failed[/]",
                border_style="yellow",
            ))
        else:
            console.print(f"[dim green]✓ Live Audit Passed (Faithfulness: {audit.get('faithfulness_score'):.2f})[/]\n")

    # Final reply
    was_re_queried = result.get("was_re_queried", False)
    title_extras = []
    if was_re_queried:
        title_extras.append("🔄 Re-queried")
    if mode == "debate":
        rounds = result.get("rounds_completed", "?")
        early = result.get("accepted_early", False)
        title_extras.append(f"{'✅ Accepted early' if early else f'{rounds} rounds'}")

    # Determine border style based on audit pass status
    border_color = "green"
    if audit and not audit.get("passed", True):
        border_color = "yellow"

    title = f"[bold {border_color}]Suggested Reply[/] [dim]({mode_label}{'  ' + ' · '.join(title_extras) if title_extras else ''})[/]"
    console.print(Panel(result["generated_reply"], title=title, border_style=border_color))

    # RAG examples table
    if result.get("retrieved_examples"):
        table = Table(title="Retrieved RAG Examples", box=box.SIMPLE_HEAVY)
        table.add_column("ID", style="dim")
        table.add_column("Category")
        table.add_column("Subject")
        table.add_column("Similarity", justify="right")
        for ex in result["retrieved_examples"]:
            table.add_row(
                str(ex.get("id", "")),
                ex.get("category", ""),
                ex.get("subject", "")[:60],
                f"{ex.get('similarity', 0):.3f}",
            )
        console.print(table)


# ---------------------------------------------------------------------------
# evaluate command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--sample", "-n", default=20, show_default=True,
              help="Number of dataset emails to evaluate (0 = all).")
@click.option(
    "--mode", "-m",
    type=click.Choice(_MODES),
    default="standard",
    show_default=True,
    help="Generation mode to evaluate.",
)
@click.option("--output", "-o", default="results/evaluation_report.json",
              show_default=True, help="Path to write the JSON report.")
@click.option("--data-path", default="data/emails.json", show_default=True)
@click.option("--moa-candidates", default=3, show_default=True)
@click.option("--debate-rounds", default=2, show_default=True)
@click.option("--no-agentic-rag", is_flag=True, default=False)
@click.option("--seed", default=42, show_default=True)
@click.option("--json-out", is_flag=True, default=False)
def evaluate(
    sample: int,
    mode: str,
    output: str,
    data_path: str,
    moa_candidates: int,
    debate_rounds: int,
    no_agentic_rag: bool,
    seed: int,
    json_out: bool,
):
    """
    Evaluate the system end-to-end on a sample of the dataset.

    Generates a reply for each email using the selected --mode, then scores
    it with the 4-metric accuracy system. Writes a full JSON report.
    """
    import random
    from src.dataset import build_index, load_emails
    from src.evaluator import evaluate_reply, aggregate_results, run_calibration, results_to_dicts

    client = OpenAI()

    with console.status("Loading dataset…"):
        emails = load_emails(data_path)

    if sample and sample < len(emails):
        random.seed(seed)
        emails = random.sample(emails, sample)
        console.print(f"[dim]Sampled {len(emails)} emails (seed={seed}, mode={mode})[/]")
    else:
        console.print(f"[dim]Evaluating all {len(emails)} emails (mode={mode})[/]")

    with console.status("Building vector index…"):
        all_emails = load_emails(data_path)
        full_index, _ = build_index(emails=all_emails, client=client)

    results = []
    console.print(f"\nEvaluating {len(emails)} emails [{mode} mode]…\n")

    for i, email in enumerate(emails, 1):
        console.print(f"  [{i}/{len(emails)}] {email.get('id', '?'):>4} — {email.get('subject', '')[:55]}")
        try:
            gen_result = _dispatch_generate(
                mode=mode,
                subject=email["subject"],
                body=email["body"],
                index=full_index,
                client=client,
                top_k=3,
                moa_candidates=moa_candidates,
                debate_rounds=debate_rounds,
                agentic_rag=not no_agentic_rag,
            )
            eval_result = evaluate_reply(
                email_record=email,
                generated_reply=gen_result["generated_reply"],
                client=client,
                retrieved_example_ids=[
                    e["id"] for e in gen_result.get("retrieved_examples", [])
                ],
            )
            # Attach generation mode metadata
            eval_result.retrieved_example_ids  # ensure exists
            results.append(eval_result)
        except Exception as exc:
            console.print(f"    [red]Error:[/] {exc}")

    agg = aggregate_results(results)
    calibration = run_calibration(results)

    report = {
        "generation_mode": mode,
        "aggregate": agg,
        "calibration": calibration,
        "per_email": results_to_dicts(results),
    }

    if json_out:
        print(json.dumps(report, indent=2))
        return

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
    console.print(f"\n[green]Report written to:[/] {out_path.resolve()}\n")

    _print_summary(agg, calibration, results, mode)


def _print_summary(agg: dict, calibration: dict, results, mode: str) -> None:
    """Print a rich summary of evaluation results."""
    overall = agg.get("overall", {})
    mode_labels = {
        "standard": "Standard RAG",
        "refine": "Self-Refine",
        "moa": "Mixture-of-Agents",
        "debate": "Agent-Agent Debate",
    }
    console.rule(f"[bold cyan]Evaluation Summary — {mode_labels.get(mode, mode)}[/]")
    console.print()

    stats_table = Table(title="Overall System Scores", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Mean", justify="right")
    stats_table.add_column("Std / Info", justify="right")

    stats_table.add_row(
        "🎯 Composite Score (0–1)",
        f"[bold]{overall.get('composite_mean', 0):.4f}[/]",
        f"±{overall.get('composite_std', 0):.4f}",
    )
    stats_table.add_row("🔵 Semantic Similarity", f"{overall.get('semantic_similarity_mean', 0):.4f}", "")
    stats_table.add_row("📄 ROUGE-L Recall",       f"{overall.get('rouge_recall_mean', 0):.4f}", "")
    stats_table.add_row("🎭 Tone Score (1–5)",      f"{overall.get('tone_score_mean', 0):.2f}", "")
    stats_table.add_row("✅ Quality Score (1–5)",   f"{overall.get('quality_score_mean', 0):.2f}", "")
    stats_table.add_row("🔒 Faithfulness",          f"{overall.get('faithfulness_mean', 0):.4f}", "")
    gpr = overall.get('guardrail_pass_rate', 1.0)
    gpr_color = "green" if gpr >= 0.95 else "yellow" if gpr >= 0.80 else "red"
    stats_table.add_row(
        "🛡 Guardrail Pass Rate",
        f"[{gpr_color}]{gpr:.1%}[/]",
        f"{int(gpr * (agg.get('n_evaluated', 0)))}/{agg.get('n_evaluated', 0)} passed",
    )
    console.print()

    # Calibration
    if calibration.get("calibration_available"):
        rho = calibration.get("spearman_rho")
        if rho is not None:
            color = "green" if rho > 0.6 else "yellow" if rho > 0.3 else "red"
            console.print(
                f"[bold]Calibration:[/] Spearman ρ = [{color}]{rho:.4f}[/] "
                f"(p={calibration.get('p_value', '?'):.3f}) — {calibration.get('interpretation', '')}"
            )
            console.print()

    # Per-category
    by_cat = agg.get("by_category", {})
    if by_cat:
        cat_table = Table(title="Per-Category Results", box=box.SIMPLE)
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Mean Composite", justify="right")
        for cat, stats in by_cat.items():
            score = stats["mean_composite"]
            color = "green" if score >= 0.7 else "yellow" if score >= 0.5 else "red"
            cat_table.add_row(cat, str(stats["count"]), f"[{color}]{score:.4f}[/]")
        console.print(cat_table)
        console.print()

    # Per-email detail
    if results:
        sorted_results = sorted(results, key=lambda r: r.scores.composite_score, reverse=True)
        detail_table = Table(title="Per-Email Scores (Best → Worst)", box=box.SIMPLE_HEAVY)
        detail_table.add_column("ID",       style="dim", width=6)
        detail_table.add_column("Category", width=22)
        detail_table.add_column("Sem",      justify="right", width=6)
        detail_table.add_column("ROUGE",    justify="right", width=6)
        detail_table.add_column("Tone",     justify="right", width=5)
        detail_table.add_column("Qual",     justify="right", width=5)
        detail_table.add_column("Composite", justify="right", width=9)
        for r in sorted_results:
            s = r.scores
            c = s.composite_score
            color = "green" if c >= 0.7 else "yellow" if c >= 0.5 else "red"
            detail_table.add_row(
                str(r.email_id), r.category[:22],
                f"{s.semantic_similarity:.3f}", f"{s.rouge_recall:.3f}",
                f"{s.tone_score:.1f}", f"{s.quality_score:.1f}",
                f"[{color}]{c:.4f}[/]",
            )
        console.print(detail_table)


if __name__ == "__main__":
    cli()
