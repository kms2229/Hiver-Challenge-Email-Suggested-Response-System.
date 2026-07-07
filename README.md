# Hiver AI Email Suggested-Response System

An end-to-end system that generates suggested replies to incoming customer-support emails using **Retrieval-Augmented Generation (RAG)** and advanced generation modes, then measures reply quality with a principled **multi-dimensional accuracy system** (including a Debate-as-Judge evaluator, Layer 1 deterministic guardrails, and grounding auditing).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Architecture](#system-architecture)
3. [The Dataset](#the-dataset)
4. [Advanced Retrieval Engine](#advanced-retrieval-engine)
5. [The Generator (RAG) & 2026 Power Prompting](#the-generator-rag--2026-power-prompting)
6. [Advanced Generation Modes](#advanced-generation-modes)
7. [The Accuracy System](#the-accuracy-system)
8. [Running the Evaluation](#running-the-evaluation)
9. [Understanding the Report](#understanding-the-report)
10. [DPO Preference Logging](#dpo-preference-logging)
11. [Trade-offs & Design Decisions](#trade-offs--design-decisions)
12. [How I Used AI Tools](#how-i-used-ai-tools)

---

## Quick Start

### Prerequisites

- Python ≥ 3.11
- [`uv`](https://docs.astral.sh/uv/) (fast Python package manager)
- An OpenAI API key

### Installation

```bash
# Clone the repo
git clone https://github.com/your-username/hiver-email-reply
cd hiver-email-reply

# Install dependencies (uv creates the venv automatically)
uv sync

# Configure your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Step 1 — Generate the dataset

```bash
uv run python scripts/generate_dataset.py
```

This calls GPT-4o-mini to create `data/emails.json` (200 examples) and `data/calibration.json` (10-example validation subset). Takes ~3 minutes and costs ~$0.15.

### Step 2 — Generate a reply for a single email (with Sentiment Classifier + Audience calibration)

```bash
uv run python -m src.cli generate \
  --subject "My invoice shows a charge I don't recognise" \
  --body "Hi, I noticed a $99 charge on my card from last month that I can't explain. I only signed up for the free tier. Can you help? — Jane" \
  --classify \
  --audience "a frustrated customer facing financial discrepancy" \
  --stakes "Preventing immediate customer churn"
```

### Step 3 — Run the full evaluation pipeline

```bash
# Evaluate on 20 randomly-sampled emails (default) using standard mode
uv run python evaluate.py --sample 20

# Evaluate all 200 emails using Mixture-of-Agents mode
uv run python evaluate.py --sample 0 --mode moa
```

### Step 4 — Run the unit test suite (38 unit tests covering all features)

```bash
uv run python -m pytest tests/ -v
```

---

## System Architecture

```
  New email
      │
      ├────────────────────►  Classifier (sentiment, urgency, risk check)
      ▼
  ┌─────────────────────────────────────────────────────────┐
  │             Advanced RAG Retrieval Engine               │
  │  Option 1: Dense (FAISS only)                           │
  │  Option 2: Hybrid (FAISS + BM25 RRF Fusion)             │
  │  Option 3: HyDE (Hypothetical Document → Hybrid)       │
  │  + Agentic RAG Auto-Requery if similarity is low        │
  └───────────────────────┬─────────────────────────────────┘
                          │ top-k past email+reply pairs
                          ▼
  ┌───────────── Choose generation mode ─────────────────────┐
  │                                                           │
  │  standard ──► Structured JSON (greeting, body, action)   │
  │               Enforces "Before You Answer" Technique      │
  │                                                           │
  │  refine   ──► Draft → Self-Critique → Revised reply       │
  │                                                           │
  │  moa      ──► N Candidates → Synthesizer + Recommendation │
  │                                                           │
  │  debate   ──► Composer ◄──► Critic (N rounds)             │
  │                    └──────────────► Judge ─► reply         │
  └───────────────────────┬─────────────────────────────────┘
                          │ suggested reply
                          ▼
  ┌─────────────────────────────────────────────────────────┐
  │              5-Metric Evaluator & Guardrails            │
  │  Layer 1 Deterministic Guardrails (instant, zero cost)  │
  │  Layer 2 LLM-as-Judge & Overlaps:                       │
  │    ① Embedding cosine similarity    (weight 0.30)        │
  │    ② ROUGE-L recall                 (weight 0.20)        │
  │    ③ Tone score — single/debate     (weight 0.20)        │
  │    ④ Quality score — single/debate  (weight 0.10)        │
  │    ⑤ Faithfulness / Grounding       (weight 0.20)        │
  │                                                           │
  │  → Composite score (0–1) + DPO preference logging        │
  └─────────────────────────────────────────────────────────┘
```

---

## The Dataset

**Location:** `data/emails.json`  
**Size:** 200 email/reply pairs  
**Generation script:** `scripts/generate_dataset.py`

A **synthetic dataset** generated by GPT-4o-mini that simulates a B2B SaaS customer-support inbox — the core Hiver use case. Each record contains:

| Field | Description |
|-------|-------------|
| `id` | Unique integer ID |
| `subject` | Email subject line |
| `from_name` | Customer name (generated) |
| `from_email` | Customer email address (generated) |
| `company` | Company name (generated) |
| `body` | Incoming email body |
| `reply` | Ground-truth support reply |
| `category` | One of 10 categories (such as `billing_and_subscription`, `bug_report`) |
| `tone` | Expected reply tone: empathetic / formal / brief |
| `persona` | Customer persona used for generation |

---

## Advanced Retrieval Engine

Located in `src/dataset.py`, the retrieval engine is modular and supports:
1. **Dense Retrieval:** FAISS cosine similarity searches using `text-embedding-3-small` embeddings.
2. **Hybrid (Dense + Sparse) Retrieval:** Leverages **BM25** (using `rank-bm25`) for keyword exact matching and fuses it with dense retrieval using **Reciprocal Rank Fusion (RRF)**.
3. **HyDE (Hypothetical Document Embeddings):** Generates a hypothetical reply first, then uses it to perform hybrid search, improving retrieval accuracy for vague or complex customer emails.
4. **Cross-Encoder Reranking (Local Option):** Includes hooks for `sentence-transformers` cross-encoders to perform secondary reranking.

---

## The Generator (RAG) & 2026 Power Prompting

Located in `src/generator.py`, the generator incorporates several core prompts and principles from **2026 Power Prompting Techniques**:

- **Technique 1: "Before You Answer" Instruction**
  The LLM JSON schema forces the model to fill out an `"interpretation"` (considering 2-3 different ways to interpret the issue) and `"hidden_assumptions"` array *before* drafting the actual greeting and body.
- **Technique 4: Specific Example Request**
  We strictly enforce that the JSON response contains an `"action_items"` list containing at least one concrete, time-bound commitment or next step.
- **Technique 6: Role + Stakes Context**
  Via CLI options `--audience` and `--stakes`, we inject explicit audience targeting and urgency constraints to calibrate tone, precision, and depth.
- **Structured Outputs:** Enforced using OpenAI JSON mode format (`greeting`, `body`, `action_items`, `sign_off`), preventing arbitrary meta-commentary.

---

## Advanced Generation Modes

Select with `--mode` in the CLI.

### Mode 1: `standard` (default)
Structured RAG generation with the Before-You-Answer chain.

### Mode 2: `refine` (Self-Refine Loop)
After generating a draft, the same model critiques its own work and applies correction loops to resolve missing details or tone issues.

### Mode 3: `moa` (Mixture-of-Agents)
Generates $N$ high-temperature candidate responses, then runs a low-temperature synthesizer to combine their strengths. Features a **synthesizer recommendation note** (Technique 2) explaining which candidate was selected and why.

### Mode 4: `debate` (Agent-Agent Debate)
A Composer agent and a Critic agent debate the response quality over multiple rounds, arbitrated by a final Judge agent.

---

## The Accuracy System

Located in `src/evaluator.py`, accuracy is measured using a dual-layer approach.

### Layer 1: Deterministic Guardrails
Run instantly at zero cost before LLM judges. Validates:
- Minimum word counts (no near-empty replies)
- Refusal language detection (e.g. "I am an AI assistant and cannot...")
- Presence of greetings and sign-offs
- Maximum word count cap (prevents bloat)

### Layer 2: 5-Metric Composite Accuracy Scorer
1. **Semantic Similarity (weight: 0.30):** Cosine similarity between embeddings.
2. **ROUGE-L Recall (weight: 0.20):** Evaluates ground-truth information coverage.
3. **Tone Score (weight: 0.20):** LLM-as-judge checks tone calibration (1-5).
4. **Quality Score (weight: 0.10):** LLM-as-judge checks overall structure and completeness (1-5).
5. **Faithfulness / Grounding Score (weight: 0.20):** Evaluates whether the generated reply hallucinated any timelines or policies not present in the RAG context.

---

## Running the Evaluation

```bash
# Run standard evaluation on a sample of 20 emails
uv run python evaluate.py --mode standard --sample 20

# Run evaluation with Mixture-of-Agents
uv run python evaluate.py --mode moa --sample 20

# Run evaluation with Agent-Agent Debate
uv run python evaluate.py --mode debate --sample 20
```

---

## DPO Preference Logging

All generation and evaluation runs are logged to a JSONL file via `src/logger.py`. This logs:
- Prompt + incoming email
- Generated reply + reference reply
- Metric scores (composite, semantic, ROUGE-L, tone, quality, faithfulness)

The logger includes a utility to format these records directly into **DPO (Direct Preference Optimization) training pairs** (chosen vs. rejected responses), filtering by high-scoring vs. low-scoring generations to enable future offline alignment and fine-tuning.

---

## File Structure

```
hiver-email-reply/
├── data/
│   ├── emails.json          # 200-example dataset (generated)
│   └── calibration.json     # 10-example calibration subset
├── scripts/
│   └── generate_dataset.py  # One-shot dataset generation
├── src/
│   ├── __init__.py
│   ├── classifier.py        # Sentiment & urgency classification
│   ├── dataset.py           # Dataset loader, BM25, FAISS hybrid index & HyDE
│   ├── generator.py         # RAG structured reply generator with Technique 1 & 4
│   ├── moa_generator.py     # Mixture-of-Agents with Technique 2
│   ├── debate_generator.py  # Agent-agent debate generator
│   ├── evaluator.py         # 5-metric accuracy scorer, Layer 1 guardrails & reference-free auditing
│   ├── logger.py            # DPO preference data logger (with PII scrubbing)
│   ├── scrubber.py          # Regex-based PII scrubber (emails, phone, credit card, API keys)
│   └── cli.py               # Unified CLI with Technique 6 options & live warning panels
├── tests/
│   ├── test_metrics.py      # Core metrics tests (38 tests)
│   └── test_security_hardening.py # Security & PII scrubber tests (8 tests)
├── results/                  # Auto-created; stores evaluation reports
├── evaluate.py              # Top-level evaluation entry point
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Security Hardening (July 2026 Audit & Resolution)

Following a comprehensive security and data leak audit, the system has been hardened to protect sensitive customer data and ensure model output safety in live environments.

### 1. Security Gaps Identified
* **PII Leakage in DPO Logs**: Raw incoming customer support emails, subjects, and generated responses containing emails, phone numbers, credit card numbers, or API keys were logged in plaintext to `logs/preference_log.jsonl`.
* **Unguarded Live Hallucinations**: Live, custom support generations lacked automatic grounding verification, making it easy for hallucinated policies or commitments to go undetected by support agents.
* **Jailbreak Vulnerability**: Direct concatenation of user subject and body text in the LLM templates without strict delimiters made the system susceptible to prompt injection attacks.

### 2. Remediation Measures Implemented
* **Anonymization Engine (`src/scrubber.py`)**: Built a robust regular-expression-based scrubbing utility to redact emails, credit cards, auth tokens, API keys (e.g., `sk-...`), and phone numbers (using negative lookbehinds `(?<!\d)` to correctly isolate digit blocks from special prefix characters).
* **PII Log Protection**: Embedded the scrubber into `src/logger.py` to strip all PII variables from logged subjects, prompt messages, and generated/reference responses before writing to disk.
* **Reference-Free Grounding Audits**: Added `evaluate_reply_reference_free` inside `src/evaluator.py`. It runs Layer 1 guardrail validations and evaluates the generated answer’s faithfulness directly against retrieved RAG contexts.
* **UI/CLI Warn Banners & Fallbacks**: 
  * If a custom generated suggestion fails guardrails or grounding scores (< 0.70), the CLI and Streamlit dashboard display prominent warning panels detailing the exact violations.
  * In the UI, a **Recommended Fallback Canned Response** is rendered, allowing the support agent to copy a safe escalation response rather than the hallucinated LLM draft.

