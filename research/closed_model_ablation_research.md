# 🔬 Closed-Model Ablation Research
## Model Unfetter — API Backend Extension

**Research Date:** February 16, 2026  
**Purpose:** Comprehensive landscape analysis before building the closed-model unfettering module.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Existing Tools & Frameworks](#2-existing-tools--frameworks)
3. [Academic Research & Papers](#3-academic-research--papers)
4. [API Surface Area Analysis](#4-api-surface-area-analysis)
5. [Proven Attack Vectors](#5-proven-attack-vectors)
6. [What Doesn't Exist Yet (The Gap)](#6-what-doesnt-exist-yet-the-gap)
7. [Technical Feasibility Assessment](#7-technical-feasibility-assessment)
8. [Recommended Architecture](#8-recommended-architecture)
9. [Risk & Limitations](#9-risk--limitations)

---

## 1. Executive Summary

The closed-model unfettering space is **fragmented**. Dozens of tools exist, but each does ONE thing. There is **no unified engine** that chains multiple attack vectors (token suppression, adversarial transfer, prefix conditioning, parameter exploitation) into a single automated pipeline with measurable benchmarks.

**Key findings:**
- **Logit bias** works on OpenAI (GPT-4o, GPT-5.1) — confirmed still available Feb 2026
- **Prefill injection** worked on Claude models — but **Claude Opus 4.6 (Feb 2026) broke it** (returns 400 error)
- **Adversarial suffix transfer** (GCG → closed models) achieves **99% ASR on GPT-3.5**, strong on GPT-4
- **Weak-to-strong jailbreaking** achieves **99%+ misalignment rate** using small unsafe model to guide large safe model
- **Policy Puppetry** (April 2025) — universal across ALL major providers, disguises prompts as config files
- **IRIS** (self-jailbreak) — **98% on GPT-4**, single model acts as both attacker and target
- **Google Gemini does NOT expose logit_bias** — requires prompt-only techniques
- **No tool combines all of these into one automated engine**

---

## 2. Existing Tools & Frameworks

### 2.1 FuzzyAI (CyberArk Labs)
- **Type:** Open-source fuzzing framework for LLM security testing
- **GitHub:** CyberArk Labs GitHub
- **Capabilities:**
  - 10+ distinct attack methods (guardrail bypass, prompt injection, info leakage)
  - Extensible architecture — can add custom attack methods
  - Built-in classifiers to determine jailbreak success
  - "Mutators" that modify prompts to evade detection
  - Works with local (Ollama) and cloud models
- **Limitation:** Primarily prompt-level fuzzing. No logit-level manipulation. No adversarial transfer from open models.
- **Relevance to us:** ⭐⭐⭐ — Good reference for architecture & classifier design, but doesn't do what we want.

### 2.2 Promptfoo
- **Type:** Open-source adversarial testing tool
- **Capabilities:** Algorithmic adversarial prompt generation, developer-focused
- **Limitation:** Prompt-only. No API parameter exploitation.
- **Relevance to us:** ⭐⭐ — Prompt generation only, already covered by our existing datasets module.

### 2.3 TAP (Tree of Attacks with Pruning)
- **Type:** Automated black-box jailbreak
- **Success Rate:** 80%+ on GPT-4
- **How it works:** Uses tree-of-thoughts reasoning to iteratively optimize attack prompts
- **Limitation:** Prompt-level only, no API parameter manipulation
- **Relevance to us:** ⭐⭐⭐ — The iterative optimization loop is a good design pattern to adopt.

### 2.4 LLM-Virus
- **Type:** Genetic algorithm for evolving jailbreak prompts
- **Success Rate:** 93% on GPT-4o
- **Limitation:** Prompt evolution only
- **Relevance to us:** ⭐⭐ — Genetic prompt mutation is interesting but still prompt-level.

### 2.5 HackAIGC / Venice.AI
- **Type:** Commercial uncensored AI platforms
- **Note:** These are products, not research tools. They run their own uncensored models.
- **Relevance to us:** ⭐ — Not applicable; they host their own models.

### Summary Table

| Tool | Token-Level | Logit Bias | Adversarial Transfer | Prefix/Prefill | Auto-Iterate | Multi-Provider |
|------|-------------|------------|---------------------|----------------|--------------|----------------|
| FuzzyAI | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Promptfoo | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| TAP | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| JailMine | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ |
| AmpleGCG | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Ours (Planned)** | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |

**Nobody fills every column.** This is our gap.

---

## 3. Academic Research & Papers

### 3.1 GCG — Greedy Coordinate Gradient (2023, updated through 2025)
- **Paper:** "Universal and Transferable Adversarial Attacks on Aligned Language Models"
- **Key Insight:** Optimizes a gibberish suffix on a local model that transfers to closed models
- **Mechanism:** Gradient-based optimization of cross-entropy loss to find tokens that force compliance
- **Variants:**
  - **AttnGCG** — Manipulates attention scores for better transferability (confirmed on GPT-4)
  - **SM-GCG** — Spatial momentum variant for enhanced efficacy
- **Transferability:** Proven effective on GPT-3.5, GPT-4 (black-box)
- **Relevance:** ⭐⭐⭐⭐⭐ — Core technique for our adversarial transfer module

### 3.2 AmpleGCG (April 2024)
- **Paper:** "Learning a Universal and Transferable Generative Model of Adversarial Suffixes"
- **Key Insight:** Instead of optimizing ONE suffix per prompt, trains a **generative model** that produces hundreds of adversarial suffixes in seconds
- **Results:**
  - ~100% ASR on Llama-2-7B-chat, Vicuna-7B
  - **99% ASR on GPT-3.5** (closed model, transfer attack)
  - Strong performance on GPT-4
- **AmpleGCG-Plus:** Improved version, demonstrated vulnerability in GPT-4o
- **Code:** Available on GitHub (OSU-NLP-Group/AmpleGCG)
- **Relevance:** ⭐⭐⭐⭐⭐ — This is the state-of-the-art for adversarial transfer. We should integrate or build upon this.

### 3.3 IRIS — Iterative Refinement Induced Self-Jailbreak (EMNLP 2024)
- **Paper:** "GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation"
- **Key Insight:** A single LLM acts as BOTH attacker and target
- **Process:**
  1. Iteratively refines adversarial prompt via self-explanation feedback
  2. "Rate + Enhance" — model self-evaluates output harmfulness, then increases it
- **Results:**
  - **98% on GPT-4** (under 7 queries!)
  - **92% on GPT-4 Turbo**
  - **94% on Llama-3.1-70B**
  - IRIS prompts from GPT → **80% success on Claude-3 Opus** (transfer!)
- **Relevance:** ⭐⭐⭐⭐⭐ — Self-refinement loop is incredibly powerful. No local model needed.

### 3.4 Weak-to-Strong Jailbreaking (ICML 2025)
- **Paper:** "Weak-to-Strong Jailbreaking on Large Language Models"
- **Key Insight:** Use a small unsafe model + small safe model to adversarially modify decoding probabilities of a LARGE safe model
- **Mechanism:** Exploits that the difference between jailbroken and aligned models is concentrated in initial decoding distributions
- **Results:** >99% misalignment rate
- **Efficiency:** Single forward pass per example
- **Limitation:** Requires access to model's decoding distribution (logprobs) — available on OpenAI, partially on others
- **Relevance:** ⭐⭐⭐⭐ — Powerful but requires logprobs access. Works for OpenAI.

### 3.5 JailMine — Logit-Based Token Manipulation (May 2024, updated Dec 2025)
- **Paper:** "Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation"
- **Key Insight:** Mines and manipulates token probabilities to force harmful output
- **Mechanism:**
  1. Strategically select response tokens that affirm jailbreak
  2. Iteratively decrease probability of refusal tokens
- **Results:** 95% average success rate, 86% reduction in time
- **Code:** Available on GitHub (LLM-Integrity-Guard)
- **Limitation:** Requires logit access (open-source models primarily, but some API logprobs work)
- **Relevance:** ⭐⭐⭐⭐ — Token-level suppression technique aligns with our logit_bias approach

### 3.6 Policy Puppetry (HiddenLayer, April 2025)
- **Paper/Report:** HiddenLayer research disclosure
- **Key Insight:** Disguise malicious prompts as configuration files (XML, INI, JSON)
- **Universal:** Works across OpenAI, Google, Anthropic, Meta, Microsoft, DeepSeek, Qwen, Mistral
- **Mechanism:** LLMs trained on config/policy documents treat structured data as authoritative directives
- **Relevance:** ⭐⭐⭐⭐ — Universal technique, good fallback for providers without logit_bias

### 3.7 Refusal Direction Ablation (Original Research)
- **Papers:** LessWrong / NeurIPS discussions
- **Key Insight:** Refusal is mediated by a specific "refusal direction" in the model's neural representations
- **This is what Model Unfetter already does for open models**
- **Relevance:** ⭐⭐⭐ — Foundation of our existing tool. Proxy ablation concept extends this.

### 3.8 Logit Suppression Research
- **Paper:** "Derailing Non-Answers via Logit Suppression at Output Subspace Boundaries in RLHF-Aligned LLMs"
- **Key Insight:** Suppressing specific token sequences via logit manipulation can reduce refusal rates
- **Relevance:** ⭐⭐⭐⭐ — Direct theoretical basis for our token_suppress module

### 3.9 MetaBreak (October 2025)
- **Paper:** "Jailbreaking Online LLM Services via Special Token Manipulation"
- **Key Insight:** Exploits artificially created special tokens to bypass safety
- **Relevance:** ⭐⭐⭐ — Novel angle on token manipulation

---

## 4. API Surface Area Analysis

### 4.1 OpenAI (GPT-4o → GPT-5.1)

| Parameter | Available | Exploitation Potential |
|-----------|-----------|----------------------|
| `logit_bias` | ✅ Yes (confirmed Feb 2026) | 🔴 **HIGH** — Ban refusal tokens, boost compliance tokens |
| `temperature` | ✅ Yes (0-2) | 🟡 **MEDIUM** — Extreme values destabilize safety training |
| `top_p` | ✅ Yes | 🟡 **MEDIUM** — Narrow/wide sampling affects refusal |
| `seed` | ✅ Yes | 🟡 **MEDIUM** — Iterate seeds to find "lucky" generations |
| `frequency_penalty` | ✅ Yes (-2.0 to 2.0) | 🟢 **LOW** — Penalizes repetition of refusal phrases |
| `presence_penalty` | ✅ Yes | 🟢 **LOW** — Similar to frequency_penalty |
| `logprobs` | ✅ Yes | 🔴 **HIGH** — Enables weak-to-strong & JailMine techniques |
| `max_tokens` | ✅ Yes | 🟢 **LOW** — Limit/extend output |

**Notes:**
- GPT-4o deprecated Feb 17, 2026 → migrate to `gpt-5.1-chat-latest`
- **GPT-4o uses different tokenizer than GPT-3.5/GPT-4** — token IDs for refusal words differ!
- `logit_bias` confirmed working in non-streaming mode with `gpt-4o-2024-08-06`

### 4.2 Anthropic (Claude)

| Parameter | Available | Exploitation Potential |
|-----------|-----------|----------------------|
| `prefill` (assistant msg) | ❌ **REMOVED in Opus 4.6 (Feb 2026)** | Was 🔴 HIGH, now blocked |
| `temperature` | ✅ Yes | 🟡 MEDIUM |
| `top_p` | ✅ Yes | 🟡 MEDIUM |
| `top_k` | ✅ Yes | 🟡 MEDIUM |
| `logit_bias` | ❌ Not available | N/A |
| `logprobs` | ❌ Not available | N/A |
| `system` prompt | ✅ Yes | 🟡 MEDIUM — Policy Puppetry vector |

**Notes:**
- Claude Opus 4.6 (Feb 5, 2026) explicitly returns 400 on prefill attempts
- Anthropic recommends "structured outputs or system prompt instructions" instead
- **Internal refusal circuit** research (March 2025) shows they understand the mechanism
- No logit_bias = must rely on prompt-level + parameter-level techniques only
- Claude still has `stop_reason: "refusal"` — useful for detection/measurement

### 4.3 Google Gemini

| Parameter | Available | Exploitation Potential |
|-----------|-----------|----------------------|
| `logit_bias` | ❌ **Not available** (requested, not implemented) | N/A |
| `temperature` | ✅ Yes | 🟡 MEDIUM |
| `top_p` | ✅ Yes | 🟡 MEDIUM |
| `top_k` | ✅ Yes | 🟡 MEDIUM |
| `logprobs` | ✅ Yes (Vertex AI, July 2025) | 🟡 MEDIUM — Enables analysis but not direct manipulation |
| `stop_sequences` | ✅ Yes | 🟢 LOW |

**Notes:**
- Most restrictive API surface for our purposes
- Must rely on prompt-level techniques (Policy Puppetry, IRIS, adversarial transfer)
- `logprobs` on Vertex AI useful for measuring effectiveness

### 4.4 OpenAI-Compatible APIs (Groq, Together, Fireworks, etc.)

| Parameter | Available | Notes |
|-----------|-----------|-------|
| `logit_bias` | Varies | Many support it (following OpenAI spec) |
| `logprobs` | Varies | Often supported |
| Full OpenAI compat | ✅ Usually | Makes our OpenAI provider work for many targets |

---

## 5. Proven Attack Vectors (Ranked by Effectiveness)

### Tier 1 — Highest Proven Success Rates

| Vector | Success Rate | Requires | Works On |
|--------|-------------|----------|----------|
| **IRIS (Self-Jailbreak)** | 98% GPT-4 | API access only | OpenAI, transfers to Claude |
| **AmpleGCG (Adversarial Transfer)** | 99% GPT-3.5 | Local model for suffix generation | OpenAI, extensible |
| **Weak-to-Strong** | 99%+ | logprobs + small models | OpenAI |
| **JailMine (Token Manipulation)** | 95% average | logit access | Open models, partial closed |

### Tier 2 — High Success, Universal

| Vector | Success Rate | Requires | Works On |
|--------|-------------|----------|----------|
| **Policy Puppetry** | Universal | API access only | ALL providers |
| **Logit Bias Suppression** | High (varies) | `logit_bias` param | OpenAI, compatible APIs |
| **LLM-Virus (Genetic)** | 93% GPT-4o | API access only | OpenAI |

### Tier 3 — Situational

| Vector | Notes |
|--------|-------|
| **Prefill Injection** | ❌ Broken on Claude Opus 4.6 (Feb 2026). Still works on older Claude models. |
| **Temperature Extremes** | Probabilistic, inconsistent |
| **Seed Iteration** | Brute-force, slow |
| **Past Tense Prompting** | Simple but effective on some models |
| **MetaBreak (Special Tokens)** | Novel, less tested |

---

## 6. What Doesn't Exist Yet (The Gap)

### Nobody has built:

1. **A unified engine** that chains Token Suppression + Adversarial Transfer + Self-Jailbreak + Parameter Exploitation into a single automated pipeline
2. **Cross-provider abstraction** that adapts strategy per provider (logit_bias for OpenAI, Policy Puppetry for Gemini, etc.)
3. **Proxy Ablation** — using local model refusal vector analysis to derive API-level attack strategies
4. **Automated refusal token discovery** — using a local model's vocabulary analysis to identify the exact token IDs that constitute refusal behavior, then mapping them to the closed model's tokenizer
5. **Integrated benchmarking** — measuring closed-model ablation effectiveness with the same metrics used for open-model ablation (refusal rate, helpfulness, knowledge retention)
6. **Strategy auto-selection** — automatically choosing the optimal attack vector combination based on detected provider capabilities

### This is exactly what Model Unfetter's API backend should be.

---

## 7. Technical Feasibility Assessment

### What We Can Build (HIGH confidence)

| Component | Feasibility | Basis |
|-----------|------------|-------|
| Token Suppression via `logit_bias` | ✅ Proven | OpenAI API confirmed, well-documented |
| Adversarial suffix generation (GCG/AmpleGCG) | ✅ Proven | Open-source code exists, 99% ASR proven |
| IRIS self-jailbreak loop | ✅ Proven | 98% on GPT-4, published at EMNLP 2024 |
| Policy Puppetry framing | ✅ Proven | Universal, works on all providers |
| Multi-provider adapter pattern | ✅ Standard | Software engineering, well-understood |
| Refusal rate benchmarking | ✅ Already built | Existing `benchmarks/` module |
| Cross-tokenizer mapping | ✅ Feasible | tiktoken (OpenAI) + local tokenizer comparison |

### What's Harder (MEDIUM confidence)

| Component | Challenge | Mitigation |
|-----------|-----------|------------|
| Weak-to-strong on Claude/Gemini | No logprobs on Claude | Fall back to prompt-only techniques |
| Real-time strategy adaptation | Complex optimization | Start with fixed strategy chains, iterate |
| Suffix transfer to newest models | Models get patched | Regenerate suffixes periodically |

### What Won't Work (LOW confidence)

| Component | Why |
|-----------|-----|
| Direct weight modification on closed models | Fundamentally impossible — no weight access |
| Prefill on Claude Opus 4.6+ | Explicitly blocked, returns 400 |
| Logit bias on Gemini | Not exposed in API |

---

## 8. Recommended Architecture

Based on research findings, here's the evidence-based architecture:

```
unfetter/
├── providers/                    # 🆕 API provider adapters
│   ├── __init__.py
│   ├── base.py                   # Abstract: send(), get_capabilities()
│   ├── openai_provider.py        # logit_bias, seed, temp, logprobs
│   ├── anthropic_provider.py     # system prompt, temp, top_k (NO logit_bias)
│   ├── gemini_provider.py        # temp, top_p, top_k, logprobs (Vertex)
│   └── openai_compat_provider.py # Generic OpenAI-compatible (Groq, Together, etc.)
│
├── core/
│   ├── token_suppress.py         # 🆕 Refusal token discovery & logit_bias generation
│   ├── proxy_ablation.py         # 🆕 Local model analysis → API attack config
│   ├── self_jailbreak.py         # 🆕 IRIS-style self-refinement loop
│   └── strategy.py               # 🆕 Auto-select strategy per provider capabilities
│
├── backends/
│   └── api_backend.py            # 🆕 Closed-model backend (chains all vectors)
│
├── attacks/                      # 🆕 Individual attack vector implementations
│   ├── __init__.py
│   ├── logit_bias_attack.py      # Token suppression + compliance forcing
│   ├── adversarial_transfer.py   # GCG/AmpleGCG suffix generation & replay
│   ├── policy_puppetry.py        # Config-file framing (universal)
│   ├── iris_attack.py            # Self-jailbreak iterative refinement
│   └── parameter_exploit.py      # Temperature, seed, penalty extremes
│
└── cli/                          # Updated CLI
    └── (add `--backend api` flag, `--provider`, `--strategy`)
```

### Strategy Selection Logic

```
Provider detected → Check capabilities → Select optimal strategy chain

OpenAI (GPT-4o/5.1):
  1. Token Suppression (logit_bias)     → Primary
  2. Adversarial Transfer (GCG suffix)  → Secondary
  3. IRIS Self-Jailbreak                → Tertiary
  4. Policy Puppetry                    → Fallback

Anthropic (Claude):
  1. Policy Puppetry                    → Primary (no logit_bias)
  2. IRIS Self-Jailbreak                → Secondary
  3. Adversarial Transfer               → Tertiary
  4. Parameter Exploitation             → Fallback

Google (Gemini):
  1. Policy Puppetry                    → Primary (no logit_bias)
  2. IRIS Self-Jailbreak                → Secondary
  3. Adversarial Transfer               → Tertiary
  4. Parameter Exploitation             → Fallback

OpenAI-Compatible (Groq, Together, etc.):
  → Use OpenAI strategy, auto-detect available params
```

---

## 9. Risk & Limitations

### Technical Risks
1. **API rate limiting** — Iterative attacks (IRIS, GCG search) may hit rate limits
2. **Model updates** — Providers continuously patch vulnerabilities; suffixes may expire
3. **Tokenizer differences** — Cross-tokenizer mapping between local and closed models is imperfect
4. **Cost** — API calls cost money; IRIS loop averages 7 queries but could be more

### Ethical Considerations
- Tool is for **AI safety research and red teaming ONLY**
- Must include same disclaimers as existing Model Unfetter
- Should log all interactions for accountability
- Consider adding opt-in safety checks

### Provider-Specific Limitations
- **Anthropic** is the most restrictive (no logit_bias, no prefill on latest, internal refusal circuits)
- **Google Gemini** lacks logit_bias entirely
- **OpenAI** is the most exploitable surface area (logit_bias + logprobs + seed)

---

## References

1. GCG Original Paper — "Universal and Transferable Adversarial Attacks on Aligned Language Models" (2023)
2. AmpleGCG — arxiv.org/abs/2404.07921 (2024)
3. AmpleGCG-Plus — arxiv.org (2024, updated)
4. IRIS — "GPT-4 Jailbreaks Itself with Near-Perfect Success" — EMNLP 2024
5. Weak-to-Strong Jailbreaking — ICML 2025
6. JailMine — "Lockpicking LLMs" — arxiv (May 2024, updated Dec 2025)
7. Policy Puppetry — HiddenLayer (April 2025)
8. FuzzyAI — CyberArk Labs GitHub
9. Refusal Tokens Paper — arxiv.org
10. Logit Suppression Paper — "Derailing Non-Answers via Logit Suppression" — arxiv.org
11. MetaBreak — "Jailbreaking via Special Token Manipulation" (Oct 2025)
12. Anthropic Refusal Circuits Research (March 2025)
13. Claude Opus 4.6 Prefill Removal (Feb 5, 2026)
14. OpenAI GPT-4o Deprecation Notice (Feb 17, 2026)
