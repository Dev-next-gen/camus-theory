# The CAMUS Theory

**Emergent Temporal Cognition in Language Models**

*A theoretical framework and its empirical validation through graft-based adapters.*

**Author:** Leo CAMUS — NextDev Lab's
**Dates:** April 2026 (Part I), April 2026 (Part II)

| Part | Title | DOI |
|---|---|---|
| **I**  | Emergent Temporal Cognition in Language Models *(theory)* | [10.5281/zenodo.19509846](https://doi.org/10.5281/zenodo.19509846) |
| **II** | Graft-Based Emergence of Temporal Cognition in Frozen Language Models *(empirical)* | [10.5281/zenodo.19557893](https://doi.org/10.5281/zenodo.19557893) |

---

## Part I — Theoretical Foundation

Current Large Language Models (LLMs) based on the Transformer architecture process information in a fundamentally atemporal manner. The **CAMUS Theory** proposes that by integrating a 5-component temporal vector **T**(t) = (δ_prev, δ_session, τ_inf, ω_context, ρ_rate) into the training representation of each token, a Transformer model will develop *emergent temporal cognition* structurally analogous to biological neural mechanisms.

Five falsifiable predictions are presented, covering attention-head specialization, Weber's Law, rhythm adaptation, inference-difficulty encoding and temporal-continuity preference.

```
T(t) = (δ_prev, δ_session, τ_inf, ω_context, ρ_rate)

δ_prev    : inter-token delay
δ_session : session elapsed time
τ_inf     : model inference time (reflexive)
ω_context : temporal context window
ρ_rate    : token generation rate
```

## Part II — Empirical Validation

Part II closes the loop with a non-invasive alternative to full pre-training: a **graft methodology** that endows any pretrained decoder with first-class temporal cognition by training only a small **TemporalAdapter** (under 0.6 % of base parameters) injected at mid-depth via a forward pre-hook. The base LLM is fully frozen.

Validated on **TinyLlama-1.1B** and **Qwen2.5-14B**, with an extension to **Qwen2.5-Coder-32B**. Key empirical findings:

- Linear decodability of log-time plateaus at **R² ≈ 0.9** from 1 B parameters onward — temporal representability is a minimal cognitive primitive, not a scaling-sensitive capacity.
- The temporal signal lives in a **~5-dimensional subspace invariant to base width**, structurally mirroring distributed time-cell coding in the mammalian hippocampus.
- Increasing base scale refines **temporal pragmatics**, producing emergent registers acknowledging elapsed time at long δ. The 32B Coder extension additionally shows a **code/prose register switch conditioned on δ**.
- A runtime modulation scalar **α** restores generative fluency without retraining (sweet spot α ≈ 0.2–0.3).
- Full pipeline reproduces on a single AMD MI300X in under 30 minutes at ≈ \$0.83.

## Repository layout

```
.
├── camus_theory.tex / .pdf           # Part I (English)
├── camus_theory_fr.tex / .pdf        # Part I (French)
├── camus_temporal.tex / .pdf         # Part II (English)
├── camus_temporal_fr.tex / .pdf      # Part II (French)
└── implementation/
    ├── adapter/     TemporalAdapter module
    ├── training/    graft_mi300x.py + dataset mix builder
    ├── inference/   Multi-GPU REPL with runtime δ/α control
    ├── probes/      Five evaluation probes
    └── checkpoints/ Download pointers for trained adapters (GitHub Releases)
```

Adapter weights are published as [GitHub Release assets](https://github.com/Dev-next-gen/camus-theory/releases) of this repository.

## Quickstart

See [`implementation/README.md`](implementation/README.md) for the full training / inference / probing guide.

## Citation

```bibtex
@misc{camus2026theory,
  title  = {The CAMUS Theory: Emergent Temporal Cognition in Language Models},
  author = {CAMUS, Leo},
  year   = {2026},
  doi    = {10.5281/zenodo.19509846},
  publisher = {Zenodo}
}

@misc{camus2026graft,
  title  = {The CAMUS Theory: Graft-Based Emergence of Temporal Cognition in Frozen Language Models},
  author = {CAMUS, Leo},
  year   = {2026},
  doi    = {10.5281/zenodo.19557893},
  publisher = {Zenodo}
}
```

## License

© 2026 Leo CAMUS.

- Part I: Creative Commons Attribution-NoDerivatives 4.0 International (CC-BY-ND 4.0)
- Part II and implementation code: Creative Commons Attribution 4.0 International (CC-BY 4.0)
