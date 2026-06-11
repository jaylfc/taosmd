---
name: research-report
description: Maintain docs/research-report.md, the living technical report of taOSmd findings. Use when recording experiment results, adding negative results, pre-registering experiments, or revising the report. Enforces methodology discipline; never lets a number in without provenance.
---

# taOSmd Research Report Discipline

The report at `docs/research-report.md` is a living lab-report-meets-review-paper. It is public, written in Jay's voice, and held to a higher standard than marketing copy. These rules are hard gates, not suggestions.

## Structure (fixed top-level sections, never reordered)

0. Index: the navigation layer (see Index system below).
1. Abstract: current one-paragraph state of results.
2. Methodology: judge protocol, datasets, metrics, hardware tiers, and the measured reasons behind each choice.
3. Results: evidence tables; every number traceable.
4. Negative results: first-class section, never pruned. A lever that failed is a finding.
5. Reproducibility: exact commands, model versions, dataset provenance.
6. Ongoing work (pre-registered): experiments declared BEFORE results exist.
7. Revision log: append-only, stamped.

## Index system (agents navigate by this; keep it perfect)

Directly under the title, before the Abstract, the report carries two things:

1. A table of contents linking every section and subsection by GitHub anchor.
2. The finding index: one table row per finding, experiment, or negative result, with columns ID | one line | section anchor | status | provenance. IDs are stable and never reused: F-NNN for findings, E-NNN for experiments (E-001 and E-002 are the surprisal and hallucination-rate kill-shots), N-NNN for negative results. An experiment keeps its E id when its outcome moves to Results (becoming also an F row) or Negative results (an N row); the index cross-links them.

Rules: every new entry anywhere in the report gets an index row in the same edit; an agent must be able to answer "what do we know about X" by reading the index alone and jumping once; grep-friendly one-liners (name the lever, the number, the dataset). The index is part of every revision-log diff.

## Hard gates

- **Number provenance**: every score, latency, or rate cites its source inline: the results file name (benchmarks/results/...), the commit that recorded it, or both. A number without provenance does not enter the report.
- **Pre-registration**: an experiment enters section 6 with its design and KILL CRITERION written down before it runs. When results land, it moves to section 3 or 4 with the original criterion quoted. Editing a kill criterion after results exist is falsification; never do it.
- **Negative results are mandatory**: if an experiment, lever, or idea failed, it goes in section 4 with the same rigor as a win. Removing a negative result requires superseding evidence, recorded in the revision log.
- **Judge honesty**: any accuracy number states its judge model and whether the judge shares a model family with the generator. Vendor-graded or same-family numbers are labeled as such.
- **Three-number reporting**: accuracy claims are accompanied by latency and context-token cost where measured. Never collapse to a single score.
- **Revision log**: every change to the report appends a dated entry (what changed, why). History is never rewritten.

## Voice (public, as Jay)

Plain human prose. No em dashes. No AI attribution or AI-pattern phrasing. Product name is taOSmd in prose; lowercase taosmd only for the package, CLI, and repo slug. Honest hedging beats confident vagueness: "subset-200, expect +/-0.02 noise" is the house style.

## Update procedure

1. Read the current report and the source of the new finding (results file or benchmarks.md entry).
2. Place the finding (Results or Negative results), with provenance.
3. If it resolves a pre-registered experiment, move and quote the original kill criterion verbatim.
4. Update the Abstract only if the headline state changed.
5. Update the index: new rows for new entries, status flips for moved ones.
6. Append the revision-log entry.
7. Run the voice checks (no em dashes, casing, no AI tells) before committing.

Concept credit: pre-registration and negative-result enforcement patterns surveyed from community academic-writing skills (Imbad0202/academic-research-skills, lingzhi227/agent-research-skills); this skill is an independent minimal implementation for a living repo document.
