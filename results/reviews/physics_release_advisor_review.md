# Advisor review: staged private PRL-writing QLoRA payload

Date: 2026-11-13
Scope: CPU-only, read-only review of `artifacts/releases/Qwen3-4B-tuneLLM-PRL-Writing-QLoRA`.
Files reviewed: `README.md`, `REPRODUCE.md`, `human_review/README.md`,
`human_review/packet.json`, `evaluation.json`, `selection.json`, `provenance.json`,
and `reproduction.zip` (privacy grep only).
Not done: no GPU use, network calls, edits to the payload or source, or reading of human ratings (none exist).

## Verdict

No blocking problems. The private upload can proceed after the root's two disclosure fixes (below) and a restage with a new manifest.
Neither fix changes any result, selection or score.

## Verified

- **Review key commitment.** `outputs/physics_release_review_key.json` has SHA-256
  `62ff9a50e1922ba8d77dac9e064bb485ad79c509bef1480bde889ee96b6df924`. This equals
  `provenance.json` `human_review_key_sha256`. The key is not in the payload.
- **Blinded packet.**
  - `human_review/packet.json` has 60 pairs: 20 title, 20 abstract and 20 introduction.
  - Each pair contains only `sample_id`, `task`, `evidence`, `A`, `B` and empty rating fields.
  - Ratings are "pending; never inferred".
  - The 30-pair local HTML uses a separate key, which is recorded in `evaluation.json` `local_human_review`.
- **Privacy.**
  - No staged file or `reproduction.zip` contains a `/home` path or a personal email.
  - The only grep matches are ordinary tokenizer vocabulary entries in `vocab.json`/`tokenizer.json`, such as `gmail`.
  - Metadata paths use `<PROJECT_ROOT>`/`<HOME>` placeholders.
- **Consistency.**
  - The card, `evaluation.json` and `selection.json` agree.
  - Selection: r128, LR 5e-5, epoch 1 (step 69). It was selected by the registered rule because the r8 and r32 NLL CIs against r128 exclude zero.
- **Paper NLL.** NF4 base 1.974938 → adapter 1.733950. BF16 base 1.929116 → control 1.713637.
- **Seed repeat.**
  - The seed-43 repeat scores 1.733926.
  - The repeat-minus-primary difference is −2.4227e-5, 95% CI [−0.00110, 0.00100], so the result is stable across the two seeds.
  - The repeat-minus-base difference is −0.241012, with a CI that excludes zero.
- **Math regression.**
  - Numeric GSM8K fell from 117 to 111 out of 128, and the card states this prominently.
  - Paired result: 9 broken, 3 fixed, exact McNemar p = 0.146, exploratory and unadjusted.
  - Strict-format counts (97 → 106) are reported separately.
- **Honest scope.**
  - No human writing-quality claim; automatic flags are described as diagnostics.
  - The exported epoch-1 exposure (522,380 tokens) is distinguished from the full-run budget (1,567,140 tokens).
  - Limits cover small subsets, the absence of a PRB control, author overlap, AI-reviewed theory labels, the rank-8 budget boundary, and the BF16 control not being independently optimized.
- **Reproduction.** `REPRODUCE.md` pins the NF4 repository commit separately from the upstream Qwen commit. It forbids re-quantization and live re-querying, and requires protocols to be committed before training.

## Disclosure gaps (the root is fixing these; they do not block)

1. **Writing rows need a prompt condition and truncation counts.**
   - The adapter's 3 automatic flags can only be interpreted against the comparator conditions. The adapter was run zero-shot; `nf4_base_fewshot` used two training-paper examples.
   - Each writing row should state its prompt condition.
   - Report final writing truncation counts as well (adapter 1 vs few-shot base 0), because truncation was a gated quantity.
2. **State what the 60-pair packet compares.**
   - `human_review/README.md` and the card should say that the packet compares the selected NF4 r128 epoch-1 adapter (zero-shot) with the NF4 base given two examples.
   - Blinding hides only which side is A or B, not which conditions are compared. The per-pair key must stay excluded.

## Optional note

General MC rose from 85 to 93 out of 128, with exact McNemar p ≈ 0.057, unadjusted and exploratory. The card correctly makes no improvement claim; keep it that way.

## Not requested and not proposed

No change to selection, gates or epoch. No inference of human ratings.

## Final verification of restage 03 (2026-11-13): fixes accepted

Read-only check of the two fixes and the new key commitment only.

1. **Writing rows.** All five writing rows in `README.md` now state the prompt condition and the generation-cap count:
   - nf4_adapter: 60/60 nonempty, 3 flags, 1 cap, zero-shot.
   - nf4_base_fewshot: 60/60 nonempty, 0 flags, 0 caps, two-example.
   - nf4_base, bf16_base and bf16_control: zero-shot, with 0, 0 and 0 caps.

   Accepted.
2. **Packet comparison.** `human_review/README.md` and `packet.json` `comparison` both identify the selected NF4 rank-128 epoch-1 adapter (zero-shot) against the saved NF4 base with two training-paper examples. A/B identities are hidden per pair, and `packet.json` contains no mapping. Accepted.
3. **Key commitment.**
   - `provenance.json` commits `32392d8809a33f2156d6c27412055ea9f84ec4af087ffd2e35c527cdad12b926`, which equals the SHA-256 of local `outputs/physics_release_review_key.json`.
   - The key is excluded from the payload.
   - The off-machine backup's existence was not checked; that was outside this scope.
4. **Privacy re-grep.** No `/home` path or personal email appears in the restaged payload, excluding tokenizer vocabulary files. Clean.

Verdict: both reported gaps are fixed and no blocking issues remain. The payload is acceptable for the authorized private upload. Human ratings remain pending.
