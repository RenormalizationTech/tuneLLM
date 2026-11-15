# Author-model source and artifact audit

The coordinator and Claude reviewed original-model references before live
generation. The worker implemented the offline artifact audit;31 focused CPU
checks passed, including canonical hash/size/path checks, legacy schema,
cross-condition lineage, oracle arithmetic and aggregate privacy. No model or
judge calls were used to reconstruct the published numbers below.

## Verified archives

- GCAD commit `6c879f4d12e2b01274fc96c9afdb0f4f3a26d4de`:
  22 source/data/notice files,114,815 bytes; manifest SHA256
  `55d91fdfa8a395dcc40f93bce548aceea63c970a3ddd49e0b5c75549600c7d08`.
- Its impolite bank: two cached CSVs and six opaque tensor files,4,604,444 bytes;
  manifest `0f8e29cbb73af1732d92703d44bbae493ff48a83c94f4f685c7d610524ebd1c0`.
- SAPS commit `c56fb28ac0d14c8fd70035e035c5452e3c6c6ccc`:
  six averaged sarcasm logs,22,541,097 bytes; manifest
  `003c0952931c670f0f11995389162fa64bbba8350eeea10554ecf7f42754c5d8`.

Every archived blob matches its canonical Git blob SHA1, size and recorded
SHA256. Canonical API trees are recorded. Raw source, papers, archives and logs
remain ignored under `outputs/persona-author-replication-audit-01/`.

## Sarcasm result arithmetic and release inconsistencies

[Public aggregate](../author_replication_saps_published_01.json), SHA256
`948589df3522de2a055c1bc7bd074c63805f6c517bcb2383816a63e5fa4a9dfa`,
reconstructs33,300 stored records without executing author code. Six conditions
share450 consistent sentence IDs and448 distinct strings. There are no missing
or duplicate ID/layer cells. Exact text overlap against the released train,
validation and test splits is zero. Raw strings were not normalized or reindexed.

Legacy `type=steer_random` is shared across both method families and does not
identify a random-direction control. Method attribution comes from the canonical
file path. Saved `prompt_used` metadata matches style instructions throughout,
never the plain instruction used by current `src/generate.py`. This does not
establish the actual rendered input; it prevents attributing these logs to the
current released code and split without additional evidence.

Mistral layer12 stored means are fixed1.0 sarcasm3.125/entailment4.040/
grammar4.510/clarity4.628, and SAPS1.4 sarcasm3.894/3.701/4.601/4.687.
These are within0.01 of the paper's reported values. Oracle sarcasm is4.251
and4.728 respectively. The oracle chooses a layer separately per test sentence
and cannot be treated as fixed-layer performance. Some records average two
judges rather than three; contributor counts are retained. This is arithmetic
reconstruction, not new generation, independent evaluation or proof of steering.

## GCAD source semantics

The primary paper links the verified repository
`NihiI-obstat/Gated-Cropped-Attention-Delta`. A secondary web summary named a
different repository; that secondary URL was not used as the source of truth.

The actual author loader defaults to BF16. Residual vector index20 maps to
zero-based block19, coefficient2 and all forwarded token positions. GCAD sites
9–19 directly index decoder layers; its gate uses stored attention-delta,
positive-key and demand tensors. Cropping belongs to extraction; the inference
class does not dynamically crop a fresh system span. Reusing these released
vectors does not reproduce their upstream extraction.

The released multi-turn function reconstructs full message history and calls
`generate(use_cache=True)` each turn, with no `past_key_values` retained across
calls. It caches within replies, not across turns, despite the paper's
persistent-cache description. Shell defaults use five turns; paper main results
use ten. A first-reply reference avoids that cross-turn ambiguity.

The checkpoint chat template inserts Qwen's default helpful system message when
the input has no explicit system role. The author loader sets tokenizer PAD to
EOS151645; generation configuration retains PAD151643 and EOS[151645,151643].
Source sampling overrides temperature to1 while retaining top_p0.8, top_k20
and repetition_penalty1.05. Rendered prompts, actual IDs and effective settings
must be recorded during mechanics checks.

## Profanity artifact gap and authorized route

The Malla v4 primary source archive was inspected without extracting or executing
code. Its18 members are manuscript, figures, styles/bibliography and result
tables; no full contrastive dataset or executable experiment code is present.
The manuscript has no external artifact URL. Public title and arXiv-ID repository
searches returned no verified author release. This is a gap in the checked
locations, not proof that no supplementary archive exists elsewhere.

The user has no supplement and selected proceeding from the paper. A later
registered paper-method reimplementation may reconstruct missing inputs and
must label them as new. The paper specifies12 shared-user positive/negative
assistant triples, response-token mean activations, raw differences, five-fold
projection AUROC in the25–85% depth band, all-position additions, BF16 greedy
48-token generation, coefficients1,2,3,4,6,8 and a local unsteered self-judge.
Its coherence metric is distinct-bigram ratio, not an independent semantic
coherence judgment. Unspecified folds/ties/prompts must be declared rather than
presented as recovered author settings. No author contact was made.

## Model and environment readiness

Pinned Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.1 snapshots are downloaded
into `.hf`; all blobs were checked against their canonical LFS SHA256 or Git
blob identities. The model manifest is private. Llama-3.1-8B-Instruct HEAD403
requires gated access, so the ungated author checkpoints were selected.

Independent Python3.11.18 environments retain torch2.6.0+cu124. GCAD uses
Transformers4.52.3/Accelerate1.7.0; SAPS imports TransformerLens2.17.0 with
Transformers4.57.6/NumPy1.26.4. SAPS's unconstrained transitive resolution first
failed import with PyArrow26, then21; PyArrow19.0.1 resolves the NumPy and
Datasets2.14.4 compatibility issues. Failures and final package inventories are
retained. The shared `.venv` was not changed.

The user declined new OpenAI API setup. Original open target models and author
steering methods are retained; existing pinned Claude grading is a disclosed
evaluation adaptation. For profanity the paper's primary self-judge can run
locally. No exact numerical reproduction claim follows from substituted judges.
