# Option C: feeding new C++ code to the ASTNN-Longformer checkpoint

Explored 16 October 2026. **Verdict: working. The converter reproduces the dataset's trees exactly: 6/6 on the pairs used to derive its rules, and 42 further held-out dataset records (34 of them non-trivial) matched exactly by unseen submissions (section 5).**

## 1. The problem

The checkpoint only understands code in the dataset's preprocessed format: lists of statement subtrees whose nodes are integer IDs into `node_w2v_128`. The conversion script used by the dataset authors was never released (the `pkt-attn/pkt-attn` repository contains only models, `codeforces.zip` and `tag_table.csv`). The tree-sitter pipeline in `pkt_backup/content/pkt/preprocessing/cpp_ast.py` produces different node names (`translation_unit`, `binary_expression` …), so its output cannot be fed to this checkpoint.

## 2. What the original format is (reverse-engineered)

Decoding the dataset's token IDs with the `node_w2v_128` vocabulary (32,972 tokens; Word2Vec skip-gram, 128-d, window 5, min_count 3) turns the files back into readable trees:

```
FUNCTION_DECL
  NAME → main
  TYPE_KIND → TYPE → int
  COMPOUND_STMT
    DECL_STMT → VAR_DECL → TYPE_KIND → TYPE → long long ; DECLARATOR → NAME → t
    WHILE_STMT
      UNARY_OPERATOR_POST_-- → DECL_REF_EXPR → t
      ...
```

Findings:

| Evidence | Conclusion |
|---|---|
| Node kinds are libclang `CursorKind` names (`DECL_REF_EXPR`, `VAR_DECL`, `COMPOUND_STMT`, `CXX_METHOD` …) with operator suffixes (`BINARY_OPERATOR_+`, `UNARY_OPERATOR_POST_++`) | The parser was **libclang (Clang)** |
| `cin`/`cout`/`scanf` lines vanish; `string s` appears as `int s`; `if` conditions that use `string`/`vector` disappear | Parsed **without standard headers**. Library calls are unresolved and dropped. |
| Calls to undeclared functions keep their arguments (`max(a,b)` → `a b`), while statements using variables of unknown type (`string`, `map`) are dropped; `while`/`for` with such a condition vanish, `if` keeps an empty condition | Exact match only with **libclang 12/13** (10/11 drop the arguments; ≥ 14 keep the invalid statements) |
| Roots per submission follow ASTNN's `get_blocks` (`astnn_original/prepare_data.py`): FUNCTION_DECL / IF / FOR / WHILE / DO roots followed by their non-compound children; struct methods contribute their nested blocks only | Block splitting is a C++ variant of ASTNN's |
| 4,190 empty `COMPOUND_STMT`s occur, plus 535 genuine `NULL_STMT`s | The converter does not pad empty blocks |
| Rare identifiers appear as ID 32972 | Out-of-vocabulary tokens map to `len(vocab)` |
| `FUNCTION_CALL`, `nullptr`, `static_cast`, lambdas, range-for and `struct` nodes all exist in the vocabulary | These shapes are supported |

## 3. Prototype converter

`content/pkt/preprocessing_live/cpp_to_astnn.py`:

- **Input → output:** C++ source text in, the dataset's integer tree list out (`source_to_blocks` then `to_ids`).
- **Runtime:** libclang 13 and pcpp (`pip install libclang==13.0.0 pcpp`), run with `-nostdinc -nostdinc++`.
  - On Windows the wheel bundles `libclang.dll`.
- **Rules beyond plain libclang:** `#define` macros are expanded first (operator spellings are read from tokens); single-statement bodies of `if`/`for`/`while`/`do` are wrapped in `COMPOUND_STMT` (except `else if`); functional casts carry `TYPE_KIND`.
- The statistics in section 4 were produced with an earlier version of the converter.

## 4. Statistical validation (2,000 real C++ submissions from the PDKT data)

| Check | Converted real code | Codeforces dataset |
|---|---|---|
| Conversion failures | 0 / 2,000 | n/a |
| Mean statement blocks per submission | 26.0 | 23.8 |
| Root kind share: IF / FOR / FUNCTION_DECL / DECL_STMT | 12.0% / 10.2% / 7.3% / 8.2% | 12.4% / 10.7% / 8.3% / 10.5% |
| Token OOV rate (mapped to unknown) | 1.8%, almost all new identifiers | 0.02% (vocab built on the dataset) |
| Structural kinds emitted but absent from vocab | `sizeof` (`CXX_UNARY_EXPR`), in-function `using` | n/a |
| Code embedding cosine to the dataset centre | 0.768 | 0.797 (random trees: 0.106) |
| Domain classifier, dataset vs converted (AUC) | 0.776 | n/a |
| End to end: 30 converted submissions → HF checkpoint → 37 mastery scores | runs (0.006 – 0.978) | n/a |

Interpretation:
- **The format is right.** Converted code lands in the same region of the model's code-embedding space as dataset code, far from random input.
- **The remaining distinguishability has two possible causes.** One is that PDKT is a different, easier online judge than Codeforces. The other is residual converter differences. Only real Codeforces pairs can separate them.

## 5. Exact validation on real Codeforces pairs

Codeforces no longer shows other users' source code (verified: even after an accepted solution the page shows "N/A"), and its robots.txt disallows automated fetching. The source was instead recovered from the public **open-r1/codeforces-submissions** dataset (12.6M submissions with code), searched by submission ID (`validation/fetch_from_hf.py`).

- 35 dataset records were aligned to Codeforces submission IDs through the official API; **6 of them are present** in the public dataset (2 learners, problems 1342A, 1342B, 1355A–D).
- Before the fixes: 0/6 exact, mean token F1 0.84 (with libclang 18) / 0.96 (with libclang 10).
- Comparing versions showed libclang 12/13 match the dataset's error-recovery behaviour; three rules (above) closed the rest.
- **After: 6/6 exact matches** (every block, every token, `validation/validate_converter.py`).

Caveat: the three rules were derived on these same 6 files, so a held-out check followed.

### Held-out check

All 91,915 C++ submissions (IDs 54M–81M) to the 23 identified problems were taken from the public dataset (`validation/extract_candidates.py`), converted with the frozen converter, and compared with the dataset's 256 records on those problems (`validation/heldout_build_index.py`, `validation/heldout_match.py`). About 1,900 inputs had not finished converting when the run was stopped.

| Result | Count |
|---|---|
| Dataset records with an exact tree match | 48 / 256 |
| … not among the 6 used to derive the rules | **42** |
| … of those, non-trivial trees (≥ 100 nodes) | 34 (each matched by exactly one submission) |
| Learners covered | 14 of 55 |

- An exact match of a 100+ node tree cannot happen by chance, so **34 independent exact reproductions** confirm the converter on unseen code. The median matched tree has 214 nodes.
- **Why not all 256:** the public dataset holds only part of Codeforces. Of the 35 records aligned through the API, 6 (17%) were present. The match rate here (48/256 = 19%) is at the same level, which suggests that most of the unmatched records are simply absent from the public dataset, not converted wrongly. This could not be checked directly.
- The matches also link 48 dataset records to their real Codeforces submissions (`validation/heldout_record_matches.json`).

## 5b. Live input in the demo (usability)

- **Problem choice:** an identified Codeforces problem (e.g. 1342A), any other problem with its topics picked by hand, or a catalogue problem by internal ID. The diagnosis uses the topics, code and verdict; the problem ID only stops the same problem being recommended again. Custom problems use ID 7152, which is outside the KSAP catalogue.
- **Guards:** code over 60,000 characters is rejected, and conversion gives up after 20 s with a clear message. About 2% of 92k public submissions converted very slowly.
- **Results table:** the change table always shows the added problem's own topics first.
- **Links:** recommendations name and link the 23 identified Codeforces problems.

## 5c. How much does the code itself matter? (honesty check)

`scripts/code_effect_ablation.py`, 204 validation attempts. Each attempt was changed in one way and re-scored (mean absolute change over the 37 concept scores):

| Change to one attempt | last-step | smoothed | smoothed + record |
|---|---|---|---|
| Adding the attempt at all | 0.159 | 0.026 | 0.025 |
| Verdict flipped, same code | 0.160 | **0.038** | 0.038 |
| Tags replaced, same code | 0.120 | 0.017 | 0.018 |
| **Code replaced** by another learner's (same verdict, same tags) | 0.105 | **0.011** | 0.011 |
| Code replaced by a near-empty tree (one token) | 0.102 | 0.011 | 0.011 |

The code does carry information (about 40% of an attempt's effect), but the verdict and the problem's tags move the diagnosis more.

The concept names come from the problem's tags; nothing in the pipeline reads topics out of the code. Because the dataset's preprocessing has no standard headers, library calls (`sort`, `map`, `set`, `cin`/`cout`, `push_back`, …), and loops or conditions built on library types, never reach the model.

The demo therefore shows, for each added submission:
- **What the model saw:** functions, recursion, loops written vs loops visible, conditions, array accesses, bit and modulo operations, and the library calls that were invisible to it.
- **The table:** before / after / total change, plus **"from the code itself"** (the same attempt re-scored with a minimal program), with the problem's own topics listed first.

## 5d. The verdict: self-reported, or estimated (no self-grading by default)

The training loader (`load_data_astnn._append_history`) feeds step *t* (tags, code, verdict) and trains the model to predict step *t+1*'s verdict from *t+1*'s tags.

So **the Objective 2 model (AUC 0.836) never sees the code of the attempt it predicts**. It forecasts success from the history before the code is written. As a result:
- The model cannot judge pasted code. Its forecast for a new attempt does not depend on that attempt's code.

The demo therefore defaults to **"Verdict not known"**:
- It shows the model's own forecast, `sigmoid(mean concept logit over the attempt's tags)`, exactly the training objective (`live_input.next_attempt_probability`).
- It updates the diagnosis as the forecast-weighted average of the accepted and failed outcomes.
- "Accepted" and "Wrong" remain available when the verdict is actually known (e.g. from the judge).

Only real judging, compiling and running tests, can establish correctness.

## 6. Files

| Path | Purpose |
|---|---|
| `content/pkt/preprocessing_live/cpp_to_astnn.py` | Prototype converter |
| `content/pkt/preprocessing_live/fidelity.py` | Statistical comparison against the dataset |
| `content/pkt/preprocessing_live/e2e.py` | End to end through the checkpoint, plus the embedding check |
