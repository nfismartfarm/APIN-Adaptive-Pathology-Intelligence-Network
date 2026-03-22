# evaluation.md — Plant Disease Detection: Kerala
# Complete evaluation specification. Defines what "done" means for this project.
# Covers all three evaluation tiers, every metric, every report, every plot,
# every acceptance threshold, and every interpretation rule.
# Claude Code reads this before writing any evaluation script.
# Last updated: [DATE]

---

## HOW TO USE THIS FILE

This file is the contract between the model and deployment readiness. It answers:
- What must be measured at each stage of the project
- What each metric means in plain language
- What constitutes passing and failing
- What output files and reports must be produced
- How to interpret results that are borderline
- What to do when a tier fails

CLAUDE.md specifies HOW to build the evaluation scripts.
This file specifies WHAT those scripts must produce and WHY each output matters.

Read this file in full before writing 07_evaluate_validation.py,
08_evaluate_tier2_plantdoc.py, 09_evaluate_tier3_kerala.py, or
10_evaluate_local_test.py.

---

## SECTION 1: THREE-TIER EVALUATION OVERVIEW

This project uses three evaluation tiers, each measuring a different type of
generalisation. Passing tier-1 does not imply passing tier-2. Passing tier-2
does not imply passing tier-3. Each tier answers a distinct question.

```
TIER 1 — Standard Validation
  Question: Is the model learning?
  Data:     15% of training pool, same distribution as training
  When:     After every training epoch (early stopping) + full report after Phase 2
  Script:   training/07_evaluate_validation.py
  Pass:     Macro F1 >= 0.50

TIER 2 — PlantDoc Wild Test
  Question: Does the model generalise to independently collected real-world data?
  Data:     Entire PlantDoc dataset (never seen during training)
  When:     ONCE, after Phase 2 + calibration are final. Never again.
  Script:   training/08_evaluate_tier2_plantdoc.py
  Pass:     Macro F1 >= 0.55 on mappable classes

TIER 3 — Kerala Field Test
  Question: Does the model work for actual Kerala farmers?
  Data:     50+ verified, manually labelled Kerala field images
  When:     Once, when 50+ Kerala images are collected
  Script:   training/09_evaluate_tier3_kerala.py
  Pass:     Per-class accuracy >= 0.70 on classes with >= 5 Kerala images

SUPPLEMENTARY — Local Test Set
  Question: How does the model perform on the held-out 15% test split?
  Data:     15% of training pool (split before training, never touched until now)
  When:     ONCE, after tier-2 evaluation
  Script:   training/10_evaluate_local_test.py
  Pass:     Macro F1 >= 0.50 (same threshold as validation)
```

**Evaluation sequence and locks:**
```
Phase 1 training → val F1 monitored per epoch
Phase 2 training → val F1 monitored per epoch
Calibration      → val ECE measured
                   ↓
             [TRAINING LOCKED]
                   ↓
Step 11: 07_evaluate_validation.py  — full validation report
Step 12: Server smoke test
Step 13: 08_evaluate_tier2_plantdoc.py  — tier-2 (RUN ONCE)
                   ↓
          [TIER-2 LOCK — no further model changes]
                   ↓
         10_evaluate_local_test.py  — local test (run after tier-2)
                   ↓
         09_evaluate_tier3_kerala.py  — tier-3 (when 50+ Kerala images ready)
                   ↓
          [PROJECT DEPLOYMENT-VALIDATED]
```

---

## SECTION 2: METRICS REFERENCE

Every metric used in this project is defined below. All evaluation scripts use
these definitions. No script may invent its own metric definition.

---

### 2.1 Macro F1 Score (primary disease metric)

```
FORMULA    : macro_f1 = mean(per_class_f1[j] for j in 0..9)
             per_class_f1[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
             where precision[j] = TP[j] / (TP[j] + FP[j])
             and   recall[j]    = TP[j] / (TP[j] + FN[j])

THRESHOLD  : For disease predictions: sigmoid probability > DISEASE_THRESH (0.50)
             counts as a positive prediction.

WHY MACRO  : Macro averaging gives equal weight to every class regardless of how
             many images it has. brassica_clubroot with 50 test images gets the
             same weight as okra_yvmv with 500 test images. This is correct for
             this project because a farmer with clubroot needs the system to work
             just as much as a farmer with YVMV. Weighted F1 (weighted by class
             frequency) would hide poor performance on thin classes.

ZERO DIVISION HANDLING : If a class has no positive predictions (precision denominator
             is 0) or no positive ground truth (recall denominator is 0):
             Use zero_division=0 in sklearn.metrics.f1_score.
             This returns 0.0 for the class rather than raising an error.
             A class with F1=0.0 due to zero_division is flagged in reports.

SKLEARN CALL:
  from sklearn.metrics import f1_score
  macro_f1     = f1_score(d_true, d_binary, average='macro', zero_division=0)
  per_class_f1 = f1_score(d_true, d_binary, average=None,    zero_division=0)
```

---

### 2.2 Per-Class F1 Score

```
FORMULA    : Computed by sklearn.metrics.f1_score with average=None.
             Returns an array of 10 values, one per class in CLASS_NAMES order.

WHY NEEDED : Macro F1 hides which specific classes are weak. Per-class F1 shows
             exactly where the model fails. A macro F1 of 0.55 with
             brassica_clubroot=0.10 and everything else=0.60 is very different
             from all classes at 0.55 — the first case needs targeted improvement,
             the second is a general performance issue.

REPORTING  : Every evaluation report must include a per-class F1 table.
             Classes with F1 < 0.40 must be flagged explicitly in the report
             with a "← LOW" marker or equivalent.
```

---

### 2.3 Crop Classification Accuracy

```
FORMULA    : crop_acc = correct_crop_predictions / total_predictions
             where correct = (argmax(crop_logits) == crop_label)
             Using sklearn.metrics.accuracy_score(c_true, c_preds)

WHY NEEDED : The crop classifier gates disease predictions — if crop classification
             is wrong, disease predictions for the wrong crop are shown to the farmer.
             Crop accuracy must be high (> 0.90) for the FiLM conditioning to be useful.

THRESHOLD  : Expected > 0.90. Below 0.85 indicates a systematic problem with crop
             feature learning. Below 0.70 indicates the crop classifier is unreliable
             and FiLM conditioning may be hurting rather than helping.

REPORTING  : Included in every evaluation report.
```

---

### 2.4 Expected Calibration Error (ECE)

```
FORMULA    : ECE = sum over bins of (|accuracy_in_bin - confidence_in_bin| * fraction_in_bin)
             Uses 15 equal-width bins from 0.0 to 1.0.
             bin_accuracy  = mean(true_labels in bin)
             bin_confidence = mean(predicted_probs in bin)
             bin_fraction  = count_in_bin / total_predictions

WHAT IT MEASURES : Calibration. A perfectly calibrated model that says "70% confident"
             is correct 70% of the time. ECE measures the average miscalibration.
             ECE=0.0 is perfect calibration. ECE=0.10 means predictions are off
             by 10 percentage points on average.

WHY IT MATTERS : Farmers make treatment decisions based on the confidence score.
             A model that says "95% confident" when the actual accuracy at that
             confidence level is 60% will mislead farmers into acting on uncertain
             diagnoses. ECE < 0.10 means the confidence scores are trustworthy enough
             for decision support.

THRESHOLD  : ECE < 0.10 after temperature scaling. Above 0.10 means calibration
             failed and confidence scores should not be shown to farmers as-is.

COMPUTATION :
  def compute_ece(probs, labels, n_bins=15):
      probs  = np.array(probs).flatten()
      labels = np.array(labels).flatten()
      bins   = np.linspace(0.0, 1.0, n_bins + 1)
      ece    = 0.0
      n      = len(probs)
      for lo, hi in zip(bins[:-1], bins[1:]):
          mask = (probs >= lo) & (probs < hi)
          if not mask.any(): continue
          acc  = labels[mask].mean()
          conf = probs[mask].mean()
          ece += np.abs(acc - conf) * mask.sum() / n
      return float(ece)

REPORTING  : ECE before and after calibration must be reported in:
             - 06_calibrate.py output
             - 07_evaluate_validation.py report
             - 10_evaluate_local_test.py report
```

---

### 2.5 Temperature Values (T_disease, T_crop, T_severity)

```
WHAT THEY ARE : Scalar divisors applied to logits before sigmoid/softmax.
               logits_calibrated = logits / T
               T > 1.0 makes the model less confident (sharpens distribution).
               T < 1.0 makes the model more confident (flattens distribution).
               T = 1.0 means no calibration effect.

EXPECTED RANGE : T_disease: 0.8 to 2.5 (most models overconfident, T > 1)
                T_crop:    0.8 to 1.5
                T_severity: 0.8 to 2.0

WARNING FLAGS  :
  T < 0.5: model is severely underconfident — check loss function and training
  T > 4.0: model is severely overconfident — check if training converged properly
  T = 1.0 exactly: LBFGS may have failed to converge — check calibration script

REPORTING  : All three T values must be included in the calibration output and
             in the validation report.
```

---

### 2.6 Confusion Matrix

```
WHAT IT IS : A NUM_CLASSES × NUM_CLASSES matrix where entry [i,j] = number of
             images with true class i that were predicted as class j.

ADAPTATION FOR MULTI-LABEL : Since disease labels are multi-hot, the confusion
             matrix here uses argmax of the multi-hot labels as the "primary class"
             for display purposes only. This is a simplification for human-readable
             reporting — it does not affect any other metric computation.

HOW TO COMPUTE :
  d_true_argmax = d_true.argmax(axis=1)    # index of true class
  d_pred_argmax = d_binary.argmax(axis=1)  # index of highest-confidence prediction
  cm = confusion_matrix(d_true_argmax, d_pred_argmax)

WHAT TO LOOK FOR:
  - Diagonal entries should be large (correct predictions)
  - Off-diagonal entries in the same crop group (okra rows with brassica columns
    or vice versa) indicate crop classification failures
  - High confusion between okra_yvmv and okra_enation is expected (both are
    begomovirus diseases with similar visual presentation) — tolerable
  - High confusion between any brassica and any okra disease is a systematic
    failure — indicates crop classifier is unreliable

REPORTING  : Confusion matrix included in every evaluation report as a text table.
```

---

### 2.7 Per-Source Breakdown (validation and local test only)

```
WHAT IT IS : F1 score computed separately for each source_dataset in the split.
             Answers: does the model perform equally well on sabbir_okra images
             vs kareem_cabbage images?

WHY NEEDED : A model might perform well on average but fail systematically on
             one source due to domain-specific label noise, lighting conditions,
             or image quality differences. Per-source breakdown identifies these
             before deployment.

HOW TO COMPUTE :
  from collections import defaultdict
  source_preds = defaultdict(list)
  source_true  = defaultdict(list)
  # during evaluation loop:
  for record, pred, true in zip(records, preds, trues):
      source = record['source_dataset']
      source_preds[source].append(pred)
      source_true[source].append(true)
  # after loop:
  for source in source_preds:
      src_f1 = f1_score(source_true[source], source_preds[source],
                        average='macro', zero_division=0)

WARNING FLAGS :
  Any source with macro F1 < 0.30 while overall macro F1 > 0.50 indicates
  domain-specific failure. This source's images are systematically harder for
  the model — investigate label quality and image characteristics.

REPORTING  : Per-source table included in 07_evaluate_validation.py and
             10_evaluate_local_test.py reports.
```

---

### 2.8 Uncertainty Distribution

```
WHAT IT IS : Distribution of uncertainty scores (MC Dropout std) across the
             validation/test set. Plotted as a histogram.

WHY NEEDED : MC Dropout uncertainty should correlate with actual model error.
             High uncertainty should correspond to images where the model is
             wrong. Low uncertainty should correspond to images where the model
             is correct. If uncertainty does not correlate with error, the
             MC Dropout mechanism is not providing useful signal.

HOW TO COMPUTE :
  uncertainties = mc_disease.std(dim=0).mean(dim=1)  # per-image uncertainty
  # Compare to: was the top prediction correct?
  # Plot histogram of uncertainties for correct vs incorrect predictions.

WHAT TO LOOK FOR:
  - Correct predictions should cluster at low uncertainty (< 0.15)
  - Incorrect predictions should show higher uncertainty (> 0.20)
  - If both distributions overlap completely, MC Dropout is not calibrated well
    — consider increasing MC_PASSES or DROPOUT_P

REPORTING  : Uncertainty histogram saved to reports/plots/uncertainty_dist_{timestamp}.png
             Description of distribution pattern in the validation report text.
```

---

### 2.9 Calibration Curve (Reliability Diagram)

```
WHAT IT IS : A plot of mean predicted probability (x-axis) vs actual fraction
             positive (y-axis) across probability bins. A perfectly calibrated
             model follows the diagonal y=x.

HOW TO COMPUTE :
  bins = np.linspace(0, 1, 11)  # 10 equal-width bins
  mean_pred = []
  frac_pos  = []
  for lo, hi in zip(bins[:-1], bins[1:]):
      mask = (probs >= lo) & (probs < hi)
      if mask.sum() > 0:
          mean_pred.append(probs[mask].mean())
          frac_pos.append(labels[mask].mean())
  # plot mean_pred vs frac_pos, and the diagonal for reference

WHAT TO LOOK FOR:
  - Before calibration: most models are overconfident (curve below diagonal)
    meaning "when the model says 0.9, it's actually right only 0.7 of the time"
  - After temperature scaling: curve should be close to diagonal
  - If the curve is S-shaped: temperature scaling is insufficient, consider
    isotonic regression calibration instead
  - ECE < 0.10 corresponds to the curve staying within ~0.10 of the diagonal

REPORTING  : Calibration curve saved to reports/plots/calibration_curve_{timestamp}.png
             One curve before calibration (T=1.0) and one after (T=T_disease).
```

---

## SECTION 3: TIER-1 VALIDATION EVALUATION (07_evaluate_validation.py)

---

### 3.1 When to run

Run exactly once after Phase 2 training and temperature calibration are complete.
This is Step 11 in the pipeline. Do not run this script during training — the
per-epoch val/macro_f1 logged to wandb is sufficient during training.

---

### 3.2 What the script must produce

**Console output (printed during run):**
```
Loading model from models/best_model.pt...
Temperature T_disease=X.XXX loaded from models/temperature.pt
Running inference on X val images...
--------------------------------------------------
Validation Results:
  Macro F1 (disease) : X.XXXX
  Crop Accuracy      : X.XXXX
  ECE                : X.XXXX
  Val images         : XXXX
--------------------------------------------------
Per-class F1:
  okra_yvmv            : X.XXXX
  okra_powdery_mildew  : X.XXXX
  ...
--------------------------------------------------
Validation report written: reports/validation_report_YYYYMMDD_HHMMSS.md
```

**Files produced:**
```
reports/validation_report_{timestamp}.md    — full text report (REQUIRED)
reports/plots/confusion_matrix_{timestamp}.png  — confusion matrix heatmap
reports/plots/calibration_curve_{timestamp}.png — reliability diagram (optional but recommended)
reports/plots/per_class_f1_{timestamp}.png  — bar chart of per-class F1 (optional)
reports/plots/uncertainty_dist_{timestamp}.png  — uncertainty histogram (optional)
```

---

### 3.3 Required content in validation_report_{timestamp}.md

The markdown report must contain ALL of the following sections in this order:

```markdown
# Validation Report
Generated: {ISO timestamp}
Model: models/best_model.pt
Val images: {count}
Temperature T_disease: {value}  T_crop: {value}  T_severity: {value}

## Summary
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Macro F1 (disease) | {value} | >= 0.50 | PASS/FAIL |
| Crop Accuracy | {value} | >= 0.90 | PASS/FAIL |
| ECE (calibration) | {value} | < 0.10 | PASS/FAIL |

## Per-Class F1 (disease head)
| Class | F1 | Status |
|-------|-----|--------|
| okra_yvmv | {value} | OK / LOW |
... (all 10 classes)

## Per-Source F1 Breakdown
| Source | Images | Macro F1 |
|--------|--------|----------|
| sabbir_okra | {count} | {value} |
... (all sources in val split)

## Confusion Matrix
(argmax of multi-hot labels, for display only)
{text representation of NUM_CLASSES × NUM_CLASSES matrix}

## Calibration
ECE before calibration (T=1.0): {value}
ECE after calibration (T={T_disease}): {value}
Improvement: {before - after}

## Uncertainty Analysis
Mean uncertainty (correct predictions):   {value}
Mean uncertainty (incorrect predictions): {value}
OOD flagging rate: {fraction of val images flagged as OOD}

## Acceptance Decision
{ONE of the following:}

✓ PASS — All primary metrics meet thresholds. Model is ready for tier-2 evaluation.

or

✗ FAIL — {metric name} = {value} does not meet threshold {threshold}.
Recommended action: {specific action based on which metric failed}

## Notes
{Any observations about specific classes, unusual patterns, or caveats}
```

---

### 3.4 Acceptance thresholds and failure actions

**If Macro F1 < 0.50:**
```
Severity: HIGH — model is not learning adequately.
Likely causes:
  1. Label mapping errors in source_map.csv — verify manually
  2. pos_weight not working — check compute_multilabel_pos_weights output
  3. Class imbalance too severe — check class_counts.csv for extreme imbalances
  4. Feature cache built with wrong transform — delete and rebuild with eval_transform
Actions:
  - Do NOT proceed to tier-2 until this is resolved
  - Diagnose using troubleshooting.md T-019
  - Document root cause in context.md Section 7
```

**If Crop Accuracy < 0.85:**
```
Severity: HIGH — FiLM conditioning is unreliable, disease predictions may be wrong crop.
Likely cause: Training data crop imbalance or label noise.
Actions:
  - Check crop distribution in class_counts.csv
  - Verify SOURCE_LABEL_OVERRIDES correctly separates okra and brassica labels
  - Consider disabling FiLM conditioning temporarily (set film_gamma output to 1.0)
  - Document in decisions.md
```

**If ECE > 0.10 after calibration:**
```
Severity: MEDIUM — confidence scores shown to farmers are not trustworthy.
Likely cause: LBFGS failed to converge, or model is extremely overconfident.
Actions:
  - Check T_disease value: if still near 1.5 (TEMP_INIT), LBFGS did not move
  - Increase LBFGS max_iter from 50 to 100 in 06_calibrate.py
  - Try different TEMP_INIT value (start at 2.0 or 3.0)
  - If ECE is 0.10-0.15, proceed but add a disclaimer to the farmer-facing UI
  - Document actual ECE in context.md Section 4.3
```

**If per-class F1 < 0.30 for any class:**
```
Severity: MEDIUM — specific disease cannot be reliably detected.
Actions:
  - Count training images for that class in class_counts.csv
  - If < 150 images: add synthetic images or find more data
  - If >= 150 images: the class may be visually similar to another (check confusion matrix)
  - Note in report which classes are below 0.30 and why
```

---

## SECTION 4: TIER-2 PLANTDOC EVALUATION (08_evaluate_tier2_plantdoc.py)

---

### 4.1 The one-time rule — CRITICAL

Tier-2 evaluation runs EXACTLY ONCE. After it runs, no model changes are permitted.
This is not a guideline — it is a data integrity rule.

If tier-2 is run multiple times with the same model:
- The result is still valid (same model, same data)
- Multiple reports in reports/ is fine

If tier-2 is run and the model is then changed and tier-2 is run again:
- The second result is invalid — you have now optimised for PlantDoc
- PlantDoc loses its status as an independent test set
- This invalidates ALL tier-2 claims about generalisation

**Lock enforcement:**
After 08_evaluate_tier2_plantdoc.py completes successfully:
1. Record the run date in context.md Section 5.2 under "Tier-2 lock status: LOCKED"
2. Do not change: BEST_MODEL, temperature.pt, any model hyperparameters
3. Do not add PlantDoc images to the training pool

---

### 4.2 What classes are evaluated

Only the four PlantDoc-mappable brassica classes:
```
brassica_black_rot     (PlantDoc: Cabbage__Black_Rot)
brassica_downy_mildew  (PlantDoc: Cabbage__Downy_Mildew)
brassica_alternaria    (PlantDoc: Cabbage__Alternaria_leaf_spot)
brassica_healthy       (PlantDoc: Cabbage__healthy)
```

The five okra classes and brassica_clubroot have NO PlantDoc equivalent and are
NOT evaluated at tier-2. This is not a limitation of the evaluation — it reflects
PlantDoc's actual content. The macro F1 for tier-2 is computed only over these 4 classes.

---

### 4.3 Temperature scaling at tier-2

Temperature scaling (T_disease from models/temperature.pt) MUST be applied at
tier-2. The same inference pipeline used in production must be used for evaluation.
Evaluating without temperature scaling tests a different model than what farmers use.

---

### 4.4 What the script must produce

**Console output:**
```
TIER-2 PLANTDOC EVALUATION
Loading model: models/best_model.pt
T_disease: X.XXX (from temperature.pt)
PlantDoc images: XXXX (split='plantdoc' from source_map.csv)
Evaluating on 4 mappable classes only...
--------------------------------------------------
Tier-2 Results (mappable classes):
  Macro F1: X.XXXX  (threshold: 0.55)
  Per-class:
    brassica_black_rot    : X.XXXX
    brassica_downy_mildew : X.XXXX
    brassica_alternaria   : X.XXXX
    brassica_healthy      : X.XXXX
--------------------------------------------------
Result: PASS / FAIL
Tier-2 report written: reports/tier2_plantdoc_{timestamp}.md
```

**Files produced:**
```
reports/tier2_plantdoc_{timestamp}.md    — full tier-2 report (REQUIRED)
reports/plots/tier2_confusion_{timestamp}.png — confusion matrix for 4 classes
reports/plots/tier2_per_class_{timestamp}.png — per-class F1 bar chart
```

---

### 4.5 Required content in tier2_plantdoc_{timestamp}.md

```markdown
# Tier-2 PlantDoc Evaluation Report
Generated: {ISO timestamp}
Model: models/best_model.pt
T_disease: {value}
PlantDoc images evaluated: {count}
Non-mappable PlantDoc classes discarded: {count}

## Results (4 mappable classes)
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Macro F1 (4 classes) | {value} | >= 0.55 | PASS/FAIL |

## Per-Class F1
| Class | Images | F1 | Status |
|-------|--------|-----|--------|
| brassica_black_rot    | {count} | {value} | OK / LOW |
| brassica_downy_mildew | {count} | {value} | OK / LOW |
| brassica_alternaria   | {count} | {value} | OK / LOW |
| brassica_healthy      | {count} | {value} | OK / LOW |

## Confusion Matrix (4 × 4)
{text or image reference}

## Tier-2 Decision
{ONE of the following:}

✓ PASS — Macro F1 {value} >= 0.55. Model generalises to independently
collected real-world data. Proceed to local test evaluation and deployment.

or

✗ FAIL — Macro F1 {value} < 0.55.
Gap analysis: [per-class breakdown of failures]
Recommended action: [see Section 4.6 below]

## Lock Status
Tier-2 evaluation run on: {date}
No further model changes permitted after this date.
```

---

### 4.6 Tier-2 failure actions

**If Macro F1 = 0.45-0.55 (near threshold):**
```
This is a borderline result. Options:
Option A — Accept and deploy with documented limitations:
  Write a gap analysis noting which classes fail and why (domain shift expected
  for brassica_alternaria given PlantDoc images are from different climates).
  Add a disclaimer to the farmer-facing UI for brassica diagnoses.
  Document in decisions.md.
Option B — Collect more diverse training data for failing classes and retrain.
  IMPORTANT: This means tier-2 has NOT yet been run (the first run is invalidated
  by training changes). The new trained model's tier-2 evaluation is the real one.
```

**If Macro F1 < 0.45 (clear failure):**
```
Do NOT deploy. Root causes to investigate:
  1. Domain shift from training-to-PlantDoc is too large
     → Add more diverse brassica disease images from different regions
  2. Temperature calibration overcorrects disease probabilities
     → Check T_disease: if > 3.0, reduce TEMP_INIT and recalibrate
  3. Specific class has near-zero recall
     → That class needs more training data or improved CLAHE parameters
     → Check the per-class F1 table to identify the bottleneck class
Document the failure in context.md and start a new training run.
```

**If one class fails but others pass (e.g. brassica_downy_mildew F1 = 0.20):**
```
The macro F1 will be pulled below 0.55 by one weak class. Options:
  1. Investigate why that specific class fails on PlantDoc
     → Download more diverse images for that class
     → Check if PlantDoc uses different symptom staging (early vs late infection)
  2. Accept with documented limitation: the system does not reliably detect
     brassica_downy_mildew in field conditions
     → Note this explicitly in the farmer UI for brassica downy mildew predictions
  3. Lower TIER2_MIN_F1 threshold (requires decision entry in decisions.md
     with strong justification)
```

---

## SECTION 5: LOCAL TEST SET EVALUATION (10_evaluate_local_test.py)

---

### 5.1 When to run

Run ONCE after tier-2 evaluation. The local 15% test split was held out during
training and must not be evaluated until the model is locked.

Unlike tier-2 (which uses externally collected data), the local test split comes
from the same training datasets. It measures in-distribution test performance —
useful for comparing against reported numbers from similar Kaggle competition work,
but not a measure of real-world deployment readiness.

---

### 5.2 What the script must produce

**Files produced:**
```
reports/local_test_report_{timestamp}.md  — full report (REQUIRED)
reports/plots/localtest_confusion_{timestamp}.png
reports/plots/localtest_per_class_{timestamp}.png
reports/plots/localtest_per_source_{timestamp}.png
```

---

### 5.3 Required content in local_test_report_{timestamp}.md

```markdown
# Local Test Set Evaluation Report
Generated: {ISO timestamp}
Model: models/best_model.pt
T_disease: {value}
Test images: {count}

## Summary
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Macro F1 (disease) | {value} | >= 0.50 | PASS/FAIL |
| Crop Accuracy | {value} | >= 0.90 | PASS/FAIL |
| ECE | {value} | < 0.10 | PASS/FAIL |

## Per-Class F1
| Class | F1 |
|-------|-----|
... (all 10 classes)

## Per-Source F1 Breakdown
| Source | Images | Macro F1 |
|--------|--------|----------|
... (all sources in test split)

## Comparison: Validation vs Local Test
| Metric | Validation | Local Test | Delta |
|--------|------------|------------|-------|
| Macro F1 | {val} | {test} | {diff} |
| Crop Acc | {val} | {test} | {diff} |
| ECE | {val} | {test} | {diff} |

## Notes on Generalisation Gap
{Commentary on the difference between val and test performance.
A gap > 0.05 in macro F1 suggests overfitting to the validation-set distribution.}
```

---

### 5.4 Interpreting the val vs test gap

```
Val F1 ≈ Test F1 (gap < 0.05):
  Good. The model generalises well within the training distribution.
  The validation set is a reliable proxy for in-distribution performance.

Val F1 > Test F1 by 0.05-0.10:
  Mild overfitting. The model was tuned (via early stopping) against the val set.
  Some overfitting to val distribution is expected and acceptable.
  No action required unless tier-2 also fails.

Val F1 > Test F1 by > 0.10:
  Significant overfitting. Possible causes:
  - EARLY_STOP_PAT was too short — the model stopped too early on val
  - The val set and test set come from different source distributions
    (check per-source breakdown for both)
  - Data leakage in the split (near-duplicate images in both splits)
  Note this in context.md. If retraining, consider increasing early stopping
  patience or using a different splitting strategy.

Test F1 > Val F1 by > 0.05:
  Unusual. Check that the test split is genuinely held-out (not accidentally
  included in training). Verify the stratified split code in 01_prepare_data.py.
```

---

## SECTION 6: TIER-3 KERALA FIELD EVALUATION (09_evaluate_tier3_kerala.py)

---

### 6.1 Minimum requirements before running

```
Required Kerala images: >= 50 (TIER3_MIN_IMGS = 50)
Required class coverage: >= 6 of 10 classes with images
Minimum per class for evaluation: >= 5 images (TIER3_MIN_CLS = 5)

How to add Kerala images:
  python tools/add_kerala_image.py --path image.jpg --class okra_yvmv

What counts as a valid Kerala image (see Section 2.3 of main CLAUDE.md):
  ✓ Photos taken in Kerala or South India with verified disease labels
  ✓ iNaturalist observations in Kerala GPS zone, manually disease-labelled
  ✓ TNAU or ICAR-IIHR images (Tamil Nadu, same climate zone)
  ✓ Farmer submissions verified manually

  ✗ Stable Diffusion synthetic images (do NOT count)
  ✗ iNaturalist observations without disease labels (domain_adapt, do NOT count)
  ✗ YouTube frames without disease labels (do NOT count)
```

---

### 6.2 What the script must produce

**Files produced:**
```
reports/tier3_kerala_{timestamp}.md        — full tier-3 report (REQUIRED)
reports/plots/tier3_per_class_{timestamp}.png  — per-class accuracy bar chart
```

---

### 6.3 Required content in tier3_kerala_{timestamp}.md

```markdown
# Tier-3 Kerala Field Evaluation Report
Generated: {ISO timestamp}
Model: models/best_model.pt
T_disease: {value}
Kerala images: {total count}
Classes with >= 5 images: {count} of 10

## Per-Class Results
| Class | Images | Accuracy | Threshold | Status |
|-------|--------|----------|-----------|--------|
| okra_yvmv | {count} | {value} | >= 0.70 | PASS/FAIL/SKIP |
... (all 10 classes — SKIP if < 5 images)

## Classes Not Evaluated (< 5 images)
{List classes with < 5 Kerala images and their current count}
{Instructions: add more images via tools/add_kerala_image.py}

## Overall Decision
{ONE of the following:}

✓ PASS — All evaluated classes meet the 0.70 accuracy threshold.
Project is DEPLOYMENT-VALIDATED for Kerala field conditions.
Evaluated {X} of 10 disease classes.
Classes not yet evaluated: {list} — collect more images to fully validate.

or

✗ FAIL — {class names} do not meet the 0.70 accuracy threshold.
[Failing class]: accuracy={value}, images={count}
Recommended action: [see Section 6.4]

## Performance vs Training Distribution
{Commentary comparing Kerala accuracy to validation F1 for each evaluated class.
A large gap (> 0.15) indicates domain shift specific to that class.}

## Kerala-Specific Observations
{Notes on image quality, lighting conditions, symptom presentation differences
from training data. Record anything that would help future data collection.}
```

---

### 6.4 Tier-3 failure actions

**If accuracy < 0.70 for a specific class:**
```
Step 1 — Check how many Kerala images exist for that class.
  If < 10: the estimate is unreliable. Collect more images before concluding failure.
  If >= 10: genuine performance gap.
Step 2 — Examine the failing images manually:
  - Are they unusually dark/bright (outside CLAHE correction range)?
  - Do they show a different disease stage than training data?
  - Are they co-infected images that training data did not prepare for?
Step 3 — If lighting is the issue: tune CLAHE parameters (clip_limit in apply_clahe).
  Test different values: 1.0, 2.0 (current), 3.0, 4.0
  Document the optimal value in decisions.md.
Step 4 — If stage/severity is the issue: collect more Kerala images for that
  specific symptom stage.
Step 5 — Accept with documented limitation if the failing class has < 15 Kerala
  images (estimate too unreliable for definitive failure claim).
```

**If overall accuracy < 0.50 across all classes:**
```
This indicates systematic Kerala domain shift — not a class-specific problem.
Causes:
  - Monsoon overcast lighting is not adequately corrected by current CLAHE
  - Local crop varieties look different from training data at the symptom level
Actions:
  1. Collect more Kerala images and check if a subset of high-quality images
     achieves > 0.70 (isolates whether it is a data quality problem)
  2. Re-examine CLAHE parameters for Kerala-specific lighting
  3. Consider adding real Kerala images (if collected) to training pool and retraining
     → This requires a completely new training run with a new test split
```

---

## SECTION 7: TRAINING-TIME MONITORING (per-epoch metrics)

These metrics are logged to wandb during training. They are NOT evaluation
outputs — they are training signals. They are listed here for completeness and
to specify what Claude Code must log to wandb.

---

### 7.1 Phase 1 — metrics logged every epoch

```python
wandb.log({
    'epoch'       : epoch,
    'train/loss'  : train_loss_avg,   # mean loss over all train batches
    'val/loss'    : val_loss_avg,     # mean loss over all val batches
    'val/macro_f1': val_f1,          # macro F1 using DISEASE_THRESH=0.50
    'val/crop_acc': crop_acc,        # crop classifier accuracy
    # Per-class F1 — one entry per class
    'val/f1_okra_yvmv'            : per_class_f1[0],
    'val/f1_okra_powdery_mildew'  : per_class_f1[1],
    'val/f1_okra_cercospora'      : per_class_f1[2],
    'val/f1_okra_enation'         : per_class_f1[3],
    'val/f1_okra_healthy'         : per_class_f1[4],
    'val/f1_brassica_black_rot'   : per_class_f1[5],
    'val/f1_brassica_downy_mildew': per_class_f1[6],
    'val/f1_brassica_alternaria'  : per_class_f1[7],
    'val/f1_brassica_clubroot'    : per_class_f1[8],
    'val/f1_brassica_healthy'     : per_class_f1[9],
})
```

---

### 7.2 Phase 2 — additional metrics logged every step

```python
# Per optimiser step (every GRAD_ACCUM_STEPS batches):
wandb.log({
    'train/grad_norm': grad_norm.item(),    # after unscaling, before clipping
    'train/lr'       : scheduler.get_last_lr()[0],  # current LR for first param group
})

# Per epoch (same as Phase 1 plus):
wandb.log({
    'epoch'       : epoch,
    'train/loss'  : epoch_loss / len(train_loader),
    'val/macro_f1': val_f1,
    'val/crop_acc': crop_acc,
    'val/ece'     : ece,
    **{f'val/f1_{cls}': f1 for cls, f1 in zip(CLASS_NAMES, per_class_f1)},
})
```

---

### 7.3 Training health signals to monitor in wandb

These are warning signs to watch during training. Investigating them early
prevents wasted compute on a run that has already gone wrong.

```
SIGNAL: train/loss increases after initially decreasing
MEANS : Overfitting OR LR too high in Phase 2 OR gradient exploding
CHECK : train/grad_norm — if > 10 consistently, LR is too high

SIGNAL: val/macro_f1 is 0.0 after epoch 3
MEANS : Model is predicting all negatives (pos_weight issue) OR label mapping error
CHECK : See troubleshooting.md T-019

SIGNAL: val/f1_brassica_clubroot is 0.0 for all epochs
MEANS : Clubroot images are not making it through the pipeline
CHECK : class_counts.csv — does clubroot appear in training split?
        Are there at least 100 clubroot training images?

SIGNAL: train/lr drops to near 0 very quickly (before epoch 3)
MEANS : OneCycleLR is misconfigured OR PHASE2_EPOCHS is set too low
CHECK : scheduler parameters in 05_train_phase2.py match ONE_CYCLE_* constants

SIGNAL: train/grad_norm is exactly GRAD_CLIP_NORM (1.0) every step
MEANS : Gradients are being clipped every single step — LR may be too high
        OR scaler.unscale_() is not being called before clip_grad_norm_
CHECK : Verify unscale_() call order (troubleshooting.md T-022)

SIGNAL: val/crop_acc drops suddenly in Phase 2 (e.g. from 0.95 to 0.60)
MEANS : Catastrophic forgetting — backbone unfreezing is destroying crop features
CHECK : Reduce fine_tune_at (unfreeze fewer backbone layers)
        Reduce PHASE2_BASE_LR
```

---

## SECTION 8: CALIBRATION EVALUATION (06_calibrate.py outputs)

---

### 8.1 What calibration must produce

**Console output:**
```
ECE before calibration: X.XXXX
Fitting T_disease via LBFGS on val set...
Fitting T_crop via LBFGS on val set...
Fitting T_severity via LBFGS on val set...
T_disease = X.XXXX
T_crop    = X.XXXX
T_severity = X.XXXX
ECE after calibration (T_disease): X.XXXX
Calibration improvement: X.XXXX → X.XXXX (delta = X.XXXX)
Saved: models/temperature.pt
```

**Files produced:**
```
models/temperature.pt  — dict with T_disease, T_crop, T_severity, ece_before, ece_after
```

**temperature.pt contents (required keys):**
```python
{
    'T_disease'  : float,   # temperature scalar for disease head
    'T_crop'     : float,   # temperature scalar for crop head
    'T_severity' : float,   # temperature scalar for severity head
    'ece_before' : float,   # ECE with T=1.0 (no calibration)
    'ece_after'  : float,   # ECE with T_disease applied
    'val_images' : int,     # number of val images used for calibration
}
```

---

### 8.2 Calibration acceptance criteria

```
ECE after < 0.10       : REQUIRED for deployment
ECE improvement        : ECE_after should be less than ECE_before. If not, LBFGS failed.
T_disease range        : 0.5 to 4.0. Outside this range: investigate.
T_disease ≈ TEMP_INIT  : If T_disease ≈ 1.5 (the init value), LBFGS did not move —
                          convergence failure. Increase max_iter or change TEMP_INIT.
T < 0.5                : Model is underconfident — unusual for a trained model. Check
                          loss function and whether val set is representative.
```

---

## SECTION 9: PLOTS AND VISUALISATIONS — COMPLETE SPECIFICATION

All plots must be saved to reports/plots/ with timestamp in filename.
All plots use matplotlib. No interactive plots — static PNG files only.
All plots include: title, axis labels, legend where applicable, gridlines.

---

### 9.1 Confusion Matrix Plot

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from app.config import CLASS_NAMES

def plot_confusion_matrix(d_true, d_binary, save_path):
    """
    Plots NUM_CLASSES × NUM_CLASSES confusion matrix as a heatmap.
    Uses argmax for display (see Section 2.6).
    """
    d_true_am   = d_true.argmax(axis=1)
    d_pred_am   = d_binary.argmax(axis=1)
    cm          = confusion_matrix(d_true_am, d_pred_am,
                                   labels=list(range(len(CLASS_NAMES))))
    # Normalise by row (true class) for readability
    cm_norm     = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[c.replace('_', '\n') for c in CLASS_NAMES],
                yticklabels=[c.replace('_', '\n') for c in CLASS_NAMES],
                ax=ax)
    ax.set_title('Confusion Matrix (row-normalised)', fontsize=14, pad=12)
    ax.set_ylabel('True Class',      fontsize=11)
    ax.set_xlabel('Predicted Class', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: {save_path}")
```

---

### 9.2 Per-Class F1 Bar Chart

```python
def plot_per_class_f1(per_class_f1, class_names, save_path, title='Per-Class F1'):
    """
    Horizontal bar chart of per-class F1.
    Bars below 0.40 shown in red, above in green.
    """
    colours = ['#e74c3c' if f1 < 0.40 else '#27ae60' for f1 in per_class_f1]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(class_names)), per_class_f1, color=colours)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axvline(x=0.40, color='orange', linestyle='--', linewidth=1.5,
               label='Low threshold (0.40)')
    ax.axvline(x=0.50, color='green',  linestyle='--', linewidth=1.5,
               label='Pass threshold (0.50)')
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    # Add value labels on bars
    for i, (bar, f1) in enumerate(zip(bars, per_class_f1)):
        ax.text(min(f1 + 0.01, 0.95), i, f'{f1:.3f}',
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Per-class F1 chart saved: {save_path}")
```

---

### 9.3 Calibration Curve (Reliability Diagram)

```python
def plot_calibration_curve(probs_before, probs_after, labels, save_path):
    """
    Plots reliability diagram: predicted probability vs actual fraction positive.
    Shows curve before and after temperature scaling.
    """
    n_bins = 10
    bins   = np.linspace(0, 1, n_bins + 1)

    def compute_curve(probs, lbl):
        mean_conf, frac_pos = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 10:
                mean_conf.append(probs[mask].mean())
                frac_pos.append(lbl[mask].mean())
        return np.array(mean_conf), np.array(frac_pos)

    p_flat  = np.array(probs_before).flatten()
    pa_flat = np.array(probs_after).flatten()
    l_flat  = np.array(labels).flatten()

    conf_b, frac_b = compute_curve(p_flat,  l_flat)
    conf_a, frac_a = compute_curve(pa_flat, l_flat)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax.plot(conf_b, frac_b, 'o-', color='#e74c3c', linewidth=2,
            label=f'Before (T=1.0)')
    ax.plot(conf_a, frac_a, 's-', color='#27ae60', linewidth=2,
            label=f'After  (T=T_disease)')
    ax.set_xlabel('Mean predicted probability', fontsize=11)
    ax.set_ylabel('Fraction of positives',      fontsize=11)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Calibration curve saved: {save_path}")
```

---

### 9.4 Uncertainty Distribution Histogram

```python
def plot_uncertainty_distribution(uncertainties_correct, uncertainties_wrong, save_path):
    """
    Overlaid histograms of uncertainty for correct vs incorrect predictions.
    Correct = model predicted the right class above threshold.
    Wrong   = model did not predict the right class (FN) or predicted wrong class (FP).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, 0.5, 30)
    ax.hist(uncertainties_correct, bins=bins, alpha=0.6, color='#27ae60',
            label=f'Correct (n={len(uncertainties_correct)})', density=True)
    ax.hist(uncertainties_wrong,   bins=bins, alpha=0.6, color='#e74c3c',
            label=f'Incorrect (n={len(uncertainties_wrong)})', density=True)
    ax.axvline(x=0.20, color='navy', linestyle=':', linewidth=1.5,
               label='OOD_UNC_THRESH (0.40 in config)')
    ax.set_xlabel('Uncertainty (MC Dropout std)', fontsize=11)
    ax.set_ylabel('Density',                      fontsize=11)
    ax.set_title('Uncertainty Distribution: Correct vs Incorrect Predictions', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Uncertainty distribution saved: {save_path}")
```

---

## SECTION 10: EVALUATION SUMMARY TABLE — WHAT "DONE" MEANS

The project is considered done when ALL of the following are true.
This is the definitive checklist. No exceptions without documented justification
in decisions.md.

```
MINIMUM FOR TRAINING COMPLETE (model can be used for testing):
  □ Phase 2 macro F1 > 0.50 (val set, from wandb)
  □ Crop accuracy > 0.85 (val set, from wandb)
  □ ECE after calibration < 0.10 (from 06_calibrate.py output)
  □ models/best_model.pt exists and is loadable
  □ models/temperature.pt exists with all three T values

MINIMUM FOR DEPLOYMENT-CANDIDATE (model can be shown to users):
  □ All "training complete" criteria above
  □ Validation report written (07_evaluate_validation.py)
    - Macro F1 >= 0.50 confirmed in report
    - Per-class F1 table shows no class below 0.20
  □ Server smoke test passes (setup/test_server.py)
  □ Tier-2 PlantDoc evaluation run (08_evaluate_tier2_plantdoc.py)
    - Macro F1 >= 0.55 on 4 mappable classes
    - Tier-2 lock recorded in context.md
  □ Local test evaluation run (10_evaluate_local_test.py)
    - Macro F1 >= 0.50

MINIMUM FOR DEPLOYMENT-VALIDATED (safe to recommend to farmers):
  □ All "deployment-candidate" criteria above
  □ Tier-3 Kerala evaluation run (09_evaluate_tier3_kerala.py)
    - >= 50 verified Kerala images collected
    - Per-class accuracy >= 0.70 on all classes with >= 5 Kerala images
    - >= 6 of 10 classes evaluated
  □ End-to-end manual test with at least 5 real diseased leaf photos
  □ Grad-CAM heatmaps verified as highlighting correct leaf regions (not background)
  □ All 10 diagnosis entries in diagnosis_lookup.json verified as agronomically correct
```

---

## SECTION 11: EVALUATION REPORTS — FILE NAMING AND STORAGE

All evaluation outputs are stored in reports/ (or reports/plots/ for images).
All filenames include a timestamp to allow multiple evaluation runs.

```
reports/
  validation_report_{YYYYMMDD_HHMMSS}.md
  tier2_plantdoc_{YYYYMMDD_HHMMSS}.md
  local_test_report_{YYYYMMDD_HHMMSS}.md
  tier3_kerala_{YYYYMMDD_HHMMSS}.md
  plots/
    confusion_matrix_{YYYYMMDD_HHMMSS}.png
    tier2_confusion_{YYYYMMDD_HHMMSS}.png
    localtest_confusion_{YYYYMMDD_HHMMSS}.png
    per_class_f1_{YYYYMMDD_HHMMSS}.png
    tier2_per_class_{YYYYMMDD_HHMMSS}.png
    localtest_per_class_{YYYYMMDD_HHMMSS}.png
    tier3_per_class_{YYYYMMDD_HHMMSS}.png
    calibration_curve_{YYYYMMDD_HHMMSS}.png
    uncertainty_dist_{YYYYMMDD_HHMMSS}.png
    localtest_per_source_{YYYYMMDD_HHMMSS}.png
```

**reports/ is committed to git** (unlike models/ and data/).
Report files are text/image files that document project progress.
They should be version-controlled so progress is recorded over time.
Add to .gitignore only: data/, models/*.pt, cache/, feedback.db.
Do NOT add reports/ to .gitignore.

---

## SECTION 12: KNOWN LIMITATIONS TO DOCUMENT IN REPORTS

Every tier-2 and tier-3 report must include a limitations section.
These are not failures — they are honest characterisations of where the model
may not work reliably.

```
LIMITATION 1: brassica_clubroot is diagnosed from above-ground symptoms only
  The model sees wilting and yellowing from leaf images. These symptoms are
  non-specific — many conditions cause wilting. True clubroot diagnosis requires
  uprooting to confirm root galls. The model can suggest clubroot as a possibility
  but cannot confirm it from leaf images alone.
  Urgency = HIGH is set in the diagnosis lookup because of the 20-year soil
  persistence, but farmers should verify by uprooting before taking drastic action.

LIMITATION 2: Co-infection detection is experimental
  The multi-label sigmoid architecture allows co-infection detection. However,
  training data for co-infected images is rare. The model is likely to miss
  co-infections and report only the dominant disease. Farmers should be advised
  that the system may undercount diseases on severely affected leaves.

LIMITATION 3: Tier-2 covers only 4 of 10 disease classes
  PlantDoc does not contain okra disease images. The 5 okra classes and
  brassica_clubroot were never evaluated on independent test data — only on
  the held-out local test split (same distribution as training). Real-world
  performance on okra diseases is validated only by tier-3 Kerala images.

LIMITATION 4: Symptom severity is a proxy
  Severity labels (mild/moderate/severe) were generated by a saliency-based
  proxy, not by agronomist expert annotation. The severity prediction is a useful
  indicator but should not be used as the sole basis for treatment urgency.
  The urgency field in diagnosis_lookup.json is based on the disease type,
  not on the predicted severity level.

LIMITATION 5: Model is trained on leaf images only
  Uploading photos of stems, roots, pods, or fruit will produce unreliable results.
  The validator rejects images that appear to lack plant content but cannot
  distinguish a healthy leaf from a diseased stem.
```
