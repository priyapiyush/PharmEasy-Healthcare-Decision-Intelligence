Domain: Healthcare

## How to run

From a clean clone of the repository root:

```text
pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The generator must print `800 10 170 e26b46e82f1a1af44dcabc133050802e`. All scripts run offline and regenerate the committed files in `outputs/`.

## Part 1 findings

### R1.1

The test rejects H0 because `z_stat = 2.8930` exceeds the two-tailed critical value 1.96. A Type I error here would mean concluding that the average care-plan cost differs from INR 8,000 when the true average is actually INR 8,000. The estimated gap is `mean_value - 8.0 = 0.9382` thousand INR, or about INR 938 per patient; this is statistically significant and large enough to matter for budget planning.

### R1.2

`n_required = 1293` rows are required compared with 800 available. The planning team needs about 493 additional records to target a 95% margin of error of 0.5 thousand INR.

### R1.3

`gd_slope_gap = 0.0000`. Gradient descent reaches the closed-form line because the standardised one-predictor squared-error objective is a convex quadratic and full-batch updates with this learning rate converge to its unique minimum.

### R1.4

`resid_sd_ratio_raw = 3.7525` and `resid_sd_ratio_log = 1.0396`. The raw model shows a strong funnel because the residual spread grows with fitted cost; the log-log model largely removes it.

### R1.5

The predictors with coefficient absolute value above 0.05 are `ln(X4)`, X2, X3, and X6. The multiple-regression adjusted R² is `mr_adj_r2 = 0.7403`, compared with `loglog_adj_r2 = 0.6561`, so the extra predictors improve adjusted fit by 0.0842 and earn their place under this mechanical comparison.

## Part 2 findings

### R2.1

The 200-update losses from lowest to highest are `eta = 1.0: 0.3703`, `eta = 0.1: 0.3776`, and `eta = 0.01: 0.5091`. The 0.01 rate is too small to learn quickly; 1.0 performs best in this run, although a still larger rate could become unstable.

### R2.2

The three largest absolute weights are X3 (`0.9013`), X1 (`-0.5554`), and X2 (`0.5505`). X3 has the largest odds ratio: a one-training-standard-deviation increase multiplies estimated no-show odds by `2.4629`, holding the other predictors fixed.

### R2.3

The blind accuracy is `0.7875`, compared with `0.8042` for logistic regression at τ = 0.5. Accuracy is weak here because only 51 of the 240 test records are positive, so a model can look accurate while missing many no-shows. Lowering τ from 0.5 to 0.3 raises recall from `0.4706` to `0.7451`, while precision falls from `0.5455` to `0.4634`.

### R2.4

At τ = 0.3, logistic regression has F1 `0.5714` and recall `0.7451`; KNN has F1 `0.3696` and recall `0.3333`. Logistic regression is better on both measures.

### R2.5

I would deploy logistic regression at `cost_tau = 0.1500`. Its test cost is INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 when nobody is flagged. The threshold is chosen on training rows so the test set remains an honest final evaluation rather than part of model selection.

## Part 3 findings

### R3.1

The successive WCSS drops are 659.8282 from K=2 to K=3, 108.0739 from K=3 to K=4, 171.0575 from K=4 to K=5, and 45.5466 from K=5 to K=6. The elbow rule selects K=3 because it is the smallest qualifying K. The non-monotone drops reflect the pinned starting rows, which provide a reproducible solution but not necessarily the global optimum for every K.

### R3.2

- Segment A — Stable low-friction users: long registration, low missed visits, low copay burden, and high adherence. Keep them engaged with routine reminders and recurring-care renewal options.
- Segment B — High-friction, high-cost users: more missed visits, high average bills, and moderate copay burden. Use proactive confirmation, flexible scheduling, and payment-support information.
- Segment C — New, high-burden users: short registration history, highest copay burden, lowest adherence, and the highest no-show rate. Provide early onboarding, stronger reminders, and an easy rescheduling path.

Segment C has the highest outcome rate at `0.4664`.

### R3.3

At the cut, single linkage produces `E1`; `E2+E3+E6`; `E4`; and `E5`. Complete linkage produces `E1+E4+E5` and `E2+E3+E6`. Single linkage is driven by the nearest pair of members, so a chain of short links can join groups even when other cross-group distances are large. Complete linkage requires the farthest cross-group pair to be small, so it preserves compact groups and delays the final merge.

### R3.4

PC1 is driven mainly by X1 (`+0.5317`), X3 (`-0.5187`), X5 (`+0.4418`), and X2 (`-0.4100`), with a smaller X4 contribution (`-0.2916`). Three components are needed to retain at least 80% of the variance. The mean PC1 scores `A = 1.7773`, `B = -0.7831`, and `C = -1.6709` show strong separation from established, low-friction users toward shorter-tenure, higher-burden users.

## Part 4 findings

### R4.1

S4 is the best split because it has the largest information gain (`0.1252`) and Gini gain (`0.0606`). Growing a tree until every leaf is pure would fit noise and produce unstable rules; a maximum depth, minimum leaf size, or pruning rule can stop excessive growth.

### R4.2

I would ship logistic regression at τ = 0.3: F1 is `0.5714` and recall is `0.7451`. Combining models did not help: hard voting has F1 `0.5344` and recall `0.6863`, while soft voting has F1 `0.5507` and recall `0.7451`.

### R4.3

After round 1 normalisation, the two points misclassified by h1 have total weight `0.5000`. Round 2 trusts h2 more because its weighted error is `0.1667`, lower than round 1's `0.2500`, giving it a larger alpha (`0.8047` versus `0.5493`).

### R4.4

Model A option 2 has these output shapes and parameter counts:

| Layer | Output shape | Parameters |
|---|---:|---:|
| Input | 28 × 28 × 1 | 0 |
| Conv1 | 28 × 28 × 6 | 156 |
| ReLU | 28 × 28 × 6 | 0 |
| Max-pool 1 | 14 × 14 × 6 | 0 |
| Conv2 | 10 × 10 × 16 | 2,416 |
| ReLU | 10 × 10 × 16 | 0 |
| Max-pool 2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense1 | 32 | 12,832 |
| Dense2 | 10 | 330 |

The CNN total is 15,734 parameters. Model B has `W_xh` shape 128 × 64, `W_hh` shape 128 × 128, and `W_hy` shape 16 × 128, with 128 and 16 bias parameters; its total is 26,768. The recurrent weights are shared at every time step, so the parameter count does not grow when the sequence gets longer. I would use ReLU in hidden layers and a 10-class softmax output with categorical cross-entropy. A very deep plain network can lose useful gradients as they pass backward through many layers, so early layers stop learning. A skip connection provides a direct identity path for information and gradients, making optimization easier.

## Model-choice memo

To: Head of the scheduling-operations team

I recommend deploying the structured-data logistic-regression model with an alert threshold of τ = 0.15. On the held-out test rows, this strategy costs INR 6,050 under the stated rule of INR 50 per false alarm and INR 400 per missed no-show. The fixed τ = 0.5 comparator costs INR 11,800, while flagging nobody costs INR 20,400. At τ = 0.3, logistic regression reaches F1 = 0.5714 and recall = 0.7451; the selected lower threshold is appropriate because missing a no-show is much more expensive than an unnecessary reminder. The model is preferable to KNN, whose test F1 is 0.3696 and recall is 0.3333. The main limitation is that this is synthetic data evaluated on one fixed split, so the threshold and cost estimate need prospective validation. Neither Model A nor Model B should be built before the structured scheduling model is operating and its data-generating process is validated; those two models are currently only architecture-sizing exercises.

## Limitations

- The records are synthetic and do not represent PharmEasy's real patients, operations, products, policies, or performance.
- Classification uses one pinned train/test split, so the reported test results may not generalise to another sample.
- The test set contains only 51 positive no-show records, making some rates sensitive to a few cases.
- The alert threshold and KNN K are selected for this split and require monitoring and recalibration in production.
- The image and text neural-network sections count parameters only; they do not train or validate either model.
- The analysis is for scheduling and cost planning only and is not clinical advice or a diagnostic system.
