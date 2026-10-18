# Anonymous review artifact

This repository contains code and saved model outputs for a study of how the number of items flagged by abusive-language detectors affects comparisons of cross-domain recall.

## Main result

Across twelve BERT runs on the YouTube evaluation set, in-distribution macro-F1 and cross-domain recall have a raw Spearman correlation of -0.739 (p = 0.006). After controlling for the number of comments flagged with rank-based partial Spearman correlation, the estimate is -0.431 (p = 0.186). Across twelve model-family/test-set comparisons, none of the partial associations is significant after Holm correction (smallest adjusted p = 0.471). The evidence supports reporting flag rate with recall; it does not establish a universal relationship or show that the association is absent.

## Contents

- `predictions/`: per-item predictions and checksums for the twelve BERT runs on the in-distribution test set.
- `external/probs/` and `external/probs_roberta/`: saved probabilities for BERT and RoBERTa across six test sets.
- `matched/`: saved probabilities and scripts for equal-flag-count comparisons.
- `data/`: run ledgers and aggregate result tables.
- `code/`: analysis, figure-generation, and training notebooks.
- `figures/fig2_tradeoff.pdf`: corrected manuscript figure; regenerate it with `python code/make_figures.py`.
- `analysis_repair/`: corrected partial Spearman script and result table.

The original text of training and evaluation corpora is not included. Reconstruct test sets from the source datasets and their published terms. Model checkpoints are not included.

## Reproduce analyses

For the corrected correlations, install `numpy`, `pandas`, and `scipy`, then run from this repository's root:

```powershell
python analysis_repair/partial_spearman.py --repo-root .
```

For the saved YouTube agreement and coverage results:

```powershell
python code/agreement_analysis.py
```

For the matched-count analysis:

```powershell
python matched/matched_analysis.py
```

The analysis scripts use saved predictions or probabilities and do not retrain models. Retraining notebooks require a compatible GPU environment and the source datasets. To retrain, set the model repository in the configuration cell to a repository under the user's own account and attach the required datasets in the notebook environment.

## Anonymity note

This copy is intended for anonymous review. It omits Git history and direct links to author-owned accounts. The author remains responsible for checking all files and metadata before sharing.
