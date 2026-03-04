# Scripts Folder

This folder provides canonical script entrypoints for the joint-angle-regression workflow.

Most training/analysis scripts now live here directly. A few runtime launchers remain thin wrappers where appropriate.

## Recommended usage

Run scripts from this folder (examples):

```bash
python scripts/session_gui.py
python scripts/train_feature_extractor.py --help
python scripts/train_regressor.py --help
python scripts/plot_session6_model_comparison.py --help
```

## Notes

- New work should use `scripts/*` paths in docs and automation.
