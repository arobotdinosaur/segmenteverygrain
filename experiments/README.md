# Segment Every Grain experiment pipeline

This folder contains repeatable experiment configs for the cleanup branch.

The new pipeline separates the workflow into five stages:

1. fit synthetic-noise parameters,
2. generate synthetic noisy image/mask pairs,
3. build a training set from clean/synthetic/real sources,
4. train or fine-tune a U-Net,
5. evaluate the trained model on a held-out image/mask folder.

Run a full configured experiment with:

```bash
python scripts/run_experiment.py --config experiments/configs/exp1_clean_synthetic.json
```

Run individual stages with:

```bash
python scripts/fit_noise.py --clean-dir surrogate_data/clean_images --reference-noisy-dir surrogate_data/reference_noisy_images --output runs/theta.json
python scripts/generate_synthetic.py --clean-dir surrogate_data/clean_images --theta runs/theta.json --reference-noisy-dir surrogate_data/reference_noisy_images --output-dir runs/synthetic_noisy
python scripts/train_experiment.py --run-dir runs/manual --clean-dir surrogate_data/clean_images --synthetic-dir runs/synthetic_noisy --pretrained-model models/seg_model.keras
python scripts/evaluate_experiment.py --model runs/manual/model.keras --eval-dir surrogate_data/annotated_eval_images --output-dir runs/manual/evaluation
```

The four starter configs are:

- `exp1_clean_synthetic.json`
- `exp2_clean_real3.json`
- `exp3_clean_synthetic_real3.json`
- `exp4_fit_r_train_3_minus_r.json`

