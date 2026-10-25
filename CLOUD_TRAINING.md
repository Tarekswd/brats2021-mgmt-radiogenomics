# Training Task 2 (MGMT) on real data with a cloud GPU

This produces the first **measured** results for this repository. Report only what these runs log.

## 1. Kaggle notebook setup

1. On Kaggle, open the competition **RSNA-MICCAI Brain Tumor Radiogenomic Classification** and accept its rules (you have to do this yourself).
2. Create a new notebook and choose **Add Input → Competition → rsna-miccai-brain-tumor-radiogenomic-classification**.
3. Under **Settings**, set the accelerator to **GPU** and turn **Internet on** (needed for `git clone`).
4. Check the mount path, since it can differ:
   ```bash
   !ls /kaggle/input/competitions/rsna-miccai-brain-tumor-radiogenomic-classification
   ```
   You should see `train/`, `test/` and `train_labels.csv`. Each `train/<BraTS21ID>/` folder holds `FLAIR/ T1w/ T1wCE/ T2w/` with `Image-N.dcm` files.

## 2. Run 5-fold cross-validation

```bash
!git clone https://github.com/Tarekswd/brats2021-mgmt-radiogenomics.git && cd brats2021-mgmt-radiogenomics && pip install -q -r requirements.txt
```

```bash
%%bash
cd brats2021-mgmt-radiogenomics
D=/kaggle/input/competitions/rsna-miccai-brain-tumor-radiogenomic-classification
for F in 0 1 2 3 4; do
  python scripts/train_task2.py --data-root $D/train --labels-csv $D/train_labels.csv \
    --model attention_mil --fold $F --epochs 30 --batch-size 4 --num-slices 16 --slice-size 128 \
    --num-workers 4 --output-dir /kaggle/working/runs/attention_mil/fold$F
done
```

- The script refuses to start if any labelled patient folder is missing. It never falls back to synthetic data when `--data-root` is given.
- Patients `00109 00123 00709` are excluded by default (`--exclude`). Pass `--exclude` with no IDs to keep them.
- Kaggle sessions have a time limit. Time the first epoch, then lower `--epochs`, or run one fold per session.
- Measured on a Kaggle T4: one epoch of fold 0 (582-patient cohort, 16 slices at 128×128) took 61 s.
- Kaggle copies the competition data into the session when you add it (about 137 GB). Expect a wait before the first cell runs.
- Repeat with `--model cnn_lstm` for the second baseline.
- The 500M model is **not** wired into `train_task2.py`. Training it needs a large GPU and code changes, so don't report numbers for it.

## 3. Summarise honestly

```python
import pandas as pd, glob
for run in sorted(glob.glob("/kaggle/working/runs/*")):
    logs = [pd.read_csv(f) for f in sorted(glob.glob(f"{run}/fold*/training_log.csv"))]
    last = pd.Series([d.roc_auc.iloc[-1] for d in logs])
    best = pd.Series([d.roc_auc.max() for d in logs])
    print(run, f"final-epoch AUC {last.mean():.3f} ± {last.std():.3f} | best-epoch AUC {best.mean():.3f} ± {best.std():.3f} (optimistic) | n_folds={len(logs)}")
```

Put the **final-epoch** mean ± std in the README table. The best-epoch number picks the epoch using the validation set itself, so it is optimistically biased. If you quote it, label it as such. Download `/kaggle/working/runs` (the logs and fold JSON files) and commit the CSV logs under `experiments/` so anyone can check the numbers.
