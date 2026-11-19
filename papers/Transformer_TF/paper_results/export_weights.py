"""
Collect the trained model weights used in the paper into archives for Zenodo,
and write a manifest (path, size, SHA-256) to paper_results/weights_manifest.csv.

All paper models carry the date tag MODEL_DATE (scripts/config_models.py) and are
read from cfg.dataPath/cache/TF_Europe/ (path_cache in config_TF_Europe.py):

- WITHIN_REGION/models/: within-region LSTM and Transformer models (2.2), for the
  Sinkhorn glacier split (GL), the measurement split (ID) and the random splits
  (RAND0-4)
- CROSS_REGION/models/<SRC>/: single-region LSTM and Transformer models (2.3, 3.2)
- FD_MODEL/models/foundation/: multi-region Transformer (3.1-3.4)
- FD_MODEL/models/finetuned/: head-only and full fine-tuned models, 10% local
  data, 5 repeats (3.2-3.4)
- FD_MODEL/ft_fraction_sensitivity/models/: fine-tuned models and loss histories
  of the fraction sensitivity (Fig. A8, A9); separate archive, as it is larger

Exploratory models in the same folders (DAN and adapter fine-tuning) are not
part of the paper and are skipped.

Usage (from the repository root):
    python papers/Transformer_TF/paper_results/export_weights.py <output_dir>
"""

import hashlib
import os
import sys
import tarfile
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repository root

import massbalancemachine as mbm
from papers.Transformer_TF.scripts.config_TF_Europe import path_cache
from papers.Transformer_TF.scripts.config_models import MODEL_DATE

HERE = Path(__file__).resolve().parent


def paper_weight_files(cache_dir: Path):
    d = MODEL_DATE
    main = (
        sorted((cache_dir / "WITHIN_REGION" / "models").glob(f"*_within_*_{d}.pt"))
        + sorted((cache_dir / "CROSS_REGION" / "models").glob(f"*/*_xreg_*_{d}.pt"))
        + sorted((cache_dir / "FD_MODEL" / "models" / "foundation").glob(f"*_{d}.pt"))
        + sorted(
            p
            for strat in ("full", "head_only")
            for p in (cache_dir / "FD_MODEL" / "models" / "finetuned").glob(
                f"transformer_ft_{strat}_*_{d}.pt"
            )
        )
    )
    sens_dir = cache_dir / "FD_MODEL" / "ft_fraction_sensitivity" / "models"
    sens = sorted(sens_dir.glob(f"*_{d}.pt")) + sorted(
        sens_dir.glob(f"*_{d}.history.json")
    )
    return {"main": main, "ft_fraction_sensitivity": sens}


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(out_dir: Path):
    cfg = mbm.EuropeTFConfig()
    cache_dir = Path(cfg.dataPath) / path_cache
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, files in paper_weight_files(cache_dir).items():
        archive = out_dir / f"MBM_TF_paper_weights_{name}_{MODEL_DATE}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for p in files:
                rel = p.relative_to(cache_dir)
                tar.add(p, arcname=str(rel))
                rows.append(
                    {
                        "archive": archive.name,
                        "path": str(rel),
                        "bytes": p.stat().st_size,
                        "sha256": sha256(p),
                    }
                )
        print(f"{archive} ({len(files)} files, {archive.stat().st_size / 1e6:.0f} MB)")

    manifest = pd.DataFrame(rows)
    manifest.to_csv(HERE / "weights_manifest.csv", index=False)
    print(f"Manifest: {HERE / 'weights_manifest.csv'} ({len(manifest)} files)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
