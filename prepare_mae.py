"""Prepare Yale Brain Mets Longitudinal volumes for DynUNet MAE pretraining.

Accept the dataset access terms and authenticate with `hf auth login` first.

    python prepare_mae.py --output ./yale_mae
    python prepare_mae.py --output ./yale_mae_small --max-patients 20

This downloads eligible PRE, POST, T2 and FLAIR NIfTI files and writes
manifest.json with patient-disjoint train, validation and test partitions.
The Hugging Face revision is pinned for both the catalog and downloads.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

REPO = "geekyrakshit/Yale-Brain-Mets-Longitudinal"
SEQUENCES = ("PRE", "POST", "T2", "FLAIR")


def usable_geometry(row):
    try:
        shape, spacing = row["shape"], row["voxel_spacing_mm"]
        return len(shape) == len(spacing) == 3 and all(
            n > 0 and math.isfinite(s) and 0 < s <= 6.5 and n * s >= 50 for n, s in zip(shape, spacing)
        )
    except (TypeError, KeyError, ValueError):
        return False


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; reuse it or choose a new output directory.")
    revision = HfApi().repo_info(REPO, repo_type="dataset", revision=args.revision).sha
    catalog = load_dataset(
        REPO,
        "volumes",
        split="train",
        revision=revision,
        streaming=True,
        columns=[
            "patient_id",
            "study_id",
            "sequence",
            "nifti_path",
            "sha256",
            "integrity_status",
            "shape",
            "voxel_spacing_mm",
        ],
    )
    sequences = [args.sequence] if args.sequence else args.sequences
    selected = []
    excluded = Counter()
    for row in catalog:
        if row["sequence"] not in sequences:
            continue
        if row["integrity_status"] != "ok":
            excluded["integrity"] += 1
            continue
        # The paper excludes tiny FOVs and spacing >6.5 mm. Do not apply its
        # 200 kB file-size cutoff to this mirror's compressed .nii.gz files.
        if not usable_geometry(row):
            excluded["geometry_or_scout"] += 1
            continue
        selected.append(
            {
                k: row[k]
                for k in ("patient_id", "study_id", "sequence", "nifti_path", "sha256", "shape", "voxel_spacing_mm")
            }
        )
    patients = sorted({r["patient_id"] for r in selected})
    random.Random(args.seed).shuffle(patients)
    if args.max_patients:
        patients = patients[: args.max_patients]
    if len(patients) < 10:
        raise ValueError("Use at least 10 patients for this illustrative 80/10/10 split.")
    n_holdout = max(1, round(0.1 * len(patients)))
    test = set(patients[:n_holdout])
    val = set(patients[n_holdout : 2 * n_holdout])
    train = set(patients[2 * n_holdout :])
    assignment = {p: name for name, ids in (("train", train), ("val", val), ("test", test)) for p in ids}
    partitions = {name: [] for name in ("train", "val", "test")}
    chosen = [r for r in selected if r["patient_id"] in assignment]
    print(
        f"Revision {revision}; downloading {len(chosen)} {sequences} volumes from {len(patients)} patients; "
        f"excluded {dict(excluded)}.",
        flush=True,
    )
    for i, row in enumerate(chosen, 1):
        path = hf_hub_download(REPO, row["nifti_path"], repo_type="dataset", revision=revision)
        partitions[assignment[row["patient_id"]]].append({**row, "image": path})
        if i % 25 == 0 or i == len(chosen):
            print(f"Downloaded {i}/{len(chosen)}", flush=True)
    manifest = {"repo": REPO, "revision": revision, "sequences": sequences, "seed": args.seed, "partitions": partitions}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved {manifest_path}. Keep this split for downstream evaluation too.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="./yale_mae", help="Directory for manifest.json and later training outputs")
    sequences = parser.add_mutually_exclusive_group()
    sequences.add_argument("--sequence", choices=SEQUENCES, help="Prepare one sequence only")
    sequences.add_argument("--sequences", choices=SEQUENCES, nargs="+", default=list(SEQUENCES))
    parser.add_argument("--revision", default=None, help="Dataset revision; resolved to an immutable commit")
    parser.add_argument("--max-patients", type=int, default=0, help="0 uses all eligible patients")
    parser.add_argument("--seed", type=int, default=42, help="Seed for patient-level split")
    args = parser.parse_args()
    if args.max_patients < 0:
        parser.error("max-patients must be nonnegative")
    prepare(args)


if __name__ == "__main__":
    main()
