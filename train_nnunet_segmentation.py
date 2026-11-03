"""Train nnU-Net v2 on PROTEAS with MONAI's nnUNetV2Runner.

Install: uv pip install 'nnunetv2>=2.6.2,<3' huggingface_hub nibabel

Download the same dataset used by train_dynmae_semantic_segmentation.py,
export nnU-Net data, plan/preprocess, and train a 3D full-resolution model:
    python train_nnunet_segmentation.py --output ./proteas_nnunet
Use an existing local PROTEAS manifest (including a restored prepared archive):
    python train_nnunet_segmentation.py \
        --manifest ./proteas_segmentation/manifest.json --output ./proteas_nnunet
Export data without installing/running nnU-Net:
    python train_nnunet_segmentation.py --manifest ./proteas_segmentation/manifest.json --prepare-only
Resume using the same arguments and existing preprocessing:
    python train_nnunet_segmentation.py --output ./proteas_nnunet --resume --skip-preprocessing
For a short run, use --trainer-class-name nnUNetTrainer_1epoch.

Default channel order: T1, T1ce, T2, FLAIR. Labels: 0 background,
1 necrotic core, 2 enhancing tumor, 3 edema. Published train/validation splits
are preserved as fold 0; the test split is unused by default. As in the DyneMAE
script, --train-on-train-val trains on train+validation and uses test for
validation and checkpoint selection. This is a single fixed split, not five-fold
cross-validation. Both roles are stored in imagesTr/labelsTr, with their training
and validation membership recorded in splits_final.json.

nnU-Net chooses its own normalization, spacing, patches, augmentation, loss,
batch size, and learning-rate schedule. Its default trainer runs 1000 epochs
of sampled patches, not complete dataset passes. Use --trainer-class-name
nnUNetTrainer_100epochs for 100 epochs. Data are symlinked by default; use
--copy-data for a portable export. Use a new --output when changing the dataset,
modalities, or split policy. Checkpoints and native nnU-Net logs are stored in
OUTPUT/nnUNet_results/DatasetXXX_PROTEAS/TRAINER__PLANS__CONFIGURATION/fold_0.
Native best-checkpoint selection uses smoothed patch Dice during training;
full-volume validation runs at the end and writes validation/summary.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from typing import Any

DATASET = "geekyrakshit/PROTEAS-Brain-Metastases-Segmentation"
MODALITIES = ("t1", "t1ce", "t2", "flair")
LABELS = {"0": "background", "1": "necrotic_core", "2": "enhancing_tumor", "3": "edema"}


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_sources(args: argparse.Namespace) -> dict[str, list[str]]:
    if args.train_on_train_val:
        return {"train": ["train", "validation"], "val": ["test"]}
    return {"train": ["train"], "val": ["validation"]}


def select_rows(manifest: dict, args: argparse.Namespace) -> dict[str, list[dict]]:
    selected = {}
    for role, sources in split_sources(args).items():
        rows = []
        for source in sources:
            if not manifest["partitions"].get(source):
                raise ValueError(f"Required source split '{source}' is missing or empty.")
            rows.extend({**row, "source_split": source} for row in manifest["partitions"][source])
        limit = args.max_train_cases if role == "train" else args.max_val_cases
        selected[role] = rows[:limit] if limit else rows
    return selected


def prepare_dataset(args: argparse.Namespace) -> Path:
    """Fetch selected original NIfTIs at one immutable Hub revision."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise ImportError(
            "Dataset downloads require `uv pip install huggingface_hub` or a local --manifest."
        ) from error

    output = args.data_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    existing = destination.is_file()
    if existing:
        manifest = json.loads(destination.read_text())
    else:
        revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
        source = hf_hub_download(args.dataset, "manifest.json", repo_type="dataset", revision=revision)
        manifest = json.loads(Path(source).read_text())
        channels = args.modalities or list(MODALITIES)
        indices = [manifest["channel_order"].index(channel) for channel in channels]
        for split, rows in manifest["partitions"].items():
            manifest["partitions"][split] = [{**row, "image": [row["image"][i] for i in indices]} for row in rows]
        manifest.update(repo_id=args.dataset, revision=revision, channel_order=channels, path_base=".")
    if args.modalities and args.modalities != manifest["channel_order"]:
        raise ValueError("--modalities must match the existing manifest channel_order; use a new --data-dir.")
    repo = manifest.get("repo_id", args.dataset)
    revision = manifest.get("revision")
    if args.dataset_revision:
        expected = HfApi().dataset_info(repo, revision=args.dataset_revision).sha
        if revision != expected:
            raise ValueError("The existing manifest does not match --dataset-revision.")
    if not revision:
        raise ValueError("Downloads require a manifest with a pinned dataset revision; use --manifest for local files.")
    if manifest.get("prepared_archive"):
        raise ValueError(
            "Use --manifest with a restored prepared archive, or --dataset " + DATASET + " for direct downloads."
        )
    names = sorted(
        {
            name
            for rows in select_rows(manifest, args).values()
            for row in rows
            for name in [*row["image"], row["label"]]
        }
    )
    checksum_path = hf_hub_download(repo, "SHA256SUMS", repo_type="dataset", revision=revision)
    checksums = {
        name.lstrip(" *"): digest
        for digest, name in (
            line.split(maxsplit=1) for line in Path(checksum_path).read_text().splitlines() if line.strip()
        )
    }

    def fetch(name: str) -> None:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"Invalid Hub dataset path: {name}")
        expected = checksums[name]
        local = output / name
        if local.is_file() and sha256(local) == expected:
            return
        downloaded = Path(
            hf_hub_download(
                repo, name, repo_type="dataset", revision=revision, local_dir=output, force_download=local.exists()
            )
        )
        if sha256(downloaded) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")

    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for _ in pool.map(fetch, names):
            pass
    if not existing:
        write_json(destination, manifest)
    return destination


def read_manifest(path: Path, args: argparse.Namespace) -> tuple[dict, dict[str, list[dict]]]:
    """Resolve the same channel, label, and patient-separated split contract as DyneMAE."""
    manifest = json.loads(path.read_text())
    channels = manifest["channel_order"]
    if not channels or len(set(channels)) != len(channels) or not set(channels).issubset(MODALITIES):
        raise ValueError("Manifest channel_order must contain unique PROTEAS modality names.")
    if args.modalities and args.modalities != channels:
        raise ValueError("--modalities must match the manifest channel_order.")
    if manifest.get("labels") != LABELS:
        raise ValueError("Expected PROTEAS labels 0 background, 1 necrotic core, 2 enhancing tumor, 3 edema.")
    if args.dataset_revision and manifest.get("revision") != args.dataset_revision:
        from huggingface_hub import HfApi

        expected = HfApi().dataset_info(manifest.get("repo_id", args.dataset), revision=args.dataset_revision).sha
        if manifest.get("revision") != expected:
            raise ValueError("The existing manifest does not match --dataset-revision.")
    if manifest.get("subset_for_smoke_test") and not (args.max_train_cases or args.max_val_cases):
        raise ValueError("This is a smoke-test manifest. Use case limits or a complete dataset.")
    patients = {split: {row["patient_id"] for row in rows} for split, rows in manifest["partitions"].items()}
    for index, first in enumerate(patients):
        for second in list(patients)[index + 1 :]:
            if patients[first] & patients[second]:
                raise ValueError(f"Patient leakage between {first} and {second}.")
    path_base = manifest.get("path_base", ".")
    base = path.parent if path_base in (".", "repository root") else path.parent / path_base
    selected = select_rows(manifest, args)
    identifiers = set()
    for rows in selected.values():
        for row in rows:
            if row["case_id"] in identifiers:
                raise ValueError(f"Duplicate case_id: {row['case_id']}")
            identifiers.add(row["case_id"])
            if len(row["image"]) != len(channels):
                raise ValueError(f"Channel count mismatch for {row['case_id']}.")
            row["image"] = [str((base / name).resolve()) for name in row["image"]]
            row["label"] = str((base / row["label"]).resolve())
            for name in [*row["image"], row["label"]]:
                if not Path(name).is_file():
                    raise FileNotFoundError(f"Missing NIfTI: {name}")
                if not name.endswith(".nii.gz"):
                    raise ValueError(f"Expected a .nii.gz PROTEAS volume: {name}")
    return manifest, selected


def export_dataset(args: argparse.Namespace, path: Path, manifest: dict, partitions: dict) -> tuple[dict, list[dict]]:
    """Export unchanged NIfTIs with explicit channel names and one fixed split."""
    output = args.output.resolve()
    dataset_name = f"Dataset{args.dataset_id:03d}_PROTEAS"
    raw = output / "nnUNet_raw" / dataset_name
    cases = []
    splits: dict[str, list[str]] = {"train": [], "val": []}
    for role, rows in partitions.items():
        for row in rows:
            identifier = f"PROTEAS_{len(cases):05d}"
            cases.append({**row, "nnunet_case_id": identifier, "role": role})
            splits[role].append(identifier)
    signature = {
        "manifest_sha256": sha256(path),
        "dataset_name": dataset_name,
        "channels": manifest["channel_order"],
        "split_sources": split_sources(args),
        "cases": cases,
        "files": {
            name: {"size": Path(name).stat().st_size, "mtime_ns": Path(name).stat().st_mtime_ns}
            for row in cases
            for name in [*row["image"], row["label"]]
        },
        "copy_data": args.copy_data,
    }
    recipe_path = output / "recipe.json"
    if recipe_path.exists():
        if json.loads(recipe_path.read_text())["signature"] != signature:
            raise ValueError("Dataset, modalities, source files, or splits changed. Use a new --output.")
    elif raw.exists():
        raise FileExistsError(f"{raw} exists without a matching recipe.json; use a new --output.")
    output.mkdir(parents=True, exist_ok=True)
    recipe = {
        "signature": signature,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset_revision": manifest.get("revision"),
    }
    # Persist the identity before exporting so interrupted exports can be retried.
    write_json(recipe_path, recipe)
    for directory in ("imagesTr", "labelsTr"):
        (raw / directory).mkdir(parents=True, exist_ok=True)
    for row in cases:
        identifier = row["nnunet_case_id"]
        pairs = [
            (Path(name), raw / "imagesTr" / f"{identifier}_{index:04d}.nii.gz")
            for index, name in enumerate(row["image"])
        ]
        pairs.append((Path(row["label"]), raw / "labelsTr" / f"{identifier}.nii.gz"))
        for source, target in pairs:
            if target.is_symlink():
                if target.resolve() != source:
                    raise FileExistsError(f"Export points to a different source: {target}")
            elif target.exists():
                if not args.copy_data or sha256(target) != sha256(source):
                    raise FileExistsError(f"Export conflicts with source: {target}")
            elif args.copy_data:
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
            else:
                target.symlink_to(source)
    write_json(
        raw / "dataset.json",
        {
            "channel_names": {str(i): channel for i, channel in enumerate(manifest["channel_order"])},
            "labels": {name: int(index) for index, name in LABELS.items()},
            "numTraining": len(cases),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "NibabelIO",
        },
    )
    write_json(raw / "splits_final.json", [splits])
    write_json(output / "case_mapping.json", cases)
    return recipe, [splits]


def run_nnunet(args: argparse.Namespace, recipe: dict, splits: list[dict]) -> None:
    """Run MONAI planning/preprocessing and launch nnU-Net in this Python environment."""
    from monai.apps.nnunet import nnUNetV2Runner
    from monai.utils import optional_import

    _, available = optional_import("nnunetv2", version="2.6.2")
    if not available:
        raise ImportError("Training requires `uv pip install 'nnunetv2>=2.6.2,<3' nibabel`.")
    output = args.output.resolve()
    dataset_name = recipe["signature"]["dataset_name"]
    roots = {
        "nnunet_raw": str(output / "nnUNet_raw"),
        "nnunet_preprocessed": str(output / "nnUNet_preprocessed"),
        "nnunet_results": str(output / "nnUNet_results"),
    }
    # The runner sets these variables before importing nnunetv2.paths. Do not
    # import nnU-Net earlier: that module caches the roots at import time.
    runner = nnUNetV2Runner(
        input_config={**roots, "dataset_name_or_id": str(args.dataset_id)},
        trainer_class_name=args.trainer_class_name,
        work_dir=str(output),
        export_validation_probabilities=args.save_probabilities,
    )
    preprocessed = Path(roots["nnunet_preprocessed"]) / dataset_name
    model_dir = (
        Path(roots["nnunet_results"])
        / dataset_name
        / f"{args.trainer_class_name}__{args.plans}__{args.configuration}"
        / "fold_0"
    )
    checkpoints = [
        path
        for stem in ("checkpoint_latest", "checkpoint_best", "checkpoint_final")
        for suffix in (".pth", ".safetensors")
        if (path := model_dir / (stem + suffix)).is_file()
    ]
    if not args.preprocess_only:
        if args.resume and not checkpoints:
            raise FileNotFoundError(f"Cannot resume: no nnU-Net checkpoint in {model_dir}.")
        if checkpoints and not args.resume:
            raise FileExistsError(f"Checkpoints exist in {model_dir}; use --resume or a new --output.")
    planning = {
        "dataset": recipe["signature"],
        "configuration": args.configuration,
        "planner": args.planner,
        "plans": args.plans,
        "gpu_memory_target": args.gpu_memory_target,
    }
    metadata_path = output / "preprocessing.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != planning:
        raise ValueError("Planning settings changed. Use the original settings or a new --output.")
    if args.skip_preprocessing:
        if not metadata_path.exists():
            raise FileNotFoundError("--skip-preprocessing requires a completed preprocessing.json from this script.")
    else:
        runner.plan_and_process(
            npfp=args.processes,
            verify_dataset_integrity=True,
            pl=args.planner,
            gpu_memory_target=args.gpu_memory_target,
            overwrite_plans_name=args.plans,
            c=(args.configuration,),
            n_proc=(args.processes,),
        )
    plans_path = preprocessed / f"{args.plans}.json"
    plans = json.loads(plans_path.read_text())
    if args.configuration not in plans["configurations"]:
        raise ValueError(f"Configuration {args.configuration} is absent from {plans_path}.")
    write_json(preprocessed / "splits_final.json", splits)
    write_json(metadata_path, planning)
    if args.preprocess_only:
        print(f"Preprocessing complete: {preprocessed}")
        return
    command, environment = runner.train_single_model_command(
        args.configuration, 0, args.gpu_id, {"p": args.plans, "c": args.resume}
    )
    # Use the active interpreter rather than a console script from another
    # environment. Device is a single-dash option, unlike the runner's kwargs.
    command = [
        sys.executable,
        "-c",
        "from nnunetv2.run.run_training import run_training_entry; run_training_entry()",
        *command[1:],
        "-device",
        args.device,
    ]
    environment["nnUNet_n_proc_DA"] = str(args.augmentation_workers)
    if args.device == "cpu":
        environment["nnUNet_compile"] = "false"
    write_json(output / "training_command.json", {"argv": command, "nnUNet_n_proc_DA": args.augmentation_workers})
    subprocess.run(command, env=environment, check=True)
    print(f"Training complete; checkpoints and validation results: {model_dir}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DATASET, help="Hub dataset containing original NIfTIs and manifest.json")
    parser.add_argument("--dataset-revision", help="Hub branch, tag, or commit; downloads pin its resolved commit")
    parser.add_argument("--manifest", type=Path, help="Existing local PROTEAS manifest; avoids downloads")
    parser.add_argument("--data-dir", type=Path, default=Path("./proteas_segmentation"))
    parser.add_argument("--output", type=Path, default=Path("./proteas_nnunet_segmentation"))
    parser.add_argument(
        "--modalities", choices=MODALITIES, nargs="+", help="Input order; must match an existing manifest"
    )
    parser.add_argument(
        "--train-on-train-val",
        action="store_true",
        help="Train on train+validation; use test for validation and checkpoint selection",
    )
    parser.add_argument("--max-train-cases", type=int, help="Limit training cases for a smoke test")
    parser.add_argument("--max-val-cases", type=int, help="Limit validation cases for a smoke test")
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--dataset-id", type=int, default=501, help="nnU-Net dataset ID, 1..999 (default: %(default)s)")
    parser.add_argument("--copy-data", action="store_true", help="Copy NIfTIs; default is absolute symlinks")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--prepare-only", action="store_true", help="Download/validate and export raw nnU-Net data, then exit"
    )
    stages.add_argument(
        "--preprocess-only", action="store_true", help="Export, plan and preprocess, then exit before training"
    )
    parser.add_argument("--skip-preprocessing", action="store_true", help="Reuse this output's completed preprocessing")
    parser.add_argument("--configuration", choices=("3d_fullres", "2d"), default="3d_fullres")
    parser.add_argument(
        "--trainer-class-name", default="nnUNetTrainer", help="Installed nnU-Net trainer; e.g. nnUNetTrainer_100epochs"
    )
    parser.add_argument("--planner", default="ExperimentPlanner")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument(
        "--gpu-memory-target", type=float, default=8, help="Planner GPU memory target in GB (default: %(default)s)"
    )
    parser.add_argument("--processes", type=int, default=4, help="Fingerprint/preprocessing worker processes")
    parser.add_argument(
        "--augmentation-workers", type=int, default=4, help="nnU-Net augmentation workers; 0 runs in-process"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA GPU; 0 preserves an existing CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--resume", action="store_true", help="Continue from this trainer/configuration's saved checkpoint"
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Export final validation probabilities as well as segmentations",
    )
    args = parser.parse_args(argv)
    for name in ("download_workers", "processes", "max_train_cases", "max_val_cases"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if not 1 <= args.dataset_id <= 999:
        parser.error("dataset-id must be between 1 and 999")
    if args.augmentation_workers < 0 or args.gpu_id < 0:
        parser.error("augmentation-workers and gpu-id must be nonnegative")
    if not math.isfinite(args.gpu_memory_target) or args.gpu_memory_target <= 0:
        parser.error("gpu-memory-target must be positive and finite")
    if args.modalities and len(set(args.modalities)) != len(args.modalities):
        parser.error("modalities must be unique")
    for name in ("trainer_class_name", "planner", "plans"):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", getattr(args, name)):
            parser.error(f"{name.replace('_', '-')} must be a class/plans identifier")
    if args.resume and (args.prepare_only or args.preprocess_only):
        parser.error("--resume is only supported when training")
    if args.prepare_only and args.skip_preprocessing:
        parser.error("--prepare-only does not run preprocessing")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    manifest_path = args.manifest.resolve() if args.manifest else prepare_dataset(args)
    manifest, partitions = read_manifest(manifest_path, args)
    recipe, splits = export_dataset(args, manifest_path, manifest, partitions)
    print(
        f"Cases: { {role: len(rows) for role, rows in partitions.items()} }; channels: {manifest['channel_order']}; source splits: {split_sources(args)}",
        flush=True,
    )
    if args.prepare_only:
        print(f"nnU-Net raw data: {args.output.resolve() / 'nnUNet_raw' / recipe['signature']['dataset_name']}")
        return
    run_nnunet(args, recipe, splits)


if __name__ == "__main__":
    main()
