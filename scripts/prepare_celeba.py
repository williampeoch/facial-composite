#!/usr/bin/env python3
"""Normalize CelebA dataset layout for this project.

Expected runtime layout:
- <target>/list_attr_celeba.csv
- <target>/img_align_celeba/img_align_celeba/*.jpg (or equivalent)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CSV_NAMES = [
    "list_attr_celeba.csv",
    "list_bbox_celeba.csv",
    "list_eval_partition.csv",
    "list_landmarks_align_celeba.csv",
]


def find_existing_file(root: Path, file_name: str) -> Path | None:
    """Return the first file named ``file_name`` under ``root``, if any."""

    direct = root / file_name
    if direct.is_file():
        return direct

    for candidate in root.rglob(file_name):
        if candidate.is_file():
            return candidate
    return None


def find_images_dir(root: Path) -> Path | None:
    """Locate a CelebA image directory containing JPG files."""

    candidates = [
        root / "img_align_celeba" / "img_align_celeba",
        root / "img_align_celeba",
        root / "celeba" / "img_align_celeba" / "img_align_celeba",
        root / "celeba" / "img_align_celeba",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate

    for candidate in root.rglob("img_align_celeba"):
        if candidate.is_dir() and any(candidate.glob("*.jpg")):
            return candidate

    return None


def copy_csv_files(source_root: Path, target_root: Path) -> None:
    """Copy expected CelebA CSV metadata files from source to target."""

    for csv_name in CSV_NAMES:
        src = find_existing_file(source_root, csv_name)
        if not src:
            if csv_name == "list_attr_celeba.csv":
                raise FileNotFoundError(
                    "Le fichier obligatoire 'list_attr_celeba.csv' est introuvable dans la source."
                )
            continue

        dst = target_root / csv_name
        if dst.resolve() == src.resolve():
            continue
        shutil.copy2(src, dst)


def link_or_copy_images(source_images_dir: Path, target_images_dir: Path, copy_images: bool) -> None:
    """Create a symlink to images or copy them when symlink creation is unavailable."""

    target_images_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_images_dir.exists() or target_images_dir.is_symlink():
        if target_images_dir.is_symlink() and target_images_dir.resolve() == source_images_dir.resolve():
            return
        if target_images_dir.is_dir() and any(target_images_dir.glob("*.jpg")):
            return
        raise FileExistsError(
            f"Le dossier cible existe deja: {target_images_dir}. Supprime-le ou choisis un autre --target."
        )

    try:
        target_images_dir.symlink_to(source_images_dir, target_is_directory=True)
        print(f"Lien symbolique cree: {target_images_dir} -> {source_images_dir}")
        return
    except OSError as exc:
        if not copy_images:
            raise RuntimeError(
                "Impossible de creer le lien symbolique. Relance avec --copy-images pour copier les images."
            ) from exc

    print("Copie des images en cours (cela peut prendre du temps)...")
    shutil.copytree(source_images_dir, target_images_dir)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset preparation."""

    parser = argparse.ArgumentParser(description="Prepare CelebA dataset structure for this project")
    parser.add_argument(
        "--source",
        default="data/celeba-dataset",
        help="Chemin source du dataset CelebA decompresse",
    )
    parser.add_argument(
        "--target",
        default="celeba",
        help="Chemin cible du dataset dans le projet",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copie les images au lieu de creer un lien symbolique",
    )
    return parser.parse_args()


def main() -> int:
    """Prepare the dataset layout and return a process exit code."""

    args = parse_args()

    source_root = Path(args.source).expanduser().resolve()
    target_root = Path(args.target).expanduser().resolve()

    if not source_root.exists():
        print(f"Source introuvable: {source_root}", file=sys.stderr)
        return 1

    target_root.mkdir(parents=True, exist_ok=True)

    try:
        copy_csv_files(source_root, target_root)
    except Exception as exc:
        print(f"Erreur CSV: {exc}", file=sys.stderr)
        return 1

    source_images_dir = find_images_dir(source_root)
    if source_images_dir is None:
        print(
            "Impossible de trouver les images. Attendu: dossier 'img_align_celeba' contenant des .jpg.",
            file=sys.stderr,
        )
        return 1

    target_images_dir = target_root / "img_align_celeba" / "img_align_celeba"

    try:
        link_or_copy_images(source_images_dir, target_images_dir, copy_images=args.copy_images)
    except Exception as exc:
        print(f"Erreur images: {exc}", file=sys.stderr)
        return 1

    print("Preparation terminee.")
    print(f"Dataset cible: {target_root}")
    print("Tu peux maintenant lancer: python UI_v5.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
