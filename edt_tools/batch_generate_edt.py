#!/usr/bin/env python3
"""Generate one EDT directory per volume using an existing EDTFromGrid binary.

Uses only the Python standard library. PNG slice order and anatomy colors follow
hisashiishida/SDF_based_assistance scripts/EdtGeneration at commit 3c4736bb.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import tempfile


ANATOMY = {
    "Bone": (255, 249, 219),
    "Malleus": (233, 0, 255),
    "Incus": (0, 255, 149),
    "Stapes": (63, 0, 255),
    "Bony_Labyrinth": (91, 123, 91),
    "IAC": (244, 142, 52),
    "Superior_Vestibular_Nerve": (255, 191, 135),
    "Inferior_Vestibular_Nerve": (121, 70, 24),
    "Cochlear_Nerve": (219, 244, 52),
    "Facial_Nerve": (244, 214, 49),
    "Chorda_Tympani": (151, 131, 29),
    "ICA": (216, 100, 79),
    "Sinus_+_Dura": (110, 184, 209),
    "Vestibular_Aqueduct": (91, 98, 123),
    "TMJ": (100, 0, 0),
    "EAC": (255, 225, 214),
}


def slices_in(folder):
    """Match the upstream script: order PNGs by their first numeric component."""
    numbered = []
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() == ".png":
            match = re.search(r"\d+", path.name)
            if match:
                numbered.append((int(match.group()), path.resolve()))
    numbered.sort(key=lambda item: item[0])
    indices = [index for index, _ in numbered]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate slice numbers; check PNG naming before generation")
    return [path for _, path in numbered]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exe", required=True, type=Path)
    args = parser.parse_args()
    source = args.volumes.expanduser().resolve()
    output = args.output.expanduser().resolve()
    executable = args.exe.expanduser().resolve()
    if not source.is_dir():
        parser.error("--volumes must be an existing directory")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error("--exe must point to an executable EDTFromGrid")
    if output == source or source in output.parents or output in source.parents:
        parser.error("Use separate, non-nested volumes and output directories")

    folders = sorted(path for path in source.iterdir() if path.is_dir())
    generated = skipped = failed = volume_count = 0
    for folder in folders:
        try:
            images = slices_in(folder)
        except ValueError as error:
            print(f"ERROR {folder.name}: {error}", flush=True)
            failed += 1
            continue
        if not images:
            print(f"SKIP {folder.name}: no numbered PNG slices", flush=True)
            continue
        volume_count += 1
        destination = output / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        print(f"VOLUME {folder.name}: {len(images)} slices -> {destination}", flush=True)

        # Temporary output prevents failed runs from leaving a final .edt file.
        with tempfile.TemporaryDirectory(prefix=".edt-build-", dir=destination) as tmp:
            temporary = Path(tmp)
            image_list = temporary / "slices.txt"
            image_list.write_text(
                "".join(str(path) + "\n" for path in images), encoding="utf-8"
            )
            for anatomy, rgb in ANATOMY.items():
                target = destination / f"{anatomy}.edt"
                if target.exists():
                    print(f"  SKIP existing {target.name} (not revalidated)", flush=True)
                    skipped += 1
                    continue
                pending = temporary / f"{anatomy}.edt"
                log_path = destination / f"{anatomy}.log"
                print(f"  GENERATE {anatomy}", flush=True)
                command = [
                    str(executable), "--in", str(image_list),
                    "--id", *map(str, rgb), "--out", str(pending),
                ]
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode != 0 or not pending.is_file() or pending.stat().st_size == 0:
                    print(f"  FAILED {anatomy}; see {log_path}", flush=True)
                    failed += 1
                    continue
                pending.rename(target)
                generated += 1

    print(f"Finished: {volume_count} volumes, {generated} generated, {skipped} skipped, {failed} failed.")
    return 1 if failed or volume_count == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
