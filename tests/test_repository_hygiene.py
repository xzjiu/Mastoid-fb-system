"""Repository-level safeguards against publishing local or study data."""

from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024

FORBIDDEN_FILE_SUFFIXES = (
    ".avi",
    ".db",
    ".dcm",
    ".dll",
    ".dylib",
    ".exe",
    ".h5",
    ".hdf5",
    ".m4a",
    ".mov",
    ".mp3",
    ".mp4",
    ".nii",
    ".nii.gz",
    ".npy",
    ".npz",
    ".ogg",
    ".pickle",
    ".pkl",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".wav",
    ".webm",
    ".zip",
)

WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/][^\\/\s]+)"
)
POSIX_USER_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:Users|home|mnt|Volumes)/[^\s\"']+"
)
CODED_PARTICIPANT_TRIAL_ID = re.compile(
    r"(?<![A-Za-z0-9.])P[0-9]{2}(?:_T[0-9]{2})?\b", re.IGNORECASE
)


def repository_candidates() -> list[Path]:
    """Return tracked and non-ignored untracked files as Git would see them."""

    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Unable to enumerate repository candidates with Git: "
            + result.stderr.decode("utf-8", errors="replace")
        )

    relative_paths = result.stdout.decode("utf-8").split("\0")
    return [
        REPOSITORY_ROOT.joinpath(*PurePosixPath(value).parts)
        for value in relative_paths
        if value
    ]


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


class RepositoryHygieneTests(unittest.TestCase):
    def test_local_data_and_result_directories_are_ignored(self) -> None:
        probes = (
            "config/local/.repository-hygiene-probe",
            "data/.repository-hygiene-probe",
            "results/.repository-hygiene-probe",
        )
        missing = []
        for probe in probes:
            result = subprocess.run(
                ["git", "check-ignore", "--quiet", "--", probe],
                cwd=REPOSITORY_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if result.returncode != 0:
                missing.append(probe)
        self.assertFalse(
            missing,
            "The following local/generated paths are not protected by "
            f".gitignore: {', '.join(missing)}",
        )

    def test_candidate_tree_contains_only_publishable_source_files(self) -> None:
        problems: list[str] = []
        for path in repository_candidates():
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            lower_name = path.name.lower()

            if path.is_symlink():
                problems.append(f"{relative}: symbolic links are not allowed")
                continue
            if not path.is_file():
                continue
            if any(lower_name.endswith(suffix) for suffix in FORBIDDEN_FILE_SUFFIXES):
                problems.append(f"{relative}: forbidden binary/data extension")
                continue
            if path.stat().st_size > MAX_SOURCE_FILE_BYTES:
                problems.append(
                    f"{relative}: exceeds the {MAX_SOURCE_FILE_BYTES}-byte source limit"
                )
                continue
            if CODED_PARTICIPANT_TRIAL_ID.search(relative):
                problems.append(f"{relative}: coded participant/trial ID in path")

            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                problems.append(f"{relative}: file is not UTF-8 text")
                continue

            for label, pattern in (
                ("local Windows/UNC absolute path", WINDOWS_ABSOLUTE_PATH),
                ("local POSIX user absolute path", POSIX_USER_ABSOLUTE_PATH),
                ("coded participant/trial ID", CODED_PARTICIPANT_TRIAL_ID),
            ):
                match = pattern.search(text)
                if match:
                    problems.append(
                        f"{relative}:{line_number(text, match.start())}: {label}"
                    )

        self.assertFalse(
            problems,
            "Repository hygiene violations:\n" + "\n".join(sorted(problems)),
        )


if __name__ == "__main__":
    unittest.main()
