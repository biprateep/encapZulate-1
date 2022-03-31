import os
from pathlib import Path
import subprocess


def git_commit_hash(levels):
    """Get git commit hash.

    Args:
        levels (int): Levels from file with function call to top level
            of package.

    Returns:
        str: git commit hash
    """
    cwd = Path.cwd()
    os.chdir(str(Path(__file__).resolve().parents[levels]))
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    commit_hash = commit_hash.strip().decode("utf8")
    os.chdir(cwd)
    return commit_hash


def import_model(model_name):
    """Import model."""
    module_name = ".".join(["encapzulate", "models", model_name])
    _tmp = __import__(module_name, fromlist=["CapsNet"])
    return _tmp.CapsNet
