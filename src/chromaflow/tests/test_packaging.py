from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path


def _run(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (
            f"Command failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
        raise AssertionError(message)
    return result.stdout


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def test_built_wheel_installs_in_clean_virtualenv(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()

    _run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "-w", str(wheelhouse)],
        cwd=project_root,
    )

    wheel = next(wheelhouse.glob("chromaflow-*.whl"))
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)

    _run([str(python), "-m", "pip", "install", "numpy>=1.21", str(wheel)])

    script = textwrap.dedent(
        """
        import pkgutil
        import chromaflow

        assert chromaflow.__version__ == "0.3.0"
        data = pkgutil.get_data("chromaflow", "data/cie_1931_2deg_locus.csv")
        assert data is not None
        print("ok")
        """
    )
    output = _run([str(python), "-c", script])
    assert "ok" in output


def test_source_tree_installs_in_clean_virtualenv(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)

    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "numpy>=1.21",
            "--no-deps",
            str(project_root),
        ]
    )

    script = textwrap.dedent(
        """
        import pkgutil
        import chromaflow

        assert chromaflow.__version__ == "0.3.0"
        data = pkgutil.get_data("chromaflow", "data/cie_1931_2deg_locus.csv")
        assert data is not None
        print("ok")
        """
    )
    output = _run([str(python), "-c", script])
    assert "ok" in output
