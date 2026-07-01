"""Shared test fixtures."""

import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pytest

TEST_FILES = "https://zenodo.org/records/20843067/files/test_suite.zip?download=1"


@pytest.fixture(scope="session")
def module_path():
    """Parent directory of the project."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def user_path() -> Path:
    """Download and unzip test files."""
    user_dir = Path("resources/user/")
    # If test suite has been downloaded, assume everything is OK.
    # Otherwise, cleanup and re-download.
    if not Path(user_dir / "test_suite.zip").exists():
        shutil.rmtree(user_dir, ignore_errors=True)
        Path(user_dir).mkdir(parents=True, exist_ok=True)
        test_zip = Path(user_dir / "test_suite.zip")
        urlretrieve(TEST_FILES, test_zip)
        with zipfile.ZipFile(test_zip, "r") as zfile:
            zfile.extractall(user_dir)
    return user_dir


@pytest.fixture(scope="session")
def integration_path(user_path: Path, module_path: Path):
    """Ensures the minimal integration test is ready."""
    integration_dir = Path(module_path / "tests/integration")
    if integration_dir.exists():
        # clean everything
        shutil.rmtree(integration_dir / "resources", ignore_errors=True)
        shutil.rmtree(integration_dir / "results/", ignore_errors=True)
    user_integ_dir = integration_dir / "resources/inputs/"
    files_to_copy = ["MNE/proxies/rooftop_pv.tif", "MNE/shapes.parquet"]
    for file in files_to_copy:
        destination_file = Path(user_integ_dir / file)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(user_path / file, destination_file)
    return integration_dir
