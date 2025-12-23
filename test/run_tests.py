#!/usr/bin/env python3
"""
Test runner script for Enzyme-JAX tests using uv.

This script manages separate virtual environments for each test group to avoid
dependency conflicts. Each test group has its own set of dependencies defined
in pyproject.toml.

Usage:
    python run_tests.py [test_group] [options]

Test Groups:
    test        - Basic enzyme tests (test.py)
    bench       - Benchmark tests (bench_vs_xla.py)
    testffi     - FFI tests (testffi.py)
    llama       - LLaMA model tests (llama.py)
    jaxmd       - JAX-MD molecular dynamics tests (jaxmd.py)
    neuralgcm   - NeuralGCM tests (neuralgcm_test.py)
    keras       - Keras tests (keras_test.py)
    all         - Run all tests (each in its own environment)

Options:
    --cuda          Use CUDA (includes xprof)
    --tpu           Use TPU (includes xprof)
    --no-cache      Don't use cached virtual environments
    --list          List available test groups
    --verbose       Verbose output
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

# Test group definitions
TEST_GROUPS = {
    "test": {
        "file": "test.py",
        "extras": [],
        "description": "Basic enzyme tests",
    },
    "bench": {
        "file": "bench_vs_xla.py",
        "extras": [],
        "description": "Benchmark tests against XLA",
    },
    "testffi": {
        "file": "testffi.py",
        "extras": [],
        "description": "FFI tests",
    },
    "llama": {
        "file": "llama.py",
        "extras": [],
        "description": "LLaMA model tests",
    },
    "jaxmd": {
        "file": "jaxmd.py",
        "extras": ["jaxmd"],
        "platform_restriction": "x86_64",
        "description": "JAX-MD molecular dynamics tests",
    },
    "neuralgcm": {
        "file": "neuralgcm_test.py",
        "extras": ["neuralgcm"],
        "platform_restriction": "x86_64",
        "description": "NeuralGCM weather model tests",
    },
    # TODO: keras
    # TODO: maxtext
}


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent.resolve()


def get_venv_dir(test_group: str, backend: str) -> Path:
    """Get the virtual environment directory for a test group."""
    script_dir = get_script_dir()
    suffix = f"-{backend}" if backend != "cpu" else ""
    return script_dir / ".venvs" / f"{test_group}{suffix}"


def is_x86_64() -> bool:
    """Check if running on x86_64 architecture."""
    return platform.machine() == "x86_64"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def check_platform_restriction(test_group: str) -> bool:
    """Check if the test group can run on this platform."""
    config = TEST_GROUPS.get(test_group, {})
    restriction = config.get("platform_restriction")

    if restriction == "x86_64":
        if not is_x86_64():
            return False

    return True


def find_enzyme_wheel() -> tuple[Path | None, str | None]:
    """Find the Bazel-built enzyme_ad wheel and its Python version.

    Returns:
        A tuple of (wheel_path, python_version) where python_version is like "3.11"
    """
    import re

    script_dir = get_script_dir()
    # The bazel-bin directory is at the repo root level
    repo_root = script_dir.parent
    bazel_bin = repo_root / "bazel-bin"

    if not bazel_bin.exists():
        return None, None

    # Look for enzyme_ad wheel
    wheels = list(bazel_bin.glob("enzyme_ad-*.whl"))
    if not wheels:
        return None, None

    # Return the most recently modified wheel
    wheel = max(wheels, key=lambda p: p.stat().st_mtime)

    # Extract Python version from wheel filename (e.g., py311 -> 3.11)
    match = re.search(r"-py(\d)(\d+)-", wheel.name)
    python_version = None
    if match:
        python_version = f"{match.group(1)}.{match.group(2)}"

    return wheel, python_version


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    verbose: bool = False,
    env: dict | None = None,
) -> int:
    """Run a command and return the exit code."""
    if verbose:
        print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=cwd, env=env)
    return result.returncode


def create_venv(
    venv_dir: Path,
    extras: list[str],
    backend: str,
    no_cache: bool,
    verbose: bool,
    enzyme_wheel: Path | None = None,
    python_version: str | None = None,
) -> bool:
    """Create a virtual environment with uv and install dependencies."""
    script_dir = get_script_dir()

    # Find enzyme_ad wheel if not provided
    if enzyme_wheel is None:
        enzyme_wheel, wheel_python_version = find_enzyme_wheel()
        if enzyme_wheel:
            print(f"Found enzyme_ad wheel: {enzyme_wheel}")
            if wheel_python_version:
                python_version = wheel_python_version
                print(f"  Requires Python {python_version}")
        else:
            print("WARNING: No enzyme_ad wheel found in bazel-bin/")
            print("  Build it with: bazel build :wheel")
            print("  Or provide --enzyme-wheel path/to/wheel.whl")
            return False

    if enzyme_wheel and not enzyme_wheel.exists():
        print(f"ERROR: Enzyme wheel not found: {enzyme_wheel}")
        return False

    # Check if venv already exists
    if venv_dir.exists() and not no_cache:
        if verbose:
            print(f"Using existing venv: {venv_dir}")
        return True

    # Create venv
    print(f"Creating virtual environment: {venv_dir}")

    # Remove existing venv if no_cache
    if venv_dir.exists():
        import shutil

        shutil.rmtree(venv_dir)

    # Create venv with uv, using specific Python version if required
    venv_cmd = ["uv", "venv", str(venv_dir)]
    if python_version:
        venv_cmd.extend(["--python", python_version])

    result = run_command(venv_cmd, verbose=verbose)
    if result != 0:
        print("Failed to create virtual environment")
        if python_version:
            print(f"  Make sure Python {python_version} is installed")
        return False

    # Build the pip install command
    install_args = [
        "uv",
        "pip",
        "install",
        "--python",
        str(venv_dir / "bin" / "python"),
    ]

    # Install enzyme-ad wheel first (it's a required dependency)
    if enzyme_wheel:
        install_args.append(str(enzyme_wheel))

    # Build extras string
    all_extras = list(extras)
    if backend == "cuda":
        all_extras.append("cuda")
    elif backend == "tpu":
        all_extras.append("tpu")

    # Install from pyproject.toml with extras
    if all_extras:
        extras_str = ",".join(all_extras)
        install_args.append(f"-e{script_dir}[{extras_str}]")
    else:
        install_args.append(f"-e{script_dir}")

    result = run_command(install_args, verbose=verbose)
    if result != 0:
        print("Failed to install dependencies")
        return False

    return True


def run_test(
    test_group: str,
    backend: str,
    no_cache: bool,
    verbose: bool,
    enzyme_wheel: Path | None = None,
    extra_args: list[str] | None = None,
    output_dir: Path | None = None,
) -> int:
    """Run a test group."""
    config = TEST_GROUPS.get(test_group)
    if not config:
        print(f"Unknown test group: {test_group}")
        return 1

    # Check platform restrictions
    if not check_platform_restriction(test_group):
        print(f"Skipping {test_group}: not supported on {platform.machine()}")
        return 0

    # Get extras for this test group
    extras = config.get("extras", [])

    # Get venv directory
    venv_dir = get_venv_dir(test_group, backend)

    # Create venv
    if not create_venv(venv_dir, extras, backend, no_cache, verbose, enzyme_wheel):
        return 1

    # Run the test
    script_dir = get_script_dir()
    test_file = script_dir / config["file"]
    python_path = venv_dir / "bin" / "python"

    cmd = [str(python_path), str(test_file)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"Running test group: {test_group}")
    print(f"Test file: {test_file}")
    print(f"Backend: {backend}")
    print(f"{'=' * 60}\n")

    # Set up environment for benchmark outputs
    import os
    env = os.environ.copy()
    
    # Set up output directory for benchmark results
    if output_dir is None:
        output_dir = script_dir / ".benchmark-outputs" / f"{test_group}-{backend}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env["TEST_UNDECLARED_OUTPUTS_DIR"] = str(output_dir)
    
    if verbose:
        print(f"Benchmark outputs will be saved to: {output_dir}")

    return run_command(cmd, cwd=script_dir, verbose=verbose, env=env)


def list_test_groups():
    """List available test groups."""
    print("\nAvailable test groups:")
    print("-" * 60)
    for name, config in TEST_GROUPS.items():
        restriction = config.get("platform_restriction", "all")
        extras = ", ".join(config.get("extras", [])) or "none"
        available = "✓" if check_platform_restriction(name) else "✗"
        print(f"  {name:12s} - {config['description']}")
        print(
            f"               Platform: {restriction}, Extras: {extras}, Available: {available}"
        )
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run Enzyme-JAX tests using uv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "test_group",
        nargs="?",
        default="all",
        help="Test group to run (default: all)",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA backend (includes xprof)",
    )
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Use TPU backend (includes xprof)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use cached virtual environments",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available test groups",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--enzyme-wheel",
        type=Path,
        help="Path to enzyme-ad wheel to install",
    )
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Extra arguments to pass to the test script",
    )

    args = parser.parse_args()

    if args.list:
        list_test_groups()
        return 0

    # Determine backend
    if args.cuda and args.tpu:
        print("ERROR: Cannot specify both --cuda and --tpu")
        return 1
    elif args.cuda:
        backend = "cuda"
    elif args.tpu:
        backend = "tpu"
    else:
        backend = "cpu"

    test_groups = (
        list(TEST_GROUPS.keys()) if args.test_group == "all" else [args.test_group]
    )

    failed = []
    for test_group in test_groups:
        if test_group not in TEST_GROUPS:
            print(f"Unknown test group: {test_group}")
            failed.append(test_group)
            continue

        result = run_test(
            test_group,
            backend,
            args.no_cache,
            args.verbose,
            args.enzyme_wheel,
            args.extra_args,
        )

        if result != 0:
            failed.append(test_group)

    if failed:
        print(f"\n{'=' * 60}")
        print(f"Failed test groups: {', '.join(failed)}")
        print(f"{'=' * 60}")
        return 1

    print(f"\n{'=' * 60}")
    print("All tests passed!")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
