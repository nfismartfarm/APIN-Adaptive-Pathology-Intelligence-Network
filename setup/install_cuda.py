# setup/install_cuda.py
"""
Step 1: CUDA 12.1 installation and verification on Windows.
[FIX GAP 72] input() wrapped in try/except EOFError — treats EOF as --yes.
[FIX GAP 51] --yes flag skips all interactive prompts.
"""

import sys
import os
import subprocess
import argparse


def prompt(message, yes_flag):
    """[FIX GAP 72] Prompt for confirmation. EOF treated as --yes."""
    if yes_flag:
        return 'yes'
    try:
        return input(message).strip().lower()
    except EOFError:
        # Running in non-interactive subprocess context — treat as --yes
        print("(EOF detected — running non-interactively, assuming yes)")
        return 'yes'


def check_cuda_available():
    result = subprocess.run(
        [sys.executable, '-c',
         'import torch; print(torch.cuda.is_available()); '
         'print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        available = lines[0].strip() == 'True'
        name      = lines[1].strip() if len(lines) > 1 else 'unknown'
        return available, name
    return False, 'torch not installed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true',
                        help='Skip all interactive prompts')
    args = parser.parse_args()

    print("=" * 50)
    print("STEP 01 — CUDA INSTALLATION")
    print("=" * 50)

    if sys.platform != 'win32':
        print("This script targets Windows 11. On Linux, install CUDA manually.")
        sys.exit(0)

    # Step A: check if CUDA already works
    available, name = check_cuda_available()
    if available and 'RTX 4060' in name or 'NVIDIA' in name:
        print(f"  ✓ CUDA already working: {name}")
        print("Step 01 complete.")
        sys.exit(0)

    print("CUDA not available. Starting installation...")
    print("This requires Administrator privileges for some steps.")
    print("Follow these steps manually:")
    print()
    print("Step B — Run in Command Prompt as Administrator:")
    print("  nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    print("  If version < 525: download driver from nvidia.com/drivers")
    print()
    print("Step C — Install CUDA Toolkit 12.1:")
    print("  URL: developer.download.nvidia.com/compute/cuda/12.1.0/"
          "network_installers/cuda_12.1.0_windows_network.exe")
    print("  Run as Administrator. Choose Express. Restart Windows after.")

    ans = prompt("\nHave you installed CUDA 12.1? Type 'yes' to continue: ",
                 args.yes)

    print()
    print("Step D — Install PyTorch with CUDA 12.1:")
    print("  pip install torch==2.2.0 torchvision==0.17.0 --index-url "
          "https://download.pytorch.org/whl/cu121")

    ans = prompt("\nHave you installed PyTorch? Type 'yes' to continue: ",
                 args.yes)

    # Verify
    available, name = check_cuda_available()
    if available:
        print(f"\n  ✓ CUDA verified: {name}")
        print("Step 01 complete.")
        sys.exit(0)
    else:
        print(f"\n  ✗ CUDA still not available. {name}")
        print("See Section 13.3 in CLAUDE.md for troubleshooting.")
        sys.exit(1)


if __name__ == '__main__':
    main()
