#!/usr/bin/env python3
r"""
Compile one or more CUDA files, each containing multiple `__global__` kernels, and check, per
kernel, that specific instructions are (or are not) present in its generated PTX. This checks
for code-generation behavior that unit tests cannot observe: e.g. that a load actually gets
vectorized, or that a "fast" math function really lowers to the expected approximate PTX
instruction rather than silently falling back to the accurate one.

Expected patterns are written directly above each kernel in the .cu source file as
comment directives:

    // CHECK: <pattern>
        The pattern must appear at least once in the kernel's generated PTX.

    // CHECK-COUNT-<N>: <pattern>
        The pattern must appear in the kernel's generated PTX exactly <N> times.

    // CHECK-NOT: <pattern>
        The pattern must not appear anywhere in the kernel's generated PTX.

    // ARCH: <sm_NN>
        Overrides --arch for this file only.

A block of directives applies to the single `__global__` kernel immediately below it; each
<pattern> is a plain Python regex, matched only against that kernel's own PTX, not the whole
file. A kernel with no directives above it is not checked (and not counted).

Usage:
    # Check one file
    python3 verify_codegen.py half_ops.cu

    # Check every .cu file in this directory (the shell expands the glob, not this script)
    python3 verify_codegen.py *.cu

    # Use a specific nvcc and target architecture
    python3 verify_codegen.py *.cu --nvcc /usr/local/cuda-12.6/bin/nvcc --arch sm_86

Exit status is 0 if every kernel's checks passed, 1 otherwise (including if a file fails to
compile, or contains no directives at all).
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CHECK_RE = re.compile(r"//\s*CHECK:\s*(.+)$")
CHECK_COUNT_RE = re.compile(r"//\s*CHECK-COUNT-(\d+):\s*(.+)$")
CHECK_NOT_RE = re.compile(r"//\s*CHECK-NOT:\s*(.+)$")
ARCH_RE = re.compile(r"//\s*ARCH:\s*(\S+)\s*$")
KERNEL_RE = re.compile(r"__global__\s+[\w:<>,\s\*&]+?\s(\w+)\s*\(")
PTX_ENTRY_RE = re.compile(r"^\.visible\s+\.entry\s+(\S+)\(")


class Kernel:
    def __init__(self, name):
        self.name = name
        self.requires = []  # list of (pattern, expected_count)
        self.forbids = []


def parse_kernels(source: Path):
    kernels = []
    pending_requires, pending_forbids = [], []
    arch = None

    for line in source.read_text().splitlines():
        if m := CHECK_COUNT_RE.search(line):
            pending_requires.append((m.group(2).strip(), int(m.group(1))))
        elif m := CHECK_RE.search(line):
            # `None` means "at least once" (exact count is not checked)
            pending_requires.append((m.group(1).strip(), 1))
        elif m := CHECK_NOT_RE.search(line):
            pending_forbids.append(m.group(1).strip())
        elif m := ARCH_RE.search(line):
            arch = m.group(1)
        elif m := KERNEL_RE.search(line):
            kernel = Kernel(m.group(1))
            kernel.requires = pending_requires
            kernel.forbids = pending_forbids
            kernels.append(kernel)
            pending_requires, pending_forbids = [], []

    return kernels, arch


def get_ptx(nvcc, source, arch):
    kf_root = Path(__file__).parent.parent.resolve() / "include"
    cmd = [
        nvcc,
        "-ptx",
        f"-arch={arch}",
        "-std=c++17",
        "-I",
        str(kf_root),
        "-o",
        "-",
        str(source),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"command failed: {' '.join(cmd)}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return None

    return result.stdout


def extract_ptx_block(ptx: str, kernel_name: str) -> str:
    lines = ptx.splitlines()
    for i, line in enumerate(lines):
        m = PTX_ENTRY_RE.match(line)
        if m and kernel_name == m.group(1):
            depth = 0
            visited_body = False

            for j in range(i + 1, len(lines)):
                depth += lines[j].count("{")
                depth -= lines[j].count("}")
                visited_body |= depth > 0

                if visited_body and depth == 0:
                    return "\n".join(lines[i : j + 1])
            return "\n".join(lines[i:])
    return None


def check_file(source: Path, nvcc: str, arch: str) -> "tuple[int, int]":
    """Returns (num_kernels, num_failed) for this source file."""
    kernels, arch_directive = parse_kernels(source)
    arch = arch_directive or arch

    kernels = [k for k in kernels if k.requires or k.forbids]
    if not kernels:
        print(
            f"error: {source} has no kernel preceded by "
            "// CHECK:, // CHECK-COUNT-<N>:, or // CHECK-NOT: directives",
            file=sys.stderr,
        )
        return 0, 1

    ptx = get_ptx(nvcc, source, arch)
    if ptx is None:
        print(f"FAIL: {source}: nvcc failed to compile this file", file=sys.stderr)
        return len(kernels), len(kernels)

    num_failed = 0

    for kernel in kernels:
        block = extract_ptx_block(ptx, kernel.name)

        if block is None:
            print(
                f"FAIL: {kernel.name}: could not find generated ptx for this kernel",
                file=sys.stderr,
            )
            num_failed += 1
            continue

        failures = []
        for pattern, expected_count in kernel.requires:
            if expected_count is None:
                if not re.search(pattern, block, re.MULTILINE):
                    failures.append(f"CHECK pattern not found: {pattern!r}")
            else:
                actual_count = len(re.findall(pattern, block, re.MULTILINE))
                if actual_count != expected_count:
                    failures.append(
                        f"CHECK pattern found {actual_count} time(s), expected {expected_count}: {pattern!r}"
                    )
        for pattern in kernel.forbids:
            if re.search(pattern, block, re.MULTILINE):
                failures.append(f"CHECK-NOT pattern found: {pattern!r}")

        if failures:
            num_failed += 1
            for msg in failures:
                print(f"FAIL: {kernel.name}: {msg}", file=sys.stderr)
            print(
                f"----- generated ptx for {kernel.name} ({arch}) -----", file=sys.stderr
            )
            print(block, file=sys.stderr)
        else:
            print(f"OK: {kernel.name} (ptx, {arch})")

    return len(kernels), num_failed


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, nargs="+")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--arch", help="used unless overridden by a // ARCH: directive", default="sm_80"
    )
    args = parser.parse_args()

    total_kernels = 0
    total_failed = 0

    for source in args.source:
        print(f"=== {source} ===")
        num_kernels, num_failed = check_file(source, args.nvcc, args.arch)
        total_kernels += num_kernels
        total_failed += num_failed

    print(f"TEST PASSED: {total_kernels - total_failed} / {total_kernels}")

    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
