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
    def __init__(self, name, requires):
        self.name = name
        self.requires = requires  # list of (pattern, expected_count)


def parse_kernels(source: Path):
    kernels = []
    pending_requires = []
    arch = None

    for line in source.read_text().splitlines():
        if m := CHECK_COUNT_RE.search(line):
            pending_requires.append((m.group(2).strip(), int(m.group(1))))
        elif m := CHECK_RE.search(line):
            pending_requires.append((m.group(1).strip(), 1))
        elif m := CHECK_NOT_RE.search(line):
            pending_requires.append((m.group(1).strip(), 0))
        elif m := ARCH_RE.search(line):
            arch = m.group(1)
        elif m := KERNEL_RE.search(line):
            kernels.append(Kernel(m.group(1), pending_requires))
            pending_requires = []

    return kernels, arch


def get_ptx(nvcc, source, arch, verbose=False):
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

    if verbose:
        print(f"$ {' '.join(cmd)}", file=sys.stderr)

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


def check_file(
    source: Path,
    nvcc: str,
    arch: str,
    verbose: bool = False,
    kernel_filter: "set[str] | None" = None,
) -> "tuple[int, int, int]":
    """Returns (num_passed, num_failed, num_skipped) for this source file."""
    all_kernels, arch_directive = parse_kernels(source)
    arch = arch_directive or arch

    checkable = [k for k in all_kernels if k.requires]
    if not checkable:
        print(
            f"error: {source} has no kernel preceded by "
            "// CHECK:, // CHECK-COUNT-<N>:, or // CHECK-NOT: directives",
            file=sys.stderr,
        )
        return 0, 1, 0

    # Kernels with no directives at all are skipped silently; they were never meant to be
    # checked. Kernels excluded by --kernel are also skipped, but reported below.
    num_skipped = len(all_kernels) - len(checkable)

    if kernel_filter is not None:
        kernels = [k for k in checkable if k.name in kernel_filter]
        num_skipped += len(checkable) - len(kernels)
        for k in checkable:
            if k.name not in kernel_filter:
                print(f"SKIP: {k.name} (excluded by --kernel)")
    else:
        kernels = checkable

    if not kernels:
        # Nothing in this file matches --kernel; not an error, just nothing to do here.
        return 0, 0, num_skipped

    ptx = get_ptx(nvcc, source, arch, verbose=verbose)
    if ptx is None:
        print(f"FAIL: {source}: nvcc failed to compile this file", file=sys.stderr)
        return 0, len(kernels), num_skipped

    num_passed = 0
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
            actual_count = len(re.findall(pattern, block, re.MULTILINE))
            ok = (
                actual_count >= 1
                if expected_count is None
                else actual_count == expected_count
            )
            want = "at least 1" if expected_count is None else str(expected_count)

            if verbose:
                status = "success" if ok else "FAILURE"
                print(
                    f"[{kernel.name}] {status} (want {want}, found {actual_count}): {pattern!r}",
                    file=sys.stderr,
                )
            if not ok:
                failures.append(
                    f"pattern found {actual_count} time(s), expected {want}: {pattern!r}"
                )

        if failures:
            num_failed += 1
            for msg in failures:
                print(f"FAIL: {kernel.name}: {msg}", file=sys.stderr)
            print(
                f"----- generated ptx for {kernel.name} ({arch}) -----", file=sys.stderr
            )
            print(block, file=sys.stderr)
        else:
            num_passed += 1
            print(f"OK: {kernel.name} (ptx, {arch})")
            if verbose:
                print(
                    f"----- generated ptx for {kernel.name} ({arch}) -----",
                    file=sys.stderr,
                )
                print(block, file=sys.stderr)

    return num_passed, num_failed, num_skipped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", type=Path, nargs="+")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the nvcc command, per-pattern match results, and the generated PTX "
        "for every kernel, not just the ones that fail",
    )
    parser.add_argument(
        "--arch", help="used unless overridden by a // ARCH: directive", default="sm_80"
    )
    parser.add_argument(
        "--kernel",
        action="append",
        help="only check the kernel with this name, skipping all others (repeatable)",
    )
    args = parser.parse_args()

    kernel_filter = set(args.kernel) if args.kernel else None

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for source in args.source:
        print(f"=== {source} ===")
        num_passed, num_failed, num_skipped = check_file(
            source,
            args.nvcc,
            args.arch,
            verbose=args.verbose,
            kernel_filter=kernel_filter,
        )
        total_passed += num_passed
        total_failed += num_failed
        total_skipped += num_skipped

    if kernel_filter is not None and total_passed + total_failed == 0:
        print(
            f"error: no kernel named {sorted(kernel_filter)} found in any source file",
            file=sys.stderr,
        )
        sys.exit(1)

    total = total_passed + total_failed + total_skipped
    print(
        f"RESULT: {total_passed} passed, {total_failed} failed, {total_skipped} skipped ({total} total)"
    )

    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
