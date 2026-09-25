#!/usr/bin/env python3
"""
Removes PyTorch's CUDA dependencies from a `uv pip compile` lock file.

The CPU images install the CPU build of PyTorch from the PyTorch index, whose Linux
wheels on PyPI depend on several GB of NVIDIA CUDA libraries. This keeps exactly the
packages reachable from the requirements files (following uv's "# via" annotations),
except that below the SEPARATE packages (removed from the lock, installed on their own)
dependencies matching a DROP pattern are not followed.

    prune_lock.py LOCKFILE --separate torch torchvision --drop 'nvidia-*' 'cuda-*' triton
"""
import argparse
import re


def parse_blocks(text):
    """Splits a lock file into [name, lines] blocks; the header block has name None."""
    blocks, current = [], [None, []]
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?==", line)
        if match:
            blocks.append(current)
            current = [normalize(match.group(1)), [line]]
        else:
            current[1].append(line)
    blocks.append(current)
    return blocks


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parents_of(lines):
    """Entries of the block's "# via" comment (single- or multi-line form)."""
    parents, in_via = set(), False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# via"):
            in_via = True
            rest = stripped[len("# via"):].strip()
            if rest:
                parents.add(rest)
        elif in_via and stripped.startswith("#   "):
            parents.add(stripped[4:].strip())
        elif in_via:
            in_via = False
    return {p if p.startswith("-r ") else normalize(p) for p in parents}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lockfile")
    parser.add_argument("--separate", nargs="*", default=[])
    parser.add_argument("--drop", nargs="*", default=[])
    args = parser.parse_args()
    separate = {normalize(p) for p in args.separate}
    drop = [p.lower() for p in args.drop]

    def dropped_below_separate(name):
        return any(name == p or (p.endswith("*") and name.startswith(p[:-1])) for p in drop)

    blocks = parse_blocks(open(args.lockfile, encoding="utf8").read())
    parents = {name: parents_of(lines) for name, lines in blocks if name}
    children = {name: set() for name in parents}
    for name, names in parents.items():
        for parent in names:
            if parent in children:
                children[parent].add(name)

    reachable = set()
    stack = [name for name, names in parents.items() if any(p.startswith("-r ") for p in names)]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        for child in children[name]:
            if name in separate and dropped_below_separate(child):
                continue
            stack.append(child)

    keep = reachable - separate
    removed = sorted(set(parents) - keep)
    with open(args.lockfile, "w", encoding="utf8") as f:
        f.write("\n".join(line for name, lines in blocks if name is None or name in keep for line in lines) + "\n")
    print(f"{args.lockfile}: removed {', '.join(removed)}")


if __name__ == "__main__":
    main()
