"""Delay sweep analysis."""

import json
import os

import numpy as np

DATA = "delay-sweep/baseline"
VIOLATIONS = "data/violations.json"
SUMMARY = "summary"


def load_beanstalk(path=VIOLATIONS):
    """Distinct violation pairs found by beanstalk, per benchmark."""
    with open(path) as f:
        data = json.load(f)
    return {name: {tuple(p) for p in pairs} for name, pairs in data.items()}


def poster_order(base=SUMMARY, k_threshold=0):
    """Poster numbering, mirroring plot/poster.py.

    Flattens every (benchmark, site) across the summary dataset, keeps sites
    with total detections > ``k_threshold``, sorts globally by heisen factor F
    (ascending) and numbers them from 1. Returns ``{(benchmark, pair): number}``.
    """
    npz = {os.path.splitext(f)[0]: np.load(os.path.join(base, f))
           for f in os.listdir(base) if f.endswith(".npz")}
    entries = []  # (F, benchmark, pair)
    for name, v in npz.items():
        for i in range(v["K"].shape[-1]):
            if np.sum(v["K"][:, :, i]) > k_threshold:
                entries.append((v["F"][i], name, tuple(int(x) for x in v["sites"][i])))
    order = np.argsort([f for f, _, _ in entries])
    return {(entries[idx][1], entries[idx][2]): num
            for num, idx in enumerate(order, start=1)}


def load(base=DATA):
    """Load each benchmark's aggregated .npz from the delay-sweep dataset."""
    return {
        os.path.splitext(f)[0]: np.load(os.path.join(base, f))
        for f in sorted(os.listdir(base)) if f.endswith(".npz")
    }


def run_counts(npz):
    """Number of runs per delay, per benchmark."""
    return {
        name: dict(zip(*np.unique(v["delay"], return_counts=True)))
        for name, v in npz.items()
    }


def unpack_bugs(v):
    """Unpack a benchmark's per-run bug bit-vectors to shape (n_runs, n_sites)."""
    n_sites = v["sites"].shape[0]
    return np.unpackbits(v["bugs"], axis=-1)[:, :n_sites].astype(bool)


def bugs_caught(npz):
    """Distinct bugs caught per delay, per benchmark (union of runs at that delay)."""
    out = {}
    for name, v in npz.items():
        bugs = unpack_bugs(v)
        delay = v["delay"]
        out[name] = {
            int(d): int(bugs[delay == d].any(axis=0).sum())
            for d in np.unique(delay)
        }
    return out


if __name__ == "__main__":
    npz = load()
    counts = run_counts(npz)
    caught = bugs_caught(npz)

    delays = sorted({int(d) for c in counts.values() for d in c})
    width = max(len(k) for k in counts)
    total_sites = {name: v["sites"].shape[0] for name, v in npz.items()}
    site_sets = {name: {tuple(s) for s in v["sites"]} for name, v in npz.items()}
    beanstalk = load_beanstalk()

    print("Runs per delay:")
    header = f"{'benchmark':<{width}}  " + "  ".join(f"{d:>6}" for d in delays) + f"  {'total':>7}"
    print(header)
    print("-" * len(header))
    for name in sorted(counts):
        c = counts[name]
        row = f"{name:<{width}}  " + "  ".join(f"{int(c.get(d, 0)):>6}" for d in delays)
        print(row + f"  {sum(int(v) for v in c.values()):>7}")

    print()
    print("Distinct bugs caught per delay (out of total sites):")
    header = (f"{'benchmark':<{width}}  " + "  ".join(f"{d:>6}" for d in delays)
              + f"  {'sites':>7}  {'beanstalk':>9}")
    print(header)
    print("-" * len(header))
    for name in sorted(caught):
        c = caught[name]
        row = f"{name:<{width}}  " + "  ".join(f"{int(c.get(d, 0)):>6}" for d in delays)
        print(row + f"  {total_sites[name]:>7}  {len(beanstalk.get(name, ())):>9}")
    print("-" * len(header))
    totals = {d: sum(int(caught[name].get(d, 0)) for name in caught) for d in delays}
    total_row = f"{'total':<{width}}  " + "  ".join(f"{totals[d]:>6}" for d in delays)
    total_row += f"  {sum(total_sites.values()):>7}"
    total_row += f"  {sum(len(v) for v in beanstalk.values()):>9}"
    print(total_row)

    print()
    poster = poster_order()
    print("Beanstalk pairs not in the sweep sites (with poster.py #number):")
    for name in sorted(beanstalk):
        extra = sorted(beanstalk[name] - site_sets.get(name, set()))
        cells = " ".join(
            f"[{a}, {b}]=#{poster.get((name, (a, b)), 'NA')}" for a, b in extra)
        print(f"{name:<{width}}  ({len(extra)})  " + cells)

    print()
    print("Sweep sites not found by beanstalk (with poster.py #number):")
    for name in sorted(site_sets):
        missed = sorted(site_sets[name] - beanstalk.get(name, set()))
        cells = " ".join(
            f"[{a}, {b}]=#{poster.get((name, (a, b)), 'NA')}" for a, b in missed)
        print(f"{name:<{width}}  ({len(missed)})  " + cells)
