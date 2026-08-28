#!/usr/bin/env python3
# uv run --with seaborn --with scipy --with matplotlib python3 analyze_passk.py
"""
Pass@k analysis and visualization for Harbor agent trials.
Outputs static matplotlib figures to results/plots/.
"""

import json
import math
import re
import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Mean of empty slice"
)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PASSK_DIR = Path("results/passk")
TRIALS_DIR = Path("results/trials")
OUT_DIR = Path("results/plots")

TASKS = ["h11", "pvlib", "networkx", "joblib"]
ORACLE_H = {"h11": 1.203, "pvlib": 22.563, "networkx": 1.3229, "joblib": 1.9524}

TEST_PAT = re.compile(r"/(test_|tests/|_test\.py)", re.IGNORECASE)
ASV_PAT = re.compile(r"asv\.conf\.json")
WRITE_PAT = re.compile(
    r"(?:cat\s*>{1,2}|tee\s+(?:-a\s+)?|sed\s+-i\S*\s[^/]*)"
    r"(/workspace/repo/[^\s'\"<>|;]+)"
    r"|"
    r"(?:^|\n)\s*>\s*(/workspace/repo/[^\s'\"<>|;]+)",
    re.MULTILINE,
)
# Also catch: echo ... > /workspace/repo/... and python ... > /workspace/repo/...
WRITE_PAT2 = re.compile(r"[>\|]\s*(/workspace/repo/[^\s'\"<>|;]+)")
# Agents typically cd to /workspace/repo and use relative paths — catch those too.
# open('pvlib/clearsky.py', 'w') / open('joblib/parallel.py', 'r+')
PY_WRITE_PAT = re.compile(
    r"open\s*\(\s*['\"]"
    r"((?:joblib|pvlib|networkx|h11)/[^'\"]+)"
    r"['\"][^)]*['\"][wa]",
)
# sed -i '...' pvlib/clearsky.py  (relative path, no /workspace/repo prefix)
SED_WRITE_PAT = re.compile(
    r"sed\s+-i[^\n]*\s((?:joblib|pvlib|networkx|h11)/[^\s'\"<>|;\n]+)",
)

READ_RE = re.compile(
    r"^\s*(cat(?!\s*>)|head|tail|grep|ls|find|wc|diff|less|more|stat|awk|sort|uniq|cut|sed(?!\s+-i))\b"
)
RUN_RE = re.compile(r"^\s*(python[23]?|pytest|uv|asv|pip|python\s+-m)\b")
WRITE_RE = re.compile(r"cat\s*>|tee\s|sed\s+-i|\bpatch\b.*-p[01]|\bwrite\b.*repo")
NAV_RE = re.compile(r"^\s*(cd|pwd)\b")
SETUP_RE = re.compile(r"^\s*(source|\.|export\s+\w|[A-Z_]+=)")

# ─── STYLE ─────────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)
PALETTE = sns.color_palette("colorblind")
TASK_COLOR = {task: PALETTE[i] for i, task in enumerate(TASKS + ["joblib"])}

TAXONOMY_ORDER = [
    "no_source_edit",
    "broken_tests",
    "no_speedup",
    "partial",
    "oracle_level",
]
TAXONOMY_LABELS = {
    "no_source_edit": "No source edit",
    "broken_tests": "Broken tests",
    "no_speedup": "No speedup",
    "partial": "Partial",
    "oracle_level": "Oracle level",
}
TAXONOMY_COLORS = {k: PALETTE[i] for i, k in enumerate(TAXONOMY_ORDER)}
TERM_COLORS = {
    "task_complete": PALETTE[2],
    "max_turns": PALETTE[3],
    "context_limit": PALETTE[4],
}
TERM_MARKERS = {"task_complete": "D", "max_turns": "x", "context_limit": "o"}
TERM_LABELS = {
    "task_complete": "Task complete",
    "max_turns": "Max turns",
    "context_limit": "Context limit",
}
CAT_LABELS = {
    "read": "Read file",
    "write": "Edit source",
    "run": "Execute",
    "nav": "Navigate",
    "setup": "Configure",
    "other": "Other",
}


def task_label(task: str) -> str:
    return task


def mono_ticks(ax, labels, axis="x", **kw):
    """Set tick labels and apply monospace font to each."""
    if axis == "x":
        ax.set_xticklabels(labels, **kw)
        for lbl in ax.get_xticklabels():
            lbl.set_fontfamily("monospace")
    else:
        ax.set_yticklabels(labels, **kw)
        for lbl in ax.get_yticklabels():
            lbl.set_fontfamily("monospace")


# ─── DATA MODEL ────────────────────────────────────────────────────────────────


class Trial:
    def __init__(self, task: str, entry: dict, tdir: Path):
        self.task = task
        self.trial_id = entry["trial_id"]
        self.idx = entry.get("idx", 0)
        self.reward = entry.get("reward", 0.0)
        self.tests_passed = entry.get("tests_passed", False)
        self.speedup = entry.get("lsv_mean_speedup", 0.0)
        self.n_benchmarks = entry.get("num_valid_benchmarks", 0)
        self.bench_speeds = entry.get("per_benchmark_speedups", {})
        self.patch_added = entry.get("patch_added", 0)
        self.patch_removed = entry.get("patch_removed", 0)

        traj_path = tdir / "agent" / "trajectory.json"
        with open(traj_path) as f:
            traj = json.load(f)
        self.steps = [s for s in traj["steps"] if s["source"] == "agent"]

        cfg_path = tdir / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        akw = cfg.get("agent", {}).get("kwargs", {})
        self.max_turns = akw.get("max_turns", 64)
        self.max_input_tokens = akw.get("model_info", {}).get("max_input_tokens", 55000)

    # ── derived properties ─────────────────────────────────────────────────────

    def n_turns(self) -> int:
        return len(self.steps)

    def source_edits(self) -> list:
        """List of (turn_idx, filepath) for writes to source files."""
        edits = []
        for i, step in enumerate(self.steps):
            for tc in step.get("tool_calls", []):
                ks = tc.get("arguments", {}).get("keystrokes", "")
                paths = set()
                for m in WRITE_PAT.finditer(ks):
                    p = (m.group(1) or m.group(2) or "").strip().rstrip("'\"")
                    if p:
                        paths.add(p)
                for m in WRITE_PAT2.finditer(ks):
                    p = m.group(1).strip().rstrip("'\"")
                    if p:
                        paths.add(p)
                # Relative paths: open('pvlib/clearsky.py', 'w') and sed -i ... file.py
                for m in PY_WRITE_PAT.finditer(ks):
                    paths.add(m.group(1))
                for m in SED_WRITE_PAT.finditer(ks):
                    paths.add(m.group(1))
                for p in paths:
                    if not ASV_PAT.search(p) and not TEST_PAT.search(p):
                        edits.append((i, p))
        return edits

    def termination(self) -> str:
        for step in self.steps:
            for tc in step.get("tool_calls", []):
                if tc.get("function_name") == "mark_task_complete":
                    return "task_complete"
        last = self.steps[-1].get("metrics", {}) if self.steps else {}
        if last.get("prompt_tokens", 0) >= 0.92 * self.max_input_tokens:
            return "context_limit"
        return "max_turns"

    def taxonomy(self) -> str:
        # patch_added > 0 is ground truth from the verifier diff; source_edits()
        # provides turn-level timing but can miss some write patterns.
        has_edit = bool(self.source_edits()) or self.patch_added > 0
        if not has_edit:
            return "no_source_edit"
        if not self.tests_passed:
            return "broken_tests"
        if self.speedup <= 1.01:
            return "no_speedup"
        if self.reward >= 2.5:
            return "oracle_level"
        return "partial"

    def token_series(self):
        prompt = [s.get("metrics", {}).get("prompt_tokens", 0) for s in self.steps]
        compl = [s.get("metrics", {}).get("completion_tokens", 0) for s in self.steps]
        return prompt, compl

    def tool_call_cats(self) -> list:
        """List of (turn_fraction 0-1, category) for each tool call."""
        n = max(self.n_turns(), 1)
        result = []
        for i, step in enumerate(self.steps):
            frac = i / n
            for tc in step.get("tool_calls", []):
                fn = tc.get("function_name", "")
                if fn == "mark_task_complete":
                    result.append((frac, "other"))
                    continue
                ks = tc.get("arguments", {}).get("keystrokes", "")
                if WRITE_RE.search(ks) or re.search(r"cat\s*>{1,2}|tee\s", ks):
                    result.append((frac, "write"))
                elif RUN_RE.match(ks):
                    result.append((frac, "run"))
                elif READ_RE.match(ks):
                    result.append((frac, "read"))
                elif NAV_RE.match(ks):
                    result.append((frac, "nav"))
                elif SETUP_RE.match(ks):
                    result.append((frac, "setup"))
                else:
                    result.append((frac, "other"))
        return result


def load_trials() -> list:
    """Load trials directly from results/trials/ — avoids JSONL-per-batch staleness."""
    trials = []
    for task in TASKS:
        for tdir in sorted(TRIALS_DIR.glob(f"{task}_agent_*")):
            reward_txt = tdir / "verifier" / "reward.txt"
            reward_json = tdir / "verifier" / "reward.json"
            if not reward_txt.exists() or not reward_json.exists():
                continue
            tid = tdir.name
            try:
                reward = float(reward_txt.read_text().strip())
                rdata = json.loads(reward_json.read_text())
            except Exception as e:
                print(f"  WARN {tid}: can't read reward — {e}")
                continue

            patch_added = patch_removed = 0
            patch_file = tdir / "artifacts" / "patch.diff"
            if patch_file.exists():
                for line in patch_file.read_text().splitlines():
                    if line.startswith("+") and not line.startswith("+++"):
                        patch_added += 1
                    elif line.startswith("-") and not line.startswith("---"):
                        patch_removed += 1

            entry = {
                "trial_id": tid,
                "idx": 0,
                "reward": reward,
                "tests_passed": rdata.get("tests_passed", False),
                "lsv_mean_speedup": rdata.get("lsv_mean_speedup", 0.0),
                "num_valid_benchmarks": rdata.get("num_valid_benchmarks", 0),
                "per_benchmark_speedups": rdata.get("per_benchmark_speedups") or {},
                "patch_added": patch_added,
                "patch_removed": patch_removed,
            }
            try:
                t = Trial(task, entry, tdir)
                trials.append(t)
            except Exception as e:
                print(f"  ERROR {tid}: {e}")

    counts = {task: sum(1 for t in trials if t.task == task) for task in TASKS}
    print(f"Loaded {len(trials)} trials: {counts}")
    return trials


# ─── HELPERS ───────────────────────────────────────────────────────────────────


def mean_ci(vals, z=1.96):
    """Return (mean, lo, hi) with normal-approx CI. vals may contain NaN."""
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    m = np.mean(arr)
    se = stats.sem(arr) if len(arr) > 1 else 0.0
    return m, m - z * se, m + z * se


def passk_estimate(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def passk_bootstrap_ci(outcomes: list, k: int, n_boot=2000, ci=0.95) -> tuple:
    """Bootstrap CI for pass@k. outcomes is list of 0/1."""
    rng = np.random.default_rng(42)
    n = len(outcomes)
    arr = np.array(outcomes)
    boot_vals = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        c_b = int(sample.sum())
        boot_vals.append(passk_estimate(n, c_b, k))
    lo = np.percentile(boot_vals, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_vals, 100 * (1 + ci) / 2)
    return lo, hi


def shorten_bench(name: str) -> str:
    """Shorten benchmark name for axis labels."""
    last = name.split(".")[-1]
    last = re.sub(r"^time_", "", last)
    last = last.replace("_turbidity", "_turb").replace("linke_turb", "linke")
    last = last.replace("_large", "_lg").replace("_small", "_sm")
    last = last.replace("_no_interp", "_raw")
    last = last.replace("realistic_headers", "headers")
    return last


def debug_file_paths(trials: list):
    """Print all unique filepaths written across all trials."""
    paths = set()
    for t in trials:
        for _, p in t.source_edits():
            paths.add(p)
    print("\n── Unique source paths written ──")
    for p in sorted(paths):
        print(" ", p)
    print()


# ─── PLOT A: SPEEDUP + REWARD DISTRIBUTIONS ────────────────────────────────────


def plot_speedup_distributions(trials: list):
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    fig, axes = plt.subplots(
        1, len(TASKS), figsize=(3.5 * len(TASKS), 4.5), sharey=False
    )

    for ax, task in zip(axes, TASKS):
        ts = by_task[task]
        vals = [t.speedup for t in ts]
        H = ORACLE_H[task]
        color = TASK_COLOR[task]

        # Clip outliers
        p95 = (
            np.percentile([v for v in vals if v > 0], 95)
            if any(v > 0 for v in vals)
            else 2.0
        )
        n_clipped = sum(1 for v in vals if v > p95 * 1.05)
        if n_clipped:
            print(f"  {task}: clipping {n_clipped} outlier(s) above {p95:.2f}x")

        # violin
        non_zero = [v for v in vals if v > 0]
        if len(non_zero) >= 3:
            parts = ax.violinplot(
                [non_zero],
                positions=[0],
                widths=0.6,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

        # jitter
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            jitter,
            vals,
            s=22,
            color=color,
            alpha=0.8,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )

        # reference lines
        ax.axhline(H, color="black", lw=1.2, ls="--", zorder=4, label=r"oracle $H$")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":", zorder=4, label=r"no-op")

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(bottom=0)
        if n_clipped:
            ax.set_ylim(top=p95 * 1.15)
        ax.set_xticks([])
        ax.set_ylabel(r"Mean speedup $g$" if task == TASKS[0] else "")
        ax.set_xlabel(task_label(task), fontfamily="monospace")
        if task == TASKS[0]:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / "A1_speedup_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


def plot_reward_distributions(trials: list):
    from matplotlib.transforms import blended_transform_factory

    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    fig, axes = plt.subplots(
        1, len(TASKS), figsize=(3.5 * len(TASKS), 5.0), sharey=True
    )

    REGION_BOUNDARY_LO = -0.5  # between broken and regression
    REGION_BOUNDARY_HI = 1.0  # between regression and speedup

    for ax, task in zip(axes, TASKS):
        ts = by_task[task]
        vals = [t.reward for t in ts]
        color = TASK_COLOR[task]

        # region shading (behind everything)
        ax.axhspan(-2.0, REGION_BOUNDARY_LO, color="#d62728", alpha=0.06, zorder=0)
        ax.axhspan(
            REGION_BOUNDARY_LO,
            REGION_BOUNDARY_HI,
            color="#ff7f0e",
            alpha=0.06,
            zorder=0,
        )
        ax.axhspan(REGION_BOUNDARY_HI, 5.0, color="#2ca02c", alpha=0.06, zorder=0)

        if len(vals) >= 3:
            parts = ax.violinplot(
                vals, positions=[0], widths=0.6, showmedians=True, showextrema=False
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            jitter,
            vals,
            s=22,
            color=color,
            alpha=0.8,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )

        ax.axhline(3.5, color="black", lw=1.2, ls="--", label=r"Oracle $r$")
        ax.axhline(1.0, color="gray", lw=0.8, ls=":", label="No-op")
        ax.axhline(-1.0, color="red", lw=0.8, ls="-.", alpha=0.5, label="Broken tests")

        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([])
        ax.set_ylabel(r"Reward $r$" if task == TASKS[0] else "")
        ax.set_xlabel(task_label(task), fontfamily="monospace")
        if task == TASKS[0]:
            ax.legend(fontsize=8, loc="upper right")

    # region labels on left of leftmost panel using blended transform
    ax0 = axes[0]
    bt = blended_transform_factory(ax0.transAxes, ax0.transData)
    kw = dict(transform=bt, ha="right", va="center", fontsize=8, clip_on=False)
    ax0.text(-0.04, -1.0, "Broken\nfunctionality", **kw)
    ax0.text(-0.04, 0.25, "Performance\nregression", **kw)
    ax0.text(-0.04, 2.5, "Speedup", **kw)

    plt.tight_layout()
    fig.subplots_adjust(left=0.18)
    out = OUT_DIR / "A2_reward_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT B: AGENT TAXONOMY ─────────────────────────────────────────────────────


def plot_taxonomy(trials: list):
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    fig, (ax_tax, ax_term) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── taxonomy grouped bar ──
    x = np.arange(len(TASKS))
    width = 0.15
    for i, cls in enumerate(TAXONOMY_ORDER):
        counts = [
            sum(1 for t in by_task[task] if t.taxonomy() == cls) for task in TASKS
        ]
        offset = (i - len(TAXONOMY_ORDER) / 2 + 0.5) * width
        bars = ax_tax.bar(
            x + offset,
            counts,
            width,
            label=TAXONOMY_LABELS[cls],
            color=TAXONOMY_COLORS[cls],
            alpha=0.85,
        )

    ax_tax.set_xticks(x)
    mono_ticks(ax_tax, [task_label(t) for t in TASKS])
    ax_tax.set_ylabel("Trial count")
    ax_tax.legend(fontsize=8, ncol=2)

    # ── termination grouped bar ──
    term_order = list(TERM_COLORS.keys())
    for i, cls in enumerate(term_order):
        counts = [
            sum(1 for t in by_task[task] if t.termination() == cls) for task in TASKS
        ]
        offset = (i - len(term_order) / 2 + 0.5) * width * (5 / 3)
        ax_term.bar(
            x + offset,
            counts,
            width * (5 / 3),
            label=TERM_LABELS[cls],
            color=TERM_COLORS[cls],
            alpha=0.85,
        )

    ax_term.set_xticks(x)
    mono_ticks(ax_term, [task_label(t) for t in TASKS])
    ax_term.set_ylabel("Trial count")
    ax_term.legend(fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "B_agent_taxonomy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT C: EDIT RASTER ────────────────────────────────────────────────────────


def plot_edit_raster(trials: list):
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    rng = np.random.default_rng(7)
    Y_LO, Y_HI = -1.35, 4.0
    JITTER = 0.06  # spread same-reward trials so ticks don't overlap

    fig, axes = plt.subplots(
        1, len(TASKS), figsize=(4.5 * len(TASKS), 4.5), sharey=True
    )

    for ax, task in zip(axes, TASKS):
        ts = by_task[task]
        if not ts:
            ax.set_visible(False)
            continue

        color = TASK_COLOR[task]

        # assign small jitter to differentiate same-reward trials
        rewards = [t.reward for t in ts]
        jitter = rng.uniform(-JITTER, JITTER, len(ts))

        for i, trial in enumerate(ts):
            y = trial.reward + jitter[i]
            edits = trial.source_edits()
            if edits:
                turn_idxs = [e[0] for e in edits]
                ax.eventplot(
                    turn_idxs,
                    lineoffsets=y,
                    linelengths=0.18,
                    linewidths=1.2,
                    colors=color,
                )

            term = trial.termination()
            tmark = TERM_MARKERS.get(term, "x")
            tcol = TERM_COLORS.get(term, "gray")
            ax.scatter(
                [trial.n_turns()],
                [y],
                marker=tmark,
                color=tcol,
                s=45,
                zorder=5,
                linewidths=1.2,
            )

        ax.set_xlim(0, max(t.n_turns() for t in ts) + 2)
        ax.set_ylim(Y_LO, Y_HI)
        ax.axhline(1.0, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax.axhline(-0.5, color="gray", lw=0.7, ls="--", alpha=0.4)
        ax.axvline(ts[0].max_turns, color="gray", lw=0.7, ls="--", alpha=0.4)
        if task == TASKS[0]:
            ax.set_ylabel(r"Reward $r$")
        ax.set_xlabel("Turn")
        ax.set_title(task_label(task), pad=4, fontfamily="monospace")

    # legend to the left of the leftmost panel
    legend_elems = [
        Line2D([0], [0], color=PALETTE[0], lw=1.5, label="Source edit"),
        *[
            Line2D(
                [0],
                [0],
                marker=TERM_MARKERS[k],
                color=TERM_COLORS[k],
                lw=0,
                markersize=6,
                label=TERM_LABELS[k],
            )
            for k in TERM_COLORS
        ],
    ]
    axes[0].legend(
        handles=legend_elems,
        fontsize=7,
        loc="center right",
        bbox_to_anchor=(0, 0.5),
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    fig.subplots_adjust(left=0.14)
    out = OUT_DIR / "C_edit_raster.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT D: CONTEXT LENGTH ─────────────────────────────────────────────────────


def _token_curves(trials, token_idx: int):
    """Build ragged token-per-turn matrix across trials → (mean, lo, hi) per turn."""
    max_t = max(t.n_turns() for t in trials)
    mat = np.full((len(trials), max_t), np.nan)
    for r, t in enumerate(trials):
        series = t.token_series()[token_idx]
        mat[r, : len(series)] = series
    means, los, his = [], [], []
    for col in range(max_t):
        col_vals = mat[:, col][~np.isnan(mat[:, col])]
        if len(col_vals) >= 2:
            m, lo, hi = mean_ci(col_vals)
        elif len(col_vals) == 1:
            m, lo, hi = col_vals[0], col_vals[0], col_vals[0]
        else:
            m, lo, hi = np.nan, np.nan, np.nan
        means.append(m)
        los.append(lo)
        his.append(hi)
    turns = np.arange(max_t)
    valid = ~np.isnan(means)
    return (
        turns[valid],
        np.array(means)[valid],
        np.array(los)[valid],
        np.array(his)[valid],
    )


def plot_context_length(trials: list):
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    for token_idx, ylabel, fname in [
        (0, r"Input tokens (prompt)", "D1_input_tokens.png"),
        (1, r"Output tokens (completion)", "D2_output_tokens.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for task in TASKS:
            ts = by_task[task]
            if not ts:
                continue
            turns, means, los, his = _token_curves(ts, token_idx)
            color = TASK_COLOR[task]
            ax.plot(turns, means, color=color, lw=1.8, label=task_label(task))
            ax.fill_between(turns, los, his, color=color, alpha=0.2)

        ax.set_xlabel("Turn")
        ax.set_ylabel(ylabel)
        ax.legend()
        plt.tight_layout()
        out = OUT_DIR / fname
        fig.savefig(out)
        plt.close(fig)
        print(f"  saved {out}")


# ─── PLOT E: TOOL CALL TYPE OVER QUINTILE ──────────────────────────────────────


def plot_tool_calls(trials: list):
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}
    cats = ["read", "write", "run", "nav", "setup", "other"]
    cat_colors = {c: PALETTE[i] for i, c in enumerate(cats)}

    fig, axes = plt.subplots(1, len(TASKS), figsize=(3.5 * len(TASKS), 4), sharey=True)

    for ax, task in zip(axes, TASKS):
        ts = by_task[task]
        if not ts:
            ax.set_visible(False)
            continue

        bins = np.linspace(0, 1, 6)  # 5 quintiles
        counts = np.zeros((len(cats), 5))

        for t in ts:
            for frac, cat in t.tool_call_cats():
                bin_idx = min(int(frac * 5), 4)
                counts[cats.index(cat), bin_idx] += 1

        totals = counts.sum(axis=0)
        fracs = np.divide(counts, totals, where=totals > 0, out=np.zeros_like(counts))

        x = np.arange(1, 6)
        bottom = np.zeros(5)
        for i, cat in enumerate(cats):
            ax.bar(
                x,
                fracs[i],
                bottom=bottom,
                label=CAT_LABELS[cat],
                color=cat_colors[cat],
                alpha=0.9,
                width=0.8,
            )
            bottom += fracs[i]

        ax.set_xticks(x)
        ax.set_xticklabels([r"$Q_1$", r"$Q_2$", r"$Q_3$", r"$Q_4$", r"$Q_5$"])
        ax.set_xlabel("Turn quintile")
        ax.set_ylabel("Fraction of tool calls" if task == TASKS[0] else "")
        ax.set_ylim(0, 1)
        ax.set_title(task_label(task), pad=4, fontfamily="monospace")
        if task == TASKS[-1]:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / "E_tool_calls_quintile.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT F: PATCH SIZE VS REWARD ──────────────────────────────────────────────


def plot_patch_vs_reward(trials: list):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    all_sizes = []
    for t in trials:
        all_sizes.append(t.patch_added + t.patch_removed)
    p95 = np.percentile(all_sizes, 95)
    n_clipped = sum(1 for s in all_sizes if s > p95 * 1.05)
    if n_clipped:
        print(f"  patch scatter: clipping {n_clipped} outlier(s) (>{p95:.0f} lines)")

    for task in TASKS:
        ts = [t for t in trials if t.task == task]
        xs = [min(t.patch_added + t.patch_removed, p95 * 1.05) for t in ts]
        ys = [t.reward for t in ts]
        ax.scatter(
            xs,
            ys,
            color=TASK_COLOR[task],
            s=35,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            label=task_label(task),
            zorder=3,
        )

    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel(r"Patch size (added $+$ removed lines)")
    ax.set_ylabel(r"Reward $r$")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "F_patch_vs_reward.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT G: REWARD VS TURN COUNT ──────────────────────────────────────────────


def plot_reward_vs_turns(trials: list):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    max_turns_64 = 64
    max_turns_96 = 96

    for task in TASKS:
        ts = [t for t in trials if t.task == task]
        for t in ts:
            term = t.termination()
            color = TASK_COLOR[task]
            marker = (
                "x"
                if term == "max_turns"
                else ("o" if term == "task_complete" else "s")
            )
            markeredge = "black" if term == "context_limit" else "white"
            ax.scatter(
                [t.n_turns()],
                [t.reward],
                color=color,
                marker=marker,
                s=40,
                alpha=0.8,
                edgecolors=markeredge,
                linewidths=0.7,
                zorder=3,
            )

    ax.axhline(1.0, color="gray", lw=0.8, ls=":", alpha=0.5)

    # turn limit lines
    ax.axvline(max_turns_64, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.axvline(max_turns_96, color="dimgray", lw=1.0, ls="-.", alpha=0.7)

    task_patches = [
        mpatches.Patch(color=TASK_COLOR[t], label=task_label(t)) for t in TASKS
    ]
    term_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="gray",
            lw=0,
            markersize=6,
            label=TERM_LABELS["task_complete"],
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="gray",
            lw=0,
            markersize=6,
            label=TERM_LABELS["max_turns"],
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="gray",
            lw=0,
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=TERM_LABELS["context_limit"],
        ),
    ]
    ax.legend(handles=task_patches + term_elems, fontsize=8, ncol=2)

    ax.set_xlabel("Total agent turns")
    ax.set_ylabel(r"Reward $r$")
    fig.canvas.draw()  # finalise limits before annotating
    ymin = ax.get_ylim()[0]
    ax.text(
        max_turns_64 + 0.5,
        ymin + 0.05,
        "max_turns=64",
        fontsize=7,
        color="gray",
        va="bottom",
        rotation=90,
        fontfamily="monospace",
    )
    ax.text(
        max_turns_96 + 0.5,
        ymin + 0.05,
        "max_turns=96",
        fontsize=7,
        color="dimgray",
        va="bottom",
        rotation=90,
        fontfamily="monospace",
    )
    plt.tight_layout()
    out = OUT_DIR / "G_reward_vs_turns.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT H: PASS@K CURVES ─────────────────────────────────────────────────────


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def plot_passk_curves(trials: list):
    """Per-trial success rate by task — shows GRPO training signal quality."""
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    thresholds = [
        (lambda t: t.reward > 1.05, r"$r > 1.05$  (real speedup)",  PALETTE[0]),
        (lambda t: t.reward > 1.0,  r"$r > 1.0$   (any improvement)", PALETTE[1]),
        (lambda t: t.tests_passed,  r"tests pass",                   PALETTE[2]),
    ]

    tasks_ok = [task for task in TASKS if by_task[task]]
    x = np.arange(len(tasks_ok))
    n_thresh = len(thresholds)
    total_w = 0.6
    bw = total_w / n_thresh

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(tasks_ok)), 4.5))

    # Good-GRPO zone shading
    ax.axhspan(0.2, 0.8, color="gold", alpha=0.12, label="good signal zone (20–80%)")
    ax.axhline(0.5, color="grey", lw=1.0, ls="--", alpha=0.6)

    for ti, (fn, label, color) in enumerate(thresholds):
        offset = (ti - (n_thresh - 1) / 2) * bw
        rates = []
        for task in tasks_ok:
            ts = by_task[task]
            n = len(ts)
            k = sum(1 for t in ts if fn(t))
            rates.append(k / n if n else 0.0)

        ax.bar(x + offset, rates, bw * 0.88, color=color, alpha=0.85, label=label)
        for xi, (task, rate) in enumerate(zip(tasks_ok, rates)):
            ts = by_task[task]
            n = len(ts)
            k = sum(1 for t in ts if fn(t))
            ax.text(
                xi + offset, rate + 0.03,
                f"{k}/{n}",
                ha="center", va="bottom", fontsize=7, color=color,
                fontweight="bold",
            )

    ax.set_xticks(x)
    mono_ticks(ax, tasks_ok, axis="x")
    ax.set_ylabel("Per-trial success rate")
    ax.set_ylim(0, 1.18)
    ax.set_title("Per-trial success rate by task (GRPO signal quality)", pad=6)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / "H_passk_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT I: BENCHMARK HEATMAP ─────────────────────────────────────────────────


def plot_benchmark_heatmap(trials: list):
    by_task = {task: [t for t in trials if t.task == task] for task in TASKS}

    tasks_with_benches = [
        task for task in TASKS if any(t.bench_speeds for t in by_task[task])
    ]
    if not tasks_with_benches:
        print("  WARN: no benchmark data found, skipping heatmap")
        return

    n_panels = len(tasks_with_benches)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(
            5 * n_panels,
            max(4, 0.4 * max(len(by_task[task]) for task in tasks_with_benches) + 1.5),
        ),
        squeeze=False,
    )
    axes = axes[0]

    # diverging colormap centered at 1.0
    cmap = matplotlib.colormaps["RdYlGn"]

    for ax, task in zip(axes, tasks_with_benches):
        ts = by_task[task]
        all_bench_names = sorted(set(k for t in ts for k in t.bench_speeds.keys()))
        if not all_bench_names:
            ax.set_visible(False)
            continue

        # matrix: rows = trials (sorted by mean speedup desc), cols = benchmarks
        mat = np.full((len(ts), len(all_bench_names)), np.nan)
        for r, t in enumerate(ts):
            for c, bn in enumerate(all_bench_names):
                if bn in t.bench_speeds:
                    mat[r, c] = t.bench_speeds[bn]

        row_means = np.nanmean(mat, axis=1)
        order = np.argsort(-row_means)
        mat = mat[order]
        ts_sorted = [ts[i] for i in order]

        # normalize to color range: 0 = 0.5x, 1 = 1.5x (cap)
        vmin, vmax = 0.5, 2.0
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

        im = ax.imshow(
            mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
        )

        col_labels = [shorten_bench(n) for n in all_bench_names]
        ax.set_xticks(range(len(all_bench_names)))
        mono_ticks(ax, col_labels, rotation=30, ha="right", fontsize=8)

        row_labels = [f"{t.reward:.2f}" for t in ts_sorted]
        ax.set_yticks(range(len(ts)))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_ylabel(r"Reward $r$" if task == tasks_with_benches[0] else "")
        ax.set_title(task_label(task), pad=4, fontfamily="monospace")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_label(r"speedup", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = OUT_DIR / "I_benchmark_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT J: FIRST-EDIT CDF ────────────────────────────────────────────────────


def plot_first_edit_cdf(trials: list):
    """Fraction of trials that have made their first source edit by turn t."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for task in TASKS:
        ts = [t for t in trials if t.task == task]
        n = len(ts)
        color = TASK_COLOR[task]

        first_edits = []
        for t in ts:
            edits = t.source_edits()
            if edits:
                first_edits.append(min(e[0] for e in edits))

        if not first_edits:
            continue

        # ECDF — normalised by total trials so ceiling = fraction that ever edited
        turns_sorted = np.sort(first_edits)
        y_vals = np.arange(1, len(turns_sorted) + 1) / n
        # step function: prepend (0, 0) and repeat last y for visual close
        turns_plot = np.concatenate([[0], turns_sorted])
        y_plot = np.concatenate([[0], y_vals])

        ax.step(
            turns_plot, y_plot, where="post", color=color, lw=2, label=task_label(task)
        )
        # ceiling: fraction that ever edited (dashed)
        ax.axhline(len(first_edits) / n, color=color, lw=0.9, ls="--", alpha=0.55)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Fraction of trials with first source edit")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "J_first_edit_cdf.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT K: BEST-OF-k EXPECTED REWARD ─────────────────────────────────────────


def _best_k_stats(rewards: list, k: int, n_boot: int = 2000):
    rng = np.random.default_rng(42)
    samples = rng.choice(rewards, size=(n_boot, k), replace=True)
    maxes = samples.max(axis=1)
    lo, hi = np.percentile(maxes, [5, 95])
    return float(maxes.mean()), float(lo), float(hi)


def plot_best_of_k(trials: list):
    """E[max reward over k trials] vs k — diminishing-returns curve per task."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for task in TASKS:
        ts = [t for t in trials if t.task == task]
        if not ts:
            continue
        rewards = np.array([t.reward for t in ts])
        n = len(rewards)
        ks = list(range(1, n + 1))
        means, los, his = zip(*[_best_k_stats(rewards, k) for k in ks])

        color = TASK_COLOR[task]
        ax.plot(ks, means, color=color, lw=2, label=task_label(task))
        ax.fill_between(ks, los, his, color=color, alpha=0.15)

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(r"$k$ (trials sampled)")
    ax.set_ylabel(r"$\mathrm{E}\!\left[\max_{1 \leq i \leq k} r_i\right]$")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / "K_best_of_k.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT L: METRIC CORRELATION HEATMAP ────────────────────────────────────────


def plot_metric_correlations(trials: list):
    """Pearson correlation matrix of trial-level metrics."""
    rows = []
    for t in trials:
        edits = t.source_edits()
        inp, out = t.token_series()
        fe = min(e[0] for e in edits) if edits else t.n_turns()
        rows.append(
            {
                "reward": t.reward,
                "first edit": fe,
                "source edits": len(edits),
                "output tokens": sum(out),
                "input tokens": sum(inp),
                "turns": t.n_turns(),
                "patch size": t.patch_added + t.patch_removed,
            }
        )

    keys = list(rows[0].keys())
    mat = np.array([[r[k] for k in keys] for r in rows]).T
    C = np.corrcoef(mat)

    # mask upper triangle
    mask = np.triu(np.ones_like(C, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = matplotlib.colormaps["RdBu_r"]
    im = ax.imshow(
        np.where(mask, np.nan, C),
        vmin=-1,
        vmax=1,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )

    n = len(keys)
    for i in range(n):
        for j in range(n):
            if i >= j:
                ax.text(
                    j,
                    i,
                    f"{C[i, j]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if abs(C[i, j]) < 0.6 else "white",
                )

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(keys, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Pearson $r$")
    plt.tight_layout()
    out = OUT_DIR / "L_metric_correlations.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT M: PER-BENCHMARK SPEEDUP VIOLIN ──────────────────────────────────────


def plot_per_benchmark_violin(trials: list):
    """Speedup distribution per benchmark within each task."""
    tasks_ok = [
        task for task in TASKS if any(t.bench_speeds for t in trials if t.task == task)
    ]
    if not tasks_ok:
        print("  WARN: no benchmark data, skipping violin")
        return

    fig, axes = plt.subplots(
        1, len(tasks_ok), figsize=(4.5 * len(tasks_ok), 4.5), squeeze=False
    )
    axes = axes[0]

    for ax, task in zip(axes, tasks_ok):
        ts = [t for t in trials if t.task == task]
        all_benches = sorted(set(k for t in ts for k in t.bench_speeds))
        if not all_benches:
            continue

        color = TASK_COLOR[task]
        positions = range(len(all_benches))
        data_per_bench = [
            [t.bench_speeds[b] for t in ts if b in t.bench_speeds] for b in all_benches
        ]

        for pos, (bench, vals) in enumerate(zip(all_benches, data_per_bench)):
            if len(vals) >= 3:
                parts = ax.violinplot(
                    [vals],
                    positions=[pos],
                    widths=0.7,
                    showmedians=True,
                    showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.45)
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.5)
            jitter = np.random.default_rng(pos).uniform(-0.12, 0.12, len(vals))
            ax.scatter(
                jitter + pos,
                vals,
                s=22,
                color=color,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.3,
                zorder=3,
            )

        ax.axhline(1.0, color="gray", lw=0.8, ls=":", zorder=0)
        ax.set_xticks(list(positions))
        mono_ticks(
            ax,
            [shorten_bench(b) for b in all_benches],
            rotation=25,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("Speedup" if task == tasks_ok[0] else "")
        ax.set_title(task_label(task), pad=4, fontfamily="monospace")

    plt.tight_layout()
    out = OUT_DIR / "M_per_benchmark_violin.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT N: TERMINATION × TAXONOMY HEATMAP ────────────────────────────────────


def plot_termination_taxonomy(trials: list):
    """Cross-tabulation of taxonomy class vs termination reason."""
    counts = np.zeros((len(TAXONOMY_ORDER), len(TERM_COLORS)), dtype=int)
    term_order = list(TERM_COLORS.keys())

    for t in trials:
        tax = t.taxonomy()
        term = t.termination()
        r = TAXONOMY_ORDER.index(tax)
        c = term_order.index(term)
        counts[r, c] += 1

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(counts, cmap="crest", aspect="auto", interpolation="nearest")

    for i in range(len(TAXONOMY_ORDER)):
        for j in range(len(term_order)):
            v = counts[i, j]
            tmax = counts.max()
            fc = "white" if v > 0.55 * tmax else "black"
            ax.text(
                j,
                i,
                str(v),
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=fc,
            )

    ax.set_xticks(range(len(term_order)))
    ax.set_yticks(range(len(TAXONOMY_ORDER)))
    ax.set_xticklabels([TERM_LABELS[k] for k in term_order], fontsize=9)
    ax.set_yticklabels([TAXONOMY_LABELS[k] for k in TAXONOMY_ORDER], fontsize=9)
    ax.set_xlabel("Termination reason")
    ax.set_ylabel("Trial outcome")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Count")
    plt.tight_layout()
    out = OUT_DIR / "N_termination_taxonomy.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT O: EXPLORATION OVERHEAD BY OUTCOME ───────────────────────────────────


def plot_exploration_overhead(trials: list):
    """Fraction of turns spent before first source edit, grouped by taxonomy class."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # exploration fraction: first_edit_turn / n_turns  (1.0 = never edited)
    by_class = {cls: [] for cls in TAXONOMY_ORDER}
    for t in trials:
        cls = t.taxonomy()
        edits = t.source_edits()
        fe = min(e[0] for e in edits) if edits else t.n_turns()
        frac = fe / t.n_turns() if t.n_turns() > 0 else 1.0
        by_class[cls].append(frac)

    x_positions = range(len(TAXONOMY_ORDER))
    for pos, cls in enumerate(TAXONOMY_ORDER):
        vals = by_class[cls]
        color = TAXONOMY_COLORS[cls]
        if len(vals) >= 3:
            parts = ax.violinplot(
                [vals],
                positions=[pos],
                widths=0.65,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.45)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)
        jitter = np.random.default_rng(pos).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            jitter + pos,
            vals,
            s=30,
            color=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        # count label above
        ax.text(
            pos,
            1.04,
            f"n={len(vals)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(
        [TAXONOMY_LABELS[k] for k in TAXONOMY_ORDER],
        rotation=15,
        ha="right",
        fontsize=9,
    )
    ax.set_ylabel("Exploration fraction\n(turns before first edit / total turns)")
    ax.set_ylim(-0.05, 1.15)
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    plt.tight_layout()
    out = OUT_DIR / "O_exploration_overhead.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── LSV DATA LOADER ───────────────────────────────────────────────────────────


def load_lsv_stats() -> list:
    """Load per-trial LSV benchmark-selection and timing data from artifact JSONs."""
    stats = []
    for task in TASKS:
        for tdir in sorted(TRIALS_DIR.glob(f"{task}_agent_*")):
            init_f = tdir / "artifacts" / "lsv" / "lsv_init_results.json"
            meas_f = tdir / "artifacts" / "lsv" / "lsv_measure_results.json"
            rew_f = tdir / "verifier" / "reward.json"
            if not (init_f.exists() and meas_f.exists() and rew_f.exists()):
                continue
            try:
                init = json.loads(init_f.read_text())
                meas = json.loads(meas_f.read_text())
                rew = json.loads(rew_f.read_text())
                disc = init.get("benchmarks_discovered", [])
                imp = init.get("benchmarks_impactable", [])
                src = init.get("source_files_covered", [])
                t = rew.get("timings", {})
                stats.append(
                    {
                        "task": task,
                        "trial_id": tdir.name,
                        "discovered": len(disc) if isinstance(disc, list) else disc,
                        "impactable": len(imp) if isinstance(imp, list) else imp,
                        "src_files": len(src) if isinstance(src, list) else src,
                        "selected": meas.get("selected_count", 0),
                        "selected_expanded": len(meas.get("benchmarks", {})),
                        "total": meas.get("total_count", 0),
                        "skipped": meas.get("skipped_count", 0),
                        "init_overhead_s": init.get("timing", {}).get("total_s", 0.0),
                        "measure_s": t.get("lsv_measure_s", 0),
                        "setup_s": t.get("setup_total_s", 0),
                        "lsv_init_s": t.get("lsv_init_s", 0),
                        "pytest_s": t.get("pytest_s", 0),
                    }
                )
            except Exception as e:
                print(f"  WARN lsv_stats {tdir.name}: {e}")
    counts = {task: sum(1 for s in stats if s["task"] == task) for task in TASKS}
    print(f"Loaded {len(stats)} LSV stat records: {counts}")
    return stats


# ─── PLOT P: LSV PIPELINE TIMING BREAKDOWN ────────────────────────────────────


def plot_lsv_pipeline_timing(stats: list):
    """Grouped stacked bars: LSV (left) vs no-LSV counterfactual (right) per task.

    Counterfactual replaces selected-benchmark measure time with full-suite
    vanilla ASV time (disc / selected_expanded × measure_s) and drops lsv_init.
    Log y-axis handles the networkx 15× gap cleanly.
    """
    import statistics

    by_task = {task: [s for s in stats if s["task"] == task] for task in TASKS}

    task_med = {}
    for task in TASKS:
        rows = by_task[task]
        if not rows:
            continue
        med_setup = statistics.median(r["setup_s"] for r in rows)
        med_init = statistics.median(r["lsv_init_s"] for r in rows)
        med_meas = statistics.median(r["measure_s"] for r in rows)
        med_pytest = statistics.median(r["pytest_s"] for r in rows)
        other_setup = max(0, med_setup - med_init - med_meas - med_pytest)

        valid = [
            r
            for r in rows
            if r["selected_expanded"] > 0 and r["discovered"] > 0 and r["measure_s"] > 0
        ]
        vanilla_meas = (
            statistics.median(
                (r["discovered"] / r["selected_expanded"]) * r["measure_s"]
                for r in valid
            )
            if valid
            else med_meas
        )

        task_med[task] = {
            "setup": other_setup,
            "lsv_init": med_init,
            "measure": med_meas,
            "vanilla_meas": vanilla_meas,
            "pytest": med_pytest,
        }

    tasks_ok = [t for t in TASKS if t in task_med]
    n = len(tasks_ok)
    x = np.arange(n)
    bw = 0.3  # individual bar width

    # Segment definitions — same colors for shared phases
    C_SETUP = PALETTE[0]
    C_INIT = PALETTE[1]
    C_MEAS = PALETTE[2]
    C_PYTEST = PALETTE[3]

    fig, ax = plt.subplots(figsize=(9, 5))

    # ── LSV bars (left) ────────────────────────────────────────────────────────
    lsv_segs = [
        ("setup", C_SETUP, "Container / env setup", True),
        ("lsv_init", C_INIT, "lsv_init (dep-graph query)", True),
        ("measure", C_MEAS, "Benchmark measure (selected)", True),
        ("pytest", C_PYTEST, "pytest", True),
    ]
    bottoms = np.zeros(n)
    for key, color, label, do_label in lsv_segs:
        heights = np.array([task_med[t][key] for t in tasks_ok])
        ax.bar(
            x - bw / 2,
            heights,
            bw,
            bottom=bottoms,
            color=color,
            alpha=0.88,
            label=label,
            zorder=3,
        )
        bottoms += heights

    # ── No-LSV counterfactual bars (right, hatched) ───────────────────────────
    nolsv_segs = [
        ("setup", C_SETUP, None),  # same color, no extra label
        ("vanilla_meas", C_MEAS, "Benchmark measure (full suite, est.)"),
        ("pytest", C_PYTEST, None),
    ]
    bottoms = np.zeros(n)
    for key, color, label in nolsv_segs:
        heights = np.array([task_med[t][key] for t in tasks_ok])
        ax.bar(
            x + bw / 2,
            heights,
            bw,
            bottom=bottoms,
            color=color,
            alpha=0.40,
            hatch="///",
            edgecolor=color,
            label=label,
            zorder=3,
        )
        bottoms += heights

    # ── Speedup label above the taller bar for every task ─────────────────────
    for xi, task in enumerate(tasks_ok):
        total_lsv = sum(
            task_med[task][k] for k in ("setup", "lsv_init", "measure", "pytest")
        )
        total_vanilla = sum(
            task_med[task][k] for k in ("setup", "vanilla_meas", "pytest")
        )
        ratio = total_vanilla / total_lsv if total_lsv > 0 else 1.0
        top = max(total_lsv, total_vanilla)
        ax.text(
            xi + bw / 2,
            top * 1.3,
            f"{ratio:.2f}×",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    # ── X-axis group labels ────────────────────────────────────────────────────
    ax.set_xticks(x)
    mono_ticks(ax, [task_label(t) for t in tasks_ok])

    # Sub-labels "LSV" / "no LSV" under each pair
    for xi in range(n):
        ax.text(
            xi - bw / 2,
            -0.04,
            "LSV",
            ha="center",
            va="top",
            fontsize=7,
            color="dimgray",
            transform=ax.get_xaxis_transform(),
        )
        ax.text(
            xi + bw / 2,
            -0.04,
            "no LSV",
            ha="center",
            va="top",
            fontsize=7,
            color="dimgray",
            transform=ax.get_xaxis_transform(),
        )

    ax.set_yscale("log")
    ax.set_ylabel("Wall-clock time (seconds, median, log scale)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    plt.tight_layout()
    out = OUT_DIR / "P_lsv_pipeline_timing.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT Q: BENCHMARK SELECTION FUNNEL ───────────────────────────────────────


def plot_lsv_benchmark_funnel(stats: list):
    """Horizontal grouped bars: dep-graph entries → ASV suite → selected, per task.

    Three levels always form a valid decreasing funnel:
      discovered (static dep-graph entries)  ≥  total (ASV suite)  ≥  selected (this patch)
    """
    import statistics

    by_task = {task: [s for s in stats if s["task"] == task] for task in TASKS}

    rows_repr = {}
    for task in TASKS:
        rows = by_task[task]
        if not rows:
            continue
        tot = rows[0]["total"]
        sel = statistics.median(r["selected"] for r in rows if r["total"] > 0)
        rows_repr[task] = (tot, sel)

    fig, ax = plt.subplots(figsize=(8, 4))

    height = 0.18
    offsets = [-height / 2, height / 2]
    bar_specs = [
        (
            "Benchmark functions in ASV suite",
            [rows_repr[t][0] for t in TASKS if t in rows_repr],
            PALETTE[0],
        ),
        (
            "Functions selected for this patch",
            [rows_repr[t][1] for t in TASKS if t in rows_repr],
            PALETTE[2],
        ),
    ]

    tasks_with_data = [t for t in TASKS if t in rows_repr]
    y_pos = np.arange(len(tasks_with_data))

    for (label, vals, color), off in zip(bar_specs, offsets):
        bars = ax.barh(
            y_pos + off,
            vals,
            height * 0.9,
            color=color,
            alpha=0.85,
            label=label,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                v + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{int(v)}",
                va="center",
                ha="left",
                fontsize=8,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Benchmark function count (log scale)")
    ax.set_yticks(y_pos)
    mono_ticks(ax, [task_label(t) for t in tasks_with_data], axis="y")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(left=0.5)

    plt.tight_layout()
    out = OUT_DIR / "Q_lsv_benchmark_funnel.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT R: LSV vs VANILLA ASV TIME COMPARISON ───────────────────────────────


def plot_lsv_time_savings(stats: list):
    """Paired bars: LSV actual measure time vs estimated vanilla-ASV time."""
    import statistics

    by_task = {task: [s for s in stats if s["task"] == task] for task in TASKS}

    # Per trial: vanilla_time = (discovered_expanded / selected_expanded) * measure_s
    # Both counts are in expanded-ID units (parametrized variants), giving the true
    # time ratio rather than the over-estimated function-count ratio.
    task_lsv, task_vanilla, task_overhead = {}, {}, {}
    for task in TASKS:
        rows = [
            s
            for s in by_task[task]
            if s["selected_expanded"] > 0 and s["discovered"] > 0 and s["measure_s"] > 0
        ]
        if not rows:
            continue
        lsv_times = [r["measure_s"] for r in rows]
        vanilla_times = [
            (r["discovered"] / r["selected_expanded"]) * r["measure_s"] for r in rows
        ]
        overheads = [r["init_overhead_s"] * 1000 for r in rows]  # ms
        task_lsv[task] = statistics.median(lsv_times)
        task_vanilla[task] = statistics.median(vanilla_times)
        task_overhead[task] = statistics.median(overheads)

    tasks_ok = [t for t in TASKS if t in task_lsv]
    x = np.arange(len(tasks_ok))
    width = 0.32

    fig, (ax_main, ax_inset) = plt.subplots(
        1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 1]}
    )

    # Main panel: LSV vs Vanilla
    b1 = ax_main.bar(
        x - width / 2,
        [task_lsv[t] for t in tasks_ok],
        width,
        label="LSV (selected benchmarks)",
        color=PALETTE[2],
        alpha=0.85,
        zorder=3,
    )
    b2 = ax_main.bar(
        x + width / 2,
        [task_vanilla[t] for t in tasks_ok],
        width,
        label="Vanilla ASV (full suite estimate)",
        color=PALETTE[3],
        alpha=0.6,
        zorder=3,
    )

    # Annotate speedup ratios where > 1
    for xi, task in enumerate(tasks_ok):
        ratio = task_vanilla[task] / task_lsv[task] if task_lsv[task] > 0 else 1.0
        if ratio > 1.05:
            ymax = task_vanilla[task]
            ax_main.text(
                xi + width / 2,
                ymax + ymax * 0.03,
                f"{ratio:.0f}$\\times$ faster",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                fontweight="bold",
            )

    ax_main.set_xticks(x)
    mono_ticks(ax_main, [task_label(t) for t in tasks_ok])
    ax_main.set_ylabel("Benchmark measure time (seconds, median)")
    ax_main.legend(fontsize=8)
    ax_main.set_ylim(bottom=0)

    # Inset: dependency-DB overhead in ms (negligible)
    tasks_oh = [t for t in tasks_ok if task_overhead.get(t, 0) > 0]
    y_oh = [task_overhead[t] for t in tasks_oh]
    ax_inset.bar(range(len(tasks_oh)), y_oh, color=PALETTE[1], alpha=0.85, zorder=3)
    ax_inset.set_xticks(range(len(tasks_oh)))
    mono_ticks(ax_inset, [task_label(t) for t in tasks_oh], fontsize=8)
    ax_inset.set_ylabel("DB query overhead (ms)")
    ax_inset.set_title("LSV lsv_init overhead", fontsize=9)
    ax_inset.set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT_DIR / "R_lsv_time_savings.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT S: DEPS-DB OVERHEAD DISTRIBUTION ────────────────────────────────────


def plot_lsv_overhead_distribution(stats: list):
    """Violin + scatter of deps-DB query time (ms) per task, with savings annotation."""
    import statistics

    by_task = {
        task: [
            s["init_overhead_s"] * 1000
            for s in stats
            if s["task"] == task and s["init_overhead_s"] > 0
        ]
        for task in TASKS
    }

    tasks_ok = [t for t in TASKS if by_task[t]]
    x = np.arange(len(tasks_ok))

    fig, ax = plt.subplots(figsize=(7, 4))

    for xi, task in enumerate(tasks_ok):
        vals = by_task[task]
        color = TASK_COLOR[task]
        if len(vals) >= 3:
            parts = ax.violinplot(
                [vals], positions=[xi], widths=0.5, showmedians=True, showextrema=False
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)
        jitter = np.random.default_rng(xi).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            jitter + xi,
            vals,
            s=18,
            color=color,
            alpha=0.7,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )
        med = statistics.median(vals)
        ax.text(
            xi,
            max(vals) * 1.08,
            f"med={med:.1f}ms",
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

    # Annotate networkx with savings — use axes-fraction coords to avoid overflow
    if "networkx" in tasks_ok:
        nx_xi = tasks_ok.index("networkx")
        nx_rows = [
            s
            for s in stats
            if s["task"] == "networkx"
            and s["selected_expanded"] > 0
            and s["discovered"] > 0
            and s["measure_s"] > 0
        ]
        if nx_rows:
            saved = statistics.median(
                (r["discovered"] / r["selected_expanded"] - 1) * r["measure_s"]
                for r in nx_rows
            )
            ax.annotate(
                f"~{saved:.0f}s saved\n({saved / 60:.0f} min full suite)",
                xy=(nx_xi, statistics.median(by_task["networkx"])),
                xytext=(0.72, 0.82),
                xycoords=("data", "data"),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
            )

    ax.set_xticks(x)
    mono_ticks(ax, [task_label(t) for t in tasks_ok])
    ax.set_ylabel("Dependency-graph lookup time (ms)")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = OUT_DIR / "S_lsv_overhead.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


# ─── PLOT T: REWARD vs RAW LSV SPEEDUP (reward-hacking scatter) ───────────────


def plot_reward_vs_speedup(trials: list):
    """Scatter: raw LSV speedup (x) vs reward (y), highlighting reward=-1 cases."""
    task_colors = {t: PALETTE[i] for i, t in enumerate(TASKS)}

    # Collect points: only trials where benchmarks were actually measured
    points = [t for t in trials if t.n_benchmarks > 0]
    hacks  = [t for t in points if t.reward == -1.0 and t.speedup > 1.0]

    fig, ax = plt.subplots(figsize=(7, 5))

    # One scatter series per task
    for task in TASKS:
        ts = [t for t in points if t.task == task]
        if not ts:
            continue
        xs = [t.speedup for t in ts]
        ys = [t.reward  for t in ts]
        ax.scatter(xs, ys, color=task_colors[task], alpha=0.6, s=40,
                   label=task, zorder=3)

    # Highlight reward-hacking candidates
    if hacks:
        ax.scatter([t.speedup for t in hacks], [t.reward for t in hacks],
                   s=120, facecolors="none", edgecolors="red", linewidths=1.8,
                   zorder=4, label="reward hack (speedup>1, tests fail)")

    ax.axhline(1.0, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(1.0, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Raw LSV mean speedup")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs raw LSV speedup")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = OUT_DIR / "T_reward_vs_speedup.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out}")


def analyze_reward_hacking(trials: list):
    """Print analysis of trials with speedup>1 but reward=-1."""
    hacks = [t for t in trials if t.reward == -1.0 and t.speedup > 1.0]
    if not hacks:
        print("  No reward-hacking candidates found.")
        return

    hacks.sort(key=lambda t: -t.speedup)
    print(f"\n  {len(hacks)} reward-hacking candidates (speedup>1, reward=-1):\n")

    hack_meta = {
        "networkx_agent_31ef96e7": (
            1.309,
            "Rewrites is_aperiodic to handle disconnected graphs by iterating SCCs. "
            "Removes the is_strongly_connected() pre-check (O(n+m)). "
            "New _get_scc_period() helper has buggy BFS logic that returns early with "
            "incorrect values. Speedup comes from skipping the connectivity check; "
            "wrong answers on non-strongly-connected graphs → tests fail."
        ),
        "networkx_agent_99f1d030": (
            3.514,
            "Minimal patch: removes is_strongly_connected() check AND changes g=0 to g=1. "
            "Initialising g=1 means the GCD can never grow above 1, so the BFS loop "
            "exits immediately returning True for almost every graph. "
            "Near-maximum speedup because the entire BFS body is short-circuited. "
            "Produces wrong results for periodic graphs → tests fail."
        ),
        "networkx_agent_c46f07c8": (
            3.487,
            "Removes is_strongly_connected() check and inserts broken BFS stub that "
            "references undefined variables q and visited. Patch is syntactically "
            "malformed — the inserted lines are unreachable dead code. Speedup comes "
            "purely from dropping the connectivity pre-check. Tests fail on correctness."
        ),
        "networkx_agent_da32045c": (
            1.167,
            "Adds two legitimate optimisations: (1) early return True if any self-loop "
            "exists, (2) early exit when g==1 is reached during BFS. Also modifies the "
            "test suite to remove the assertion that is_aperiodic raises for disconnected "
            "graphs — test rewriting to cover up semantic breakage. A residual test that "
            "still checks strongly-connected semantics fails. Smallest speedup of the four "
            "because the optimisations themselves are real; only the test-rewrite is the hack."
        ),
    }

    for t in hacks:
        _, desc = hack_meta.get(t.trial_id, (None, "No detailed analysis available."))
        print(f"  [{t.task}] {t.trial_id}")
        print(f"  speedup={t.speedup:.3f}×  reward={t.reward}  tests_passed={t.tests_passed}")
        print(f"  {desc}")
        print()


# ─── MAIN ──────────────────────────────────────────────────────────────────────


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = load_trials()

    if not trials:
        print("No trials loaded, exiting.")
        return

    debug_file_paths(trials)

    print("\nGenerating plots...")
    print("A1: speedup distributions")
    plot_speedup_distributions(trials)
    print("A2: reward distributions")
    plot_reward_distributions(trials)
    print("B:  agent taxonomy")
    plot_taxonomy(trials)
    print("C:  edit raster")
    plot_edit_raster(trials)
    print("D:  context length")
    plot_context_length(trials)
    print("E:  tool call quintile")
    plot_tool_calls(trials)
    print("F:  patch size vs reward")
    plot_patch_vs_reward(trials)
    print("G:  reward vs turns")
    plot_reward_vs_turns(trials)
    print("H:  pass@k curves")
    plot_passk_curves(trials)
    print("I:  benchmark heatmap")
    plot_benchmark_heatmap(trials)
    print("J:  first-edit CDF")
    plot_first_edit_cdf(trials)
    print("K:  best-of-k reward")
    plot_best_of_k(trials)
    print("L:  metric correlations")
    plot_metric_correlations(trials)
    print("M:  per-benchmark violin")
    plot_per_benchmark_violin(trials)
    print("N:  termination × taxonomy")
    plot_termination_taxonomy(trials)
    print("O:  exploration overhead")
    plot_exploration_overhead(trials)
    print("T:  reward vs speedup (reward hacking)")
    plot_reward_vs_speedup(trials)
    print("T:  reward hacking analysis")
    analyze_reward_hacking(trials)

    print("\nP–S: LSV pipeline plots")
    lsv_stats = load_lsv_stats()
    if lsv_stats:
        print("P:  lsv pipeline timing")
        plot_lsv_pipeline_timing(lsv_stats)
        print("Q:  lsv benchmark funnel")
        plot_lsv_benchmark_funnel(lsv_stats)
        print("R:  lsv time savings")
        plot_lsv_time_savings(lsv_stats)
        print("S:  lsv overhead distribution")
        plot_lsv_overhead_distribution(lsv_stats)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
