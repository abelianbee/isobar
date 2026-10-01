"""Figures for the README, built from the release bundle in results/release/.

Run after `make reproduce`:  python3 scripts/make_figures.py
Writes PNGs to docs/img/. Everything here reads the frozen trade files, so a
figure can never disagree with RESULTS.md without the build failing first.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from edge_lib import ROOT as EDGE_ROOT

OUT = EDGE_ROOT / "docs" / "img"
REL = EDGE_ROOT / "results" / "release"
DPI = 200

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8c8b86"
BLUE = "#2a78d6"
GREY = "#b9b8b2"
SHADE = "#eceae4"

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.edgecolor": MUTED,
    }
)


def _clean(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
        ax.spines[s].set_color(MUTED)
    ax.grid(axis=grid_axis, color=SHADE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", (OUT / name).relative_to(EDGE_ROOT))


def clock():
    """One city-day: where the information cutoff sits. Hours from midday the day before."""
    fig, ax = plt.subplots(figsize=(8.4, 2.5))
    ax.set_xlim(-1.5, 49.5)
    ax.set_ylim(-0.95, 1.05)
    ax.axis("off")

    ax.axhline(0, color=MUTED, linewidth=1, zorder=2)
    for h in (0, 12, 24, 36, 48):
        ax.plot([h, h], [-0.05, 0], color=MUTED, linewidth=1, zorder=2)
        ax.text(h, -0.14, "12:00" if h % 24 == 0 else "00:00", color=MUTED,
                fontsize=8, ha="center", va="top")
    for h, lab in ((6, "day before"), (24, "the day itself"), (42, "day after")):
        ax.text(h, -0.32, lab, color=INK2, fontsize=8, ha="center", va="top")

    # the evening model runs become public
    ax.add_patch(plt.Rectangle((7, 0), 2, 0.10, color=GREY, zorder=3))
    ax.text(9.6, 0.17, "13 model runs publish", color=INK2, fontsize=9, ha="right")

    # the decision
    ax.plot([10, 10], [0, 0.98], color=BLUE, linewidth=1.6, zorder=3)
    ax.plot([10], [0], marker="o", markersize=6, color=BLUE, zorder=4)
    ax.text(10.5, 0.96, "22:00 UTC: the trade is decided", color=BLUE, fontsize=10,
            fontweight="bold", va="top")
    ax.text(10.5, 0.76, "the blend against the bracket's own price, taken only",
            color=INK2, fontsize=8.5, va="top")
    ax.text(10.5, 0.62, "if the two disagree by more than 5c after the fee",
            color=INK2, fontsize=8.5, va="top")

    # the high, then settlement
    ax.add_patch(plt.Rectangle((29, 0), 5, 0.10, color=GREY, zorder=3))
    ax.text(31.5, 0.17, "the high happens", color=INK2, fontsize=9, ha="center")
    ax.plot([42, 42], [0, 0.10], color=INK2, linewidth=1, zorder=3)
    ax.plot([42], [0], marker="o", markersize=4.5, color=INK2, zorder=4)
    ax.text(42, 0.17, "NWS report settles", color=INK2, fontsize=9, ha="center")

    # what the trade is allowed to see
    ax.plot([0, 10], [-0.50, -0.50], color=BLUE, linewidth=1, linestyle=(0, (3, 3)), zorder=3)
    ax.text(10.6, -0.50, "everything the trade is allowed to know", color=INK2,
            fontsize=8.5, va="center")

    ax.text(-1.5, -0.74, "The thirteen models are NBM, ICON, GEM, ECMWF, GFS, HRRR, UKMO, "
            "MeteoFrance, JMA, KNMI, DMI, MetNo and AIFS.", color=MUTED, fontsize=8, va="top")
    ax.text(-1.5, -0.88, "Those evening runs are public by 22:00 in the Apr-Sep archive. "
            "For winter that is an assumption, not a measurement.", color=INK2, fontsize=8, va="top")
    ax.set_title("One city-day, and where the information cutoff sits", color=INK,
                 fontsize=10.5, loc="left", pad=10)
    _save(fig, "clock.png")


def walkforward():
    """One row per monthly refit: what it was fitted on, what it then traded."""
    t = pd.read_parquet(REL / "weather_trades.parquet")
    t["target"] = pd.to_datetime(t["target"])
    t["train_cutoff"] = pd.to_datetime(t["train_cutoff"])
    g = (
        t.groupby("train_cutoff")
        .agg(first=("target", "min"), last=("target", "max"), n=("pnl", "size"))
        .sort_index()
    )

    left = pd.Timestamp("2024-10-15")
    fig, ax = plt.subplots(figsize=(8.4, 0.235 * len(g) + 1.5))
    for i, (cut, r) in enumerate(g.iterrows()):
        y = len(g) - i
        ax.barh(y, cut - left, left=left, height=0.62, color=GREY, zorder=3)
        ax.barh(
            y,
            max(r["last"] - r["first"], pd.Timedelta(days=1)),
            left=r["first"],
            height=0.62,
            color=BLUE,
            zorder=3,
        )

    ax.set_ylim(0.3, len(g) + 0.9)
    ax.set_xlim(left, pd.Timestamp("2026-10-20"))
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    _clean(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)

    first = g.iloc[0]
    ax.annotate(
        "fitted on prior data only",
        xy=(left + (first.name - left) / 2, len(g) + 0.55),
        color=INK2,
        fontsize=8.5,
        ha="center",
    )
    ax.annotate(
        "then traded this month",
        xy=(first["last"] + pd.Timedelta(days=6), len(g) + 0.55),
        color=BLUE,
        fontsize=8.5,
        ha="left",
        fontweight="bold",
    )
    ax.set_title(
        f"Every monthly refit of the logistic ({len(g)} of them, {len(t):,} trades)",
        color=INK,
        fontsize=10.5,
        loc="left",
        pad=26,
    )
    ax.set_xlabel("Each row is one refit. Nothing to the right of a grey bar was visible when it was fitted.")
    _save(fig, "walkforward.png")


def equity():
    """Strategy 1: cumulative P&L at one contract per signal."""
    t = pd.read_parquet(REL / "weather_trades.parquet").sort_values("decision_utc")
    t["d"] = pd.to_datetime(t["decision_utc"]).dt.tz_localize(None)
    cum = t["pnl"].cumsum()

    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    for a, b in [("2024-11-01", "2025-04-01"), ("2025-11-01", "2026-04-01")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=SHADE, zorder=0)
    ax.plot(t["d"], cum, color=BLUE, linewidth=1.6, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=2)

    _clean(ax)
    ax.set_xlim(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-12-15"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("cumulative $, one contract per signal")
    ax.annotate(
        f"${cum.iloc[-1]:,.0f}\non {len(t):,} trades",
        xy=(t["d"].iloc[-1], cum.iloc[-1]),
        xytext=(7, -2),
        textcoords="offset points",
        color=BLUE,
        fontweight="bold",
        ha="left",
        va="center",
        fontsize=9.5,
    )
    ax.annotate(
        "shaded: Nov-Mar, where the edge lives",
        xy=(pd.Timestamp("2025-12-05"), cum.max() * 0.08),
        color=INK2,
        fontsize=8.5,
    )
    ax.set_title(
        "Strategy 1: forecast blend against Kalshi daily-high brackets",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    ax.set_xlabel("Historical simulation. Fills modelled at the quoted candle ask/bid plus the taker fee, not demonstrated.")
    _save(fig, "equity.png")


def holdout():
    """Strategy 2: the specification was fixed before the holdout data existed."""
    t = pd.read_parquet(REL / "xvenue_trades.parquet").sort_values("ts")
    t["d"] = pd.to_datetime(t["ts"]).dt.tz_localize(None)
    t["cum"] = t["pnl"].cumsum()
    # the DEV/HOLDOUT label comes from the release build, so these counts cannot
    # drift away from RESULTS.md
    dev = t[t["dev_or_holdout"] == "DEV"]
    hold = t[t["dev_or_holdout"] == "HOLDOUT"]
    split = dev["d"].max() + (hold["d"].min() - dev["d"].max()) / 2

    fig, ax = plt.subplots(figsize=(8.4, 3.1))
    ax.plot(dev["d"], dev["cum"], color=GREY, linewidth=1.6, zorder=3)
    ax.plot(
        pd.concat([dev["d"].tail(1), hold["d"]]),
        pd.concat([dev["cum"].tail(1), hold["cum"]]),
        color=BLUE,
        linewidth=1.6,
        zorder=4,
    )
    ax.axvline(split, color=INK2, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=2)

    _clean(ax)
    ax.set_xlim(pd.Timestamp("2025-03-01"), pd.Timestamp("2026-10-10"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("cumulative $, one contract per signal")
    ax.annotate(
        "specification fixed here;\nthe Apr-Sep data was pulled afterwards",
        xy=(split, ax.get_ylim()[1] * 0.42),
        xytext=(10, 0),
        textcoords="offset points",
        color=INK2,
        fontsize=8.5,
        va="center",
    )
    ax.legend(
        handles=[
            Patch(color=GREY, label=f"development, {len(dev):,} trades"),
            Patch(color=BLUE, label=f"holdout, {len(hold):,} trades"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=8.5,
        labelcolor=INK2,
        handlelength=1.1,
        handleheight=1.1,
    )
    ax.set_title(
        "Strategy 2: Kalshi's price traded against Polymarket's brackets",
        color=INK,
        fontsize=10.5,
        loc="left",
    )
    ax.set_xlabel("The chronology is self-reported: git history on this repo only starts at the release build.")
    _save(fig, "holdout.png")


if __name__ == "__main__":
    clock()
    walkforward()
    equity()
    holdout()
