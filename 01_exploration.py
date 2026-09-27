"""
Week 2 – Exploratory Data Analysis (EDA)
=========================================
CRMLS 6-Month-Focus Sold Data  (Jan–Jun 2025)

Generates basic EDA plots using pandas + matplotlib.
All figures are saved to  ./eda_plots/
"""

import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ──────────────────────────────────────────────
# 1.  LOAD & CONCATENATE
# ──────────────────────────────────────────────
DATA_DIR = "6month-focus"
OUT_DIR  = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATA_DIR, "CRMLSSold2025*.csv")))
print(f"Loading {len(files)} files …")

dfs = []
for f in files:
    tmp = pd.read_csv(f, low_memory=False)
    dfs.append(tmp)
    print(f"  {os.path.basename(f):40s}  →  {len(tmp):>6,} rows")

df = pd.concat(dfs, ignore_index=True)
print(f"\nCombined: {len(df):,} rows × {len(df.columns)} columns\n")

# ──────────────────────────────────────────────
# 2.  QUICK CLEANING / PREP
# ──────────────────────────────────────────────
df["CloseDate"] = pd.to_datetime(df["CloseDate"], errors="coerce")
df["CloseMonth"] = df["CloseDate"].dt.to_period("M")

# Numeric cols we'll plot
PRICE_COLS = ["ClosePrice", "ListPrice", "OriginalListPrice"]
for c in PRICE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
           "DaysOnMarket", "YearBuilt", "LotSizeSquareFeet", "GarageSpaces"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Filter to residential-only for most plots (exclude commercial / biz-opp)
res = df[df["PropertyType"].isin(["Residential", "ResidentialLease"])].copy()

# ──────────────────────────────────────────────
# Style helper
# ──────────────────────────────────────────────
COLORS = ["#2563eb", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#ec4899"]

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {path}")


# ══════════════════════════════════════════════
# PLOT 1 — Monthly Sold Volume (bar chart)
# ══════════════════════════════════════════════
print("\nGenerating plots …")
monthly = df.groupby("CloseMonth").size()

fig, ax = plt.subplots(figsize=(8, 4))
monthly.plot.bar(ax=ax, color=COLORS[0], edgecolor="white", width=0.7)
style_ax(ax, "Monthly Sold Volume (Jan–Jun 2025)", "Close Month", "Number of Sales")
ax.set_xticklabels([str(p) for p in monthly.index], rotation=0)
for i, v in enumerate(monthly.values):
    ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=8, fontweight="bold")
save(fig, "01_monthly_sold_volume.png")


# ══════════════════════════════════════════════
# PLOT 2 — Close Price Distribution (histogram)
# ══════════════════════════════════════════════
# Cap at $3M to keep the histogram readable
prices = res["ClosePrice"].dropna()
prices_capped = prices[prices.between(1, 3_000_000)]

fig, ax = plt.subplots(figsize=(8, 4))
prices_capped.plot.hist(bins=60, ax=ax, color=COLORS[0], edgecolor="white", alpha=0.85)
style_ax(ax, "Close Price Distribution (Residential, capped at $3 M)",
         "Close Price ($)", "Frequency")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.axvline(prices_capped.median(), color=COLORS[1], ls="--", lw=1.5,
           label=f"Median ${prices_capped.median():,.0f}")
ax.legend(fontsize=9)
save(fig, "02_close_price_distribution.png")


# ══════════════════════════════════════════════
# PLOT 3 — Price by Property Sub-Type (box plot, top 8)
# ══════════════════════════════════════════════
top_subtypes = res["PropertySubType"].value_counts().head(8).index
sub_df = res[res["PropertySubType"].isin(top_subtypes) & res["ClosePrice"].between(1, 3_000_000)]

fig, ax = plt.subplots(figsize=(10, 5))
sub_df.boxplot(column="ClosePrice", by="PropertySubType", ax=ax,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=COLORS[0], alpha=0.6),
               medianprops=dict(color=COLORS[1], linewidth=2))
ax.set_title("Close Price by Property Sub-Type (top 8)", fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Close Price ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
plt.suptitle("")  # remove pandas auto-title
plt.xticks(rotation=30, ha="right", fontsize=9)
save(fig, "03_price_by_subtype_boxplot.png")


# ══════════════════════════════════════════════
# PLOT 4 — Days on Market Distribution
# ══════════════════════════════════════════════
dom = res["DaysOnMarket"].dropna()
dom_capped = dom[dom.between(0, 365)]

fig, ax = plt.subplots(figsize=(8, 4))
dom_capped.plot.hist(bins=50, ax=ax, color=COLORS[2], edgecolor="white", alpha=0.85)
style_ax(ax, "Days on Market Distribution (0–365 days)",
         "Days on Market", "Frequency")
ax.axvline(dom_capped.median(), color=COLORS[1], ls="--", lw=1.5,
           label=f"Median {dom_capped.median():.0f} days")
ax.legend(fontsize=9)
save(fig, "04_days_on_market_hist.png")


# ══════════════════════════════════════════════
# PLOT 5 — Bedrooms vs Close Price (scatter)
# ══════════════════════════════════════════════
scat = res[res["ClosePrice"].between(1, 5_000_000) & res["BedroomsTotal"].between(1, 8)].copy()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(scat["BedroomsTotal"], scat["ClosePrice"],
           alpha=0.08, s=8, color=COLORS[0])
# overlay median line
med_by_bed = scat.groupby("BedroomsTotal")["ClosePrice"].median()
ax.plot(med_by_bed.index, med_by_bed.values, color=COLORS[1],
        marker="o", linewidth=2.5, markersize=6, label="Median")
style_ax(ax, "Bedrooms vs. Close Price", "Bedrooms", "Close Price ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend(fontsize=9)
save(fig, "05_bedrooms_vs_price.png")


# ══════════════════════════════════════════════
# PLOT 6 — Living Area vs Close Price (scatter)
# ══════════════════════════════════════════════
scat2 = res[res["ClosePrice"].between(1, 5_000_000) & res["LivingArea"].between(100, 6000)].copy()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(scat2["LivingArea"], scat2["ClosePrice"],
           alpha=0.06, s=6, color=COLORS[4])
style_ax(ax, "Living Area vs. Close Price", "Living Area (sq ft)", "Close Price ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
# Add a trend line via numpy polyfit
z = np.polyfit(scat2["LivingArea"], scat2["ClosePrice"], 1)
xs = np.linspace(100, 6000, 100)
ax.plot(xs, np.polyval(z, xs), color=COLORS[1], lw=2,
        label=f"Trend: ~${z[0]:,.0f}/sq ft")
ax.legend(fontsize=9)
save(fig, "06_living_area_vs_price.png")


# ══════════════════════════════════════════════
# PLOT 7 — Year Built Distribution
# ══════════════════════════════════════════════
yb = res["YearBuilt"].dropna()
yb = yb[yb.between(1900, 2026)]

fig, ax = plt.subplots(figsize=(8, 4))
yb.plot.hist(bins=60, ax=ax, color=COLORS[3], edgecolor="white", alpha=0.85)
style_ax(ax, "Year Built Distribution (Sold Properties)", "Year Built", "Frequency")
ax.axvline(yb.median(), color=COLORS[0], ls="--", lw=1.5,
           label=f"Median {yb.median():.0f}")
ax.legend(fontsize=9)
save(fig, "07_year_built_distribution.png")


# ══════════════════════════════════════════════
# PLOT 8 — Top 15 Cities by Volume
# ══════════════════════════════════════════════
top_cities = res["City"].value_counts().head(15)

fig, ax = plt.subplots(figsize=(8, 5))
top_cities[::-1].plot.barh(ax=ax, color=COLORS[0], edgecolor="white")
style_ax(ax, "Top 15 Cities by Sold Volume", "Number of Sales", "")
for i, v in enumerate(top_cities[::-1].values):
    ax.text(v + 50, i, f"{v:,}", va="center", fontsize=8)
save(fig, "08_top_cities_volume.png")


# ══════════════════════════════════════════════
# PLOT 9 — Median Price by Month (line chart)
# ══════════════════════════════════════════════
# Residential sales only (not leases)
sales_only = df[df["PropertyType"] == "Residential"].copy()
monthly_med = sales_only.groupby("CloseMonth")["ClosePrice"].median()

fig, ax = plt.subplots(figsize=(8, 4))
monthly_med.plot(ax=ax, marker="o", color=COLORS[0], linewidth=2.5, markersize=7)
style_ax(ax, "Median Close Price by Month (Residential Sales Only)",
         "Month", "Median Close Price ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
for i, (idx, v) in enumerate(monthly_med.items()):
    ax.annotate(f"${v/1e3:.0f}K", (i, v), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=8, fontweight="bold")
save(fig, "09_median_price_by_month.png")


# ══════════════════════════════════════════════
# PLOT 10 — Correlation Heatmap (key numerics)
# ══════════════════════════════════════════════
heat_cols = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
             "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket",
             "YearBuilt", "LotSizeSquareFeet", "GarageSpaces"]
corr = res[heat_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(heat_cols)))
ax.set_yticks(range(len(heat_cols)))
ax.set_xticklabels(heat_cols, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(heat_cols, fontsize=8)
# Annotate cells
for i in range(len(heat_cols)):
    for j in range(len(heat_cols)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color=color)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Correlation Heatmap — Key Numeric Features",
             fontsize=13, fontweight="bold", pad=12)
save(fig, "10_correlation_heatmap.png")


# ──────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────
print(f"\n✅  All 10 EDA plots saved to ./{OUT_DIR}/")
