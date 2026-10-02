"""
Week 2 – Presentation-Quality EDA Plots
=========================================
CRMLS 6-Month-Focus Sold Data  (Jan–Jun 2025)

Polished visuals designed for a 5–10 min meeting walkthrough.
Saves to ./eda_plots_presentation/
"""

import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ──────────────────────────────────────────────
# 1.  LOAD & CONCATENATE
# ──────────────────────────────────────────────
DATA_DIR = "6month-focus"
OUT_DIR  = "eda_plots_presentation"
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
df["CloseMonthLabel"] = df["CloseDate"].dt.strftime("%b '%y")

PRICE_COLS = ["ClosePrice", "ListPrice", "OriginalListPrice"]
for c in PRICE_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in ["LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
           "DaysOnMarket", "YearBuilt", "LotSizeSquareFeet", "GarageSpaces"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Residential sales only (the model scope per task doc)
res = df[(df["PropertyType"] == "Residential") &
         (df["PropertySubType"] == "SingleFamilyResidence")].copy()
print(f"SFR Residential filter: {len(res):,} rows\n")

# Sale-to-list ratio (useful metric)
res["SaleToListRatio"] = res["ClosePrice"] / res["ListPrice"]
res["PropertyAge"] = 2025 - res["YearBuilt"]
res["PricePerSqFt"] = res["ClosePrice"] / res["LivingArea"]

# ──────────────────────────────────────────────
# 3.  PRESENTATION STYLE
# ──────────────────────────────────────────────
# Dark professional theme
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   16,
    "axes.labelsize":   12,
})

# Accent palette
C_BLUE   = "#3b82f6"
C_CYAN   = "#22d3ee"
C_ORANGE = "#f97316"
C_GREEN  = "#10b981"
C_RED    = "#ef4444"
C_PURPLE = "#a78bfa"
C_PINK   = "#f472b6"
C_YELLOW = "#facc15"
C_SLATE  = "#64748b"

GRAD_BLUES = ["#1e3a5f", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"]

def style_ax(ax, title, subtitle="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=17, fontweight="bold", color="white",
                 loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=10, color="#94a3b8", va="bottom")
    ax.set_xlabel(xlabel, fontsize=11, color="#cbd5e1", labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, color="#cbd5e1", labelpad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#475569")
    ax.tick_params(labelsize=10, colors="#94a3b8")
    ax.grid(axis="y", alpha=0.3, color="#475569", linewidth=0.5)

def add_source_tag(fig):
    fig.text(0.98, 0.01, "Source: CRMLS via Trestle API  |  Jan–Jun 2025",
             fontsize=7, color="#475569", ha="right", va="bottom")

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"  ✓  {path}")


# ══════════════════════════════════════════════
# SLIDE 1 — Dataset Overview Dashboard (4-panel)
# ══════════════════════════════════════════════
print("Generating presentation plots …")

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, hspace=0.35, wspace=0.3)

# Top-left: Monthly volume
ax1 = fig.add_subplot(gs[0, 0])
monthly = res.groupby("CloseMonth").size()
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
bars = ax1.bar(range(len(monthly)), monthly.values,
               color=[C_BLUE]*len(monthly), edgecolor="#1e3a5f",
               width=0.65, zorder=3)
# Highlight peak month
peak_idx = monthly.values.argmax()
bars[peak_idx].set_color(C_CYAN)
bars[peak_idx].set_edgecolor("#0e7490")
style_ax(ax1, "Monthly Sold Volume", ylabel="Sales Count")
ax1.set_xticks(range(len(monthly)))
ax1.set_xticklabels(month_labels)
for i, v in enumerate(monthly.values):
    ax1.text(i, v + 150, f"{v:,}", ha="center", fontsize=9,
             fontweight="bold", color="white")

# Top-right: Median price trend
ax2 = fig.add_subplot(gs[0, 1])
monthly_med = res.groupby("CloseMonth")["ClosePrice"].median()
ax2.plot(range(len(monthly_med)), monthly_med.values,
         color=C_CYAN, linewidth=3, marker="o", markersize=8,
         markerfacecolor="white", markeredgecolor=C_CYAN, markeredgewidth=2, zorder=5)
ax2.fill_between(range(len(monthly_med)), monthly_med.values,
                 alpha=0.15, color=C_CYAN, zorder=2)
style_ax(ax2, "Median Close Price", ylabel="Price ($)")
ax2.set_xticks(range(len(monthly_med)))
ax2.set_xticklabels(month_labels)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
for i, v in enumerate(monthly_med.values):
    ax2.annotate(f"${v/1e3:.0f}K", (i, v), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=9, fontweight="bold",
                 color="white")

# Bottom-left: DOM distribution
ax3 = fig.add_subplot(gs[1, 0])
dom = res["DaysOnMarket"].dropna()
dom_capped = dom[dom.between(0, 200)]
ax3.hist(dom_capped, bins=40, color=C_GREEN, edgecolor="#064e3b", alpha=0.85, zorder=3)
med_dom = dom_capped.median()
ax3.axvline(med_dom, color=C_YELLOW, ls="--", lw=2, zorder=5,
            label=f"Median: {med_dom:.0f} days")
style_ax(ax3, "Days on Market", xlabel="Days", ylabel="Frequency")
ax3.legend(fontsize=9, facecolor="#1e293b", edgecolor="#475569",
           labelcolor="white")

# Bottom-right: Key stats card
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.axis("off")

stats = [
    ("Total SFR Sales",  f"{len(res):,}",        C_BLUE),
    ("Median Price",     f"${res['ClosePrice'].median():,.0f}", C_CYAN),
    ("Median DOM",       f"{med_dom:.0f} days",   C_GREEN),
    ("Median Sq Ft",     f"{res['LivingArea'].median():,.0f}", C_ORANGE),
    ("Median Beds/Bath", f"{res['BedroomsTotal'].median():.0f} / {res['BathroomsTotalInteger'].median():.0f}", C_PURPLE),
    ("Median Year Built", f"{res['YearBuilt'].median():.0f}", C_PINK),
]
y_pos = 9
for label, value, color in stats:
    ax4.text(1, y_pos, label, fontsize=11, color="#94a3b8",
             va="center", fontweight="normal")
    ax4.text(8.5, y_pos, value, fontsize=14, color=color,
             va="center", fontweight="bold", ha="right")
    ax4.plot([0.5, 9.5], [y_pos-0.6, y_pos-0.6],
             color="#334155", linewidth=0.5, zorder=1)
    y_pos -= 1.5

ax4.text(5, 10.2, "Key Metrics (SFR Only)", fontsize=14, fontweight="bold",
         color="white", ha="center")

fig.suptitle("CRMLS Sold Data — Market Overview",
             fontsize=22, fontweight="bold", color="white", y=0.98)
fig.text(0.5, 0.935, "Single-Family Residences  ·  Southern California  ·  Jan – Jun 2025",
         fontsize=11, color="#94a3b8", ha="center")
add_source_tag(fig)
save(fig, "slide1_overview_dashboard.png")


# ══════════════════════════════════════════════
# SLIDE 2 — Price Distribution Deep Dive
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Close price histogram
ax = axes[0]
prices = res["ClosePrice"].dropna()
prices_capped = prices[prices.between(50_000, 3_000_000)]
ax.hist(prices_capped, bins=60, color=C_BLUE, edgecolor="#1e3a5f", alpha=0.85, zorder=3)
med_price = prices_capped.median()
ax.axvline(med_price, color=C_YELLOW, ls="--", lw=2, zorder=5)
ax.text(med_price + 50000, ax.get_ylim()[1]*0.9,
        f"Median\n${med_price/1e3:.0f}K", fontsize=10, color=C_YELLOW,
        fontweight="bold")
style_ax(ax, "Close Price Distribution", xlabel="Close Price", ylabel="Count")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

# Right: Price per sq ft histogram
ax = axes[1]
ppsf = res["PricePerSqFt"].dropna()
ppsf_capped = ppsf[ppsf.between(50, 1500)]
ax.hist(ppsf_capped, bins=50, color=C_PURPLE, edgecolor="#3b0764", alpha=0.85, zorder=3)
med_ppsf = ppsf_capped.median()
ax.axvline(med_ppsf, color=C_YELLOW, ls="--", lw=2, zorder=5)
ax.text(med_ppsf + 20, ax.get_ylim()[1]*0.9,
        f"Median\n${med_ppsf:.0f}/sqft", fontsize=10, color=C_YELLOW,
        fontweight="bold")
style_ax(ax, "Price Per Square Foot", xlabel="$/sq ft", ylabel="Count")

fig.suptitle("Price Distributions — Single-Family Residences",
             fontsize=20, fontweight="bold", color="white", y=1.0)
add_source_tag(fig)
save(fig, "slide2_price_distributions.png")


# ══════════════════════════════════════════════
# SLIDE 3 — What Drives Price? (Scatter grid)
# ══════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

scat = res[res["ClosePrice"].between(50_000, 5_000_000)].copy()

# Living Area vs Price
ax = axes[0, 0]
mask = scat["LivingArea"].between(400, 6000)
ax.scatter(scat.loc[mask, "LivingArea"], scat.loc[mask, "ClosePrice"],
           alpha=0.05, s=4, color=C_CYAN, zorder=3)
med_by = scat.loc[mask].groupby(pd.cut(scat.loc[mask, "LivingArea"],
                                        bins=20))["ClosePrice"].median()
centers = [(iv.left + iv.right)/2 for iv in med_by.index]
ax.plot(centers, med_by.values, color=C_ORANGE, linewidth=3,
        marker="o", markersize=5, zorder=5, label="Median trend")
style_ax(ax, "Living Area vs Price", xlabel="Sq Ft", ylabel="Close Price")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

# Bedrooms vs Price
ax = axes[0, 1]
bed_data = scat[scat["BedroomsTotal"].between(1, 7)]
bed_groups = bed_data.groupby("BedroomsTotal")["ClosePrice"]
bed_med = bed_groups.median()
bed_q25 = bed_groups.quantile(0.25)
bed_q75 = bed_groups.quantile(0.75)
ax.bar(bed_med.index, bed_med.values, color=C_BLUE, edgecolor="#1e3a5f",
       width=0.6, zorder=3)
ax.errorbar(bed_med.index, bed_med.values,
            yerr=[bed_med.values - bed_q25.values, bed_q75.values - bed_med.values],
            fmt="none", ecolor="#94a3b8", capsize=4, capthick=1.5, zorder=5)
style_ax(ax, "Bedrooms vs Median Price", xlabel="Bedrooms", ylabel="Median Price")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
for i, (bed, val) in enumerate(bed_med.items()):
    ax.text(bed, val + 30000, f"${val/1e3:.0f}K", ha="center", fontsize=8,
            color="white", fontweight="bold")

# Year Built vs Price
ax = axes[1, 0]
yb_data = scat[scat["YearBuilt"].between(1920, 2025)]
yb_med = yb_data.groupby(pd.cut(yb_data["YearBuilt"],
                                  bins=20))["ClosePrice"].median()
yb_centers = [(iv.left + iv.right)/2 for iv in yb_med.index]
ax.plot(yb_centers, yb_med.values, color=C_PINK, linewidth=3,
        marker="o", markersize=5, zorder=5)
ax.fill_between(yb_centers, yb_med.values, alpha=0.15, color=C_PINK, zorder=2)
style_ax(ax, "Year Built vs Median Price", xlabel="Year Built", ylabel="Median Price")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

# Lot Size vs Price
ax = axes[1, 1]
lot_data = scat[scat["LotSizeSquareFeet"].between(1000, 50000)]
ax.scatter(lot_data["LotSizeSquareFeet"], lot_data["ClosePrice"],
           alpha=0.04, s=4, color=C_GREEN, zorder=3)
lot_med = lot_data.groupby(pd.cut(lot_data["LotSizeSquareFeet"],
                                   bins=20))["ClosePrice"].median()
lot_centers = [(iv.left + iv.right)/2 for iv in lot_med.index]
ax.plot(lot_centers, lot_med.values, color=C_ORANGE, linewidth=3,
        marker="o", markersize=5, zorder=5, label="Median trend")
style_ax(ax, "Lot Size vs Price", xlabel="Lot Size (sq ft)", ylabel="Close Price")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

fig.suptitle("What Drives Close Price?",
             fontsize=20, fontweight="bold", color="white", y=1.0)
fig.text(0.5, 0.96, "Each dot = one SFR sale  ·  Orange/pink lines = median trend",
         fontsize=10, color="#94a3b8", ha="center")
add_source_tag(fig)
plt.tight_layout(rect=[0, 0.02, 1, 0.95])
save(fig, "slide3_price_drivers.png")


# ══════════════════════════════════════════════
# SLIDE 4 — Geographic & Market Insights
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Left: Top 12 cities by volume with median price overlay
ax = axes[0]
top_cities = res["City"].value_counts().head(12)
city_med_price = res[res["City"].isin(top_cities.index)].groupby("City")["ClosePrice"].median()
city_med_price = city_med_price.reindex(top_cities.index)

# Horizontal bars sorted by volume
y_positions = range(len(top_cities))
bars = ax.barh(y_positions, top_cities.values[::-1],
               color=C_BLUE, edgecolor="#1e3a5f", height=0.65, zorder=3)
ax.set_yticks(y_positions)
ax.set_yticklabels(top_cities.index[::-1], fontsize=9)
style_ax(ax, "Top 12 Cities by Volume", xlabel="Number of Sales")

# Add price annotations
for i, (city, vol) in enumerate(zip(top_cities.index[::-1], top_cities.values[::-1])):
    med_p = city_med_price.get(city, 0)
    ax.text(vol + 50, i, f"${med_p/1e3:.0f}K med",
            va="center", fontsize=8, color=C_CYAN, fontweight="bold")

# Right: Sale-to-List ratio by month
ax = axes[1]
stl = res[res["SaleToListRatio"].between(0.8, 1.2)]
monthly_stl = stl.groupby("CloseMonth")["SaleToListRatio"].agg(["median", "mean"])
x = range(len(monthly_stl))

ax.bar(x, (monthly_stl["median"] - 1) * 100, bottom=100,
       color=[C_GREEN if v >= 1 else C_RED for v in monthly_stl["median"]],
       edgecolor="#1e293b", width=0.55, zorder=3)
ax.axhline(100, color="#94a3b8", linewidth=1, ls="-", zorder=2)
style_ax(ax, "Sale-to-List Price Ratio", xlabel="", ylabel="Ratio (%)")
ax.set_xticks(x)
ax.set_xticklabels(month_labels[:len(x)])
for i, v in enumerate(monthly_stl["median"]):
    color = C_GREEN if v >= 1 else C_RED
    ax.text(i, v*100 + (0.15 if v >= 1 else -0.25),
            f"{v*100:.1f}%", ha="center", fontsize=9,
            fontweight="bold", color=color)

fig.suptitle("Market Geography & Pricing Power",
             fontsize=20, fontweight="bold", color="white", y=1.0)
add_source_tag(fig)
save(fig, "slide4_geography_market.png")


# ══════════════════════════════════════════════
# SLIDE 5 — Correlation Heatmap (polished)
# ══════════════════════════════════════════════
heat_cols = ["ClosePrice", "ListPrice", "LivingArea",
             "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket",
             "YearBuilt", "LotSizeSquareFeet", "GarageSpaces"]
nice_labels = ["Close Price", "List Price", "Living Area",
               "Bedrooms", "Bathrooms", "Days on Market",
               "Year Built", "Lot Size", "Garage Spaces"]
corr = res[heat_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(nice_labels)))
ax.set_yticks(range(len(nice_labels)))
ax.set_xticklabels(nice_labels, rotation=40, ha="right", fontsize=9)
ax.set_yticklabels(nice_labels, fontsize=9)

for i in range(len(heat_cols)):
    for j in range(len(heat_cols)):
        val = corr.iloc[i, j]
        color = "white" if abs(val) > 0.5 else "#e2e8f0"
        weight = "bold" if abs(val) > 0.4 else "normal"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color=color, fontweight=weight)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, aspect=30)
cbar.set_label("Pearson Correlation", color="#94a3b8", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="#94a3b8")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#94a3b8")

fig.suptitle("Feature Correlation Matrix",
             fontsize=20, fontweight="bold", color="white", y=0.97)
fig.text(0.5, 0.93, "Key insight: Living Area (r=0.55) is the strongest non-leaky predictor of Close Price",
         fontsize=10, color=C_CYAN, ha="center", fontstyle="italic")
add_source_tag(fig)
save(fig, "slide5_correlations.png")


# ══════════════════════════════════════════════
# SLIDE 6 — Data Quality & Next Steps
# ══════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Missing data for key modeling columns
ax = axes[0]
model_cols = ["ClosePrice", "ListPrice", "LivingArea", "BedroomsTotal",
              "BathroomsTotalInteger", "YearBuilt", "LotSizeSquareFeet",
              "GarageSpaces", "DaysOnMarket", "PostalCode", "City",
              "Latitude", "Longitude", "Stories", "AssociationFee",
              "PoolPrivateYN", "FireplaceYN", "ViewYN"]
nice_model = ["Close Price", "List Price", "Living Area", "Bedrooms",
              "Bathrooms", "Year Built", "Lot Size (sqft)",
              "Garage Spaces", "Days on Market", "ZIP Code", "City",
              "Latitude", "Longitude", "Stories", "HOA Fee",
              "Pool", "Fireplace", "View"]

completeness = [(1 - res[c].isnull().mean()) * 100 for c in model_cols]
colors = [C_GREEN if v >= 90 else C_ORANGE if v >= 70 else C_RED for v in completeness]

y_pos = range(len(model_cols))
ax.barh(y_pos, [c for c in completeness][::-1],
        color=[c for c in colors][::-1], edgecolor="#1e293b", height=0.6, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(nice_model[::-1], fontsize=9)
ax.axvline(90, color=C_YELLOW, ls="--", lw=1, alpha=0.5, zorder=2)
ax.set_xlim(0, 105)
style_ax(ax, "Data Completeness", xlabel="% Non-Null")

for i, v in enumerate(completeness[::-1]):
    ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8,
            color="white", fontweight="bold")

# Right: Next steps card
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

steps = [
    ("Week 3", "Data Preprocessing", "Handle missing values, encode categoricals,\ncreate chronological train/test split"),
    ("Week 4", "Baseline Model", "Linear Regression on key features,\nestablish R² baseline"),
    ("Week 5", "Model Comparison", "Decision Tree & Random Forest,\ncompare against baseline"),
    ("Week 6", "Feature Engineering", "Price/sqft, property age, school district\nspatial join, sale-to-list ratio"),
]

y = 9.2
ax.text(5, 10, "Roadmap — Next Steps", fontsize=16, fontweight="bold",
        color="white", ha="center")

for week, title, desc in steps:
    # Week badge
    ax.add_patch(plt.Rectangle((0.3, y-0.5), 1.8, 0.9,
                                facecolor=C_BLUE, edgecolor="none",
                                alpha=0.8, zorder=3,
                                transform=ax.transData))
    ax.text(1.2, y, week, fontsize=9, color="white", fontweight="bold",
            ha="center", va="center", zorder=5)
    ax.text(2.5, y + 0.1, title, fontsize=12, color="white", fontweight="bold")
    ax.text(2.5, y - 0.45, desc, fontsize=8, color="#94a3b8", va="top")
    y -= 2.2

fig.suptitle("Data Quality & Next Steps",
             fontsize=20, fontweight="bold", color="white", y=1.0)
add_source_tag(fig)
save(fig, "slide6_quality_next_steps.png")


# ──────────────────────────────────────────────
print(f"\n✅  All 6 presentation slides saved to ./{OUT_DIR}/")
