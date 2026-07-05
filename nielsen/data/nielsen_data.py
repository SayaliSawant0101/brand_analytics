"""
nielsen_data.py  (v2 — data-driven market classification)
----------------------------------------------------------
Market type is NO longer hardcoded. Each market gets a latent
trend drawn from a calibrated distribution. The notebooks
discover and classify declining / stable / growing markets
by computing the slope of Luminos dollar share over time.

Seed = 42 produces a realistic split: ~3 declining, ~2 growing,
~5 stable (verified in validation block below).

Output files (save to nielsen/data/):
  nielsen_weekly.csv        Weekly sales + share + velocity
  nielsen_distribution.csv  ACV by brand x market x channel x week
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ─────────────────────────────────────────────
SEED = 37
rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path("nielsen/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dimensions ───────────────────────────────────────────────────
BRANDS = ["Luminos", "Dove", "Suave", "Pantene", "Private Label"]

MARKETS = {
    "New York":     "Northeast",
    "Los Angeles":  "West",
    "Chicago":      "Midwest",
    "Houston":      "South",
    "Philadelphia": "Northeast",
    "Phoenix":      "West",
    "San Antonio":  "South",
    "Dallas":       "South",
    "Detroit":      "Midwest",
    "Seattle":      "West",
}

CHANNELS = ["Food", "Mass", "Drug"]

START_DATE = pd.Timestamp("2023-01-01")
WEEKS = pd.date_range(START_DATE, periods=104, freq="W-SAT")

# ── Latent Market Trends for Luminos ────────────────────────────
# Draw a 2-year cumulative trend (multiplier) per market from a
# distribution calibrated to produce ~3 declining, ~2 growing,
# ~5 stable markets. Declining = trend < 0.88, growing = > 1.08.
#
# We draw from a slightly left-skewed normal so Luminos as a
# challenger brand has more headwinds than tailwinds — realistic.

raw_trends = rng.normal(loc=0.97, scale=0.12, size=len(MARKETS))
# Clip to a sensible range: no market loses more than 25% or gains more than 20%
raw_trends = np.clip(raw_trends, 0.75, 1.20)

MARKET_TRENDS = {
    market: float(trend)
    for market, trend in zip(MARKETS.keys(), raw_trends)
}

# ── Brand Parameters ─────────────────────────────────────────────
BRAND_BASE_SALES = {
    "Luminos":       1_800,
    "Dove":          3_400,
    "Suave":         2_200,
    "Pantene":       2_800,
    "Private Label": 1_500,
}

BRAND_BASE_PRICE = {
    "Luminos":       6.49,
    "Dove":          7.29,
    "Suave":         4.99,
    "Pantene":       7.99,
    "Private Label": 3.49,
}

BRAND_BASE_ACV = {
    "Luminos":       0.72,
    "Dove":          0.94,
    "Suave":         0.91,
    "Pantene":       0.89,
    "Private Label": 0.85,
}

BRAND_PROMO_FREQ = {
    "Luminos":       0.28,
    "Dove":          0.32,
    "Suave":         0.38,
    "Pantene":       0.30,
    "Private Label": 0.20,
}

BRAND_PROMO_DEPTH = {
    "Luminos":       0.18,
    "Dove":          0.22,
    "Suave":         0.25,
    "Pantene":       0.20,
    "Private Label": 0.15,
}

BRAND_PROMO_LIFT = {
    "Luminos":       1.35,
    "Dove":          1.55,
    "Suave":         1.45,
    "Pantene":       1.50,
    "Private Label": 1.20,
}

CHANNEL_SALES_MULT = {"Food": 1.0, "Mass": 1.4, "Drug": 0.6}

# ── Trend & Seasonal Functions ────────────────────────────────────

def luminos_trend_multiplier(week_idx: int, cumulative_trend: float) -> float:
    """
    Linear interpolation from 1.0 at week 0 to cumulative_trend at week 103.
    e.g. cumulative_trend=0.82 means Luminos ends 18% below its starting sales.
    """
    t = week_idx / 103
    return 1.0 + (cumulative_trend - 1.0) * t


def luminos_acv_trend(week_idx: int, cumulative_trend: float, base_acv: float) -> float:
    """
    ACV moves in the same direction as sales trend but at ~40% the magnitude.
    Declining sales -> stores gradually reduce shelf presence.
    """
    t = week_idx / 103
    acv_delta = (cumulative_trend - 1.0) * 0.40 * t
    return float(np.clip(base_acv + acv_delta, 0.30, 0.99))


def seasonal_multiplier(week_idx: int) -> float:
    """
    Personal care: mild Aug/Sep peak (back-to-school), slight Feb dip.
    """
    week_of_year = week_idx % 52
    return 1.0 + 0.08 * np.sin(2 * np.pi * (week_of_year - 10) / 52)


# ── Data Generation ───────────────────────────────────────────────

def generate_sales_data() -> pd.DataFrame:
    rows = []

    for week_idx, week_end in enumerate(WEEKS):
        seasonal = seasonal_multiplier(week_idx)

        for market, region in MARKETS.items():
            luminos_cumulative_trend = MARKET_TRENDS[market]

            for channel in CHANNELS:
                ch_mult = CHANNEL_SALES_MULT[channel]

                for brand in BRANDS:
                    base        = BRAND_BASE_SALES[brand]
                    price       = BRAND_BASE_PRICE[brand]
                    promo_freq  = BRAND_PROMO_FREQ[brand]
                    promo_depth = BRAND_PROMO_DEPTH[brand]
                    promo_lift  = BRAND_PROMO_LIFT[brand]

                    # Trend: only Luminos has a market-specific trend
                    # Private Label gets a mild counter-trend (gains when Luminos loses)
                    if brand == "Luminos":
                        trend = luminos_trend_multiplier(week_idx, luminos_cumulative_trend)
                    elif brand == "Private Label":
                        # counter-trend: gains where Luminos loses
                        counter = 1.0 + (1.0 - luminos_cumulative_trend) * 0.30
                        trend = luminos_trend_multiplier(week_idx, counter)
                    else:
                        trend = 1.0 + rng.normal(0, 0.004)

                    # Promotion
                    on_promo = rng.random() < promo_freq
                    if on_promo:
                        promo_price        = price * (1 - promo_depth)
                        unit_multiplier    = promo_lift
                        pct_sales_on_promo = rng.uniform(0.55, 0.85)
                    else:
                        promo_price        = price
                        unit_multiplier    = 1.0
                        pct_sales_on_promo = 0.0

                    noise        = rng.normal(1.0, 0.04)
                    dollar_sales = max(base * ch_mult * trend * seasonal * unit_multiplier * noise, 0)

                    effective_price = (
                        promo_price * pct_sales_on_promo + price * (1 - pct_sales_on_promo)
                        if on_promo else price
                    )
                    unit_sales = dollar_sales / effective_price

                    rows.append({
                        "week_end_date":      week_end.date(),
                        "week_index":         week_idx,
                        "market":             market,
                        "region":             region,
                        "channel":            channel,
                        "brand":              brand,
                        "dollar_sales":       round(dollar_sales, 2),
                        "unit_sales":         round(unit_sales, 2),
                        "regular_price":      price,
                        "effective_price":    round(effective_price, 4),
                        "on_promo":           on_promo,
                        "pct_sales_on_promo": round(pct_sales_on_promo, 4),
                    })

    return pd.DataFrame(rows)


def generate_distribution_data() -> pd.DataFrame:
    rows = []

    for week_idx, week_end in enumerate(WEEKS):
        for market, region in MARKETS.items():
            luminos_cumulative_trend = MARKET_TRENDS[market]

            for channel in CHANNELS:
                for brand in BRANDS:
                    base_acv = BRAND_BASE_ACV[brand]

                    # Luminos: Drug channel penalty + market-specific ACV trend
                    if brand == "Luminos":
                        if channel == "Drug":
                            base_acv = np.clip(base_acv - 0.12, 0.30, 0.99)
                        acv = luminos_acv_trend(week_idx, luminos_cumulative_trend, float(base_acv))
                    else:
                        acv = float(np.clip(base_acv + rng.normal(0, 0.008), 0.50, 0.99))

                    acv = float(np.clip(acv + rng.normal(0, 0.005), 0.20, 0.99))

                    rows.append({
                        "week_end_date":    week_end.date(),
                        "week_index":       week_idx,
                        "market":           market,
                        "region":           region,
                        "channel":          channel,
                        "brand":            brand,
                        "acv_distribution": round(acv, 4),
                    })

    return pd.DataFrame(rows)


def compute_share_and_velocity(sales_df: pd.DataFrame, dist_df: pd.DataFrame) -> pd.DataFrame:
    # Category totals per week x market x channel
    totals = (
        sales_df
        .groupby(["week_end_date", "week_index", "market", "region", "channel"])[
            ["dollar_sales", "unit_sales"]
        ]
        .sum()
        .rename(columns={"dollar_sales": "total_dollar_sales",
                          "unit_sales":   "total_unit_sales"})
        .reset_index()
    )

    df = sales_df.merge(totals, on=["week_end_date", "week_index", "market", "region", "channel"])
    df["dollar_share"] = (df["dollar_sales"] / df["total_dollar_sales"]).round(4)
    df["unit_share"]   = (df["unit_sales"]   / df["total_unit_sales"]).round(4)

    # Merge ACV
    dist_cols = ["week_end_date", "market", "channel", "brand", "acv_distribution"]
    df = df.merge(dist_df[dist_cols], on=["week_end_date", "market", "channel", "brand"])

    # Velocity = dollar sales per point of ACV (per 1% weighted distribution)
    df["velocity"] = (df["dollar_sales"] / (df["acv_distribution"] * 100)).round(4)

    return df


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating Nielsen synthetic data (v2 — data-driven market classification)")
    print(f"  Seed: {SEED}")
    print(f"  Brands: {BRANDS}")
    print(f"  Markets: {list(MARKETS.keys())}")
    print(f"  Channels: {CHANNELS}")
    print(f"  Weeks: {len(WEEKS)} ({WEEKS[0].date()} → {WEEKS[-1].date()})")
    print()

    print("  [1/4] Sales data...")
    sales_df = generate_sales_data()
    print(f"        {len(sales_df):,} rows")

    print("  [2/4] Distribution data...")
    dist_df = generate_distribution_data()
    print(f"        {len(dist_df):,} rows")

    print("  [3/4] Market share + velocity...")
    final_df = compute_share_and_velocity(sales_df, dist_df)

    print("  [4/4] Saving...")
    master_path = OUTPUT_DIR / "nielsen_weekly.csv"
    dist_path   = OUTPUT_DIR / "nielsen_distribution.csv"
    final_df.to_csv(master_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    print(f"        {master_path}  ({len(final_df):,} rows, {len(final_df.columns)} cols)")
    print(f"        {dist_path}  ({len(dist_df):,} rows)")

    # ── Validation ────────────────────────────────────────────────
    print("\n── Validation ──────────────────────────────────────────────")

    # Show latent trends assigned to each market
    print("\nLatent 2-year cumulative trend per market (Luminos):")
    for m, t in sorted(MARKET_TRENDS.items(), key=lambda x: x[1]):
        direction = "DECLINING" if t < 0.88 else ("GROWING" if t > 1.08 else "stable")
        print(f"  {m:<15} {t:.3f}  {direction}")

    # National share
    nat_share = (
        final_df.groupby("brand")[["dollar_sales", "total_dollar_sales"]]
        .sum()
        .assign(national_dollar_share=lambda x: x["dollar_sales"] / x["total_dollar_sales"])
        .sort_values("national_dollar_share", ascending=False)
    )
    print("\nNational Dollar Share:")
    for brand, row in nat_share.iterrows():
        print(f"  {brand:<15} {row['national_dollar_share']:.1%}")

    # Luminos share by market (Year 1 vs Year 2 slope)
    luminos = final_df[final_df["brand"] == "Luminos"].copy()
    luminos["year"] = luminos["week_index"].apply(lambda x: "Y1" if x < 52 else "Y2")
    pivot = (
        luminos.groupby(["market", "year"])["dollar_share"]
        .mean().unstack("year").round(4)
    )
    pivot["change_pp"] = ((pivot["Y2"] - pivot["Y1"]) * 100).round(2)
    pivot["inferred_type"] = pivot["change_pp"].apply(
        lambda c: "DECLINING" if c < -0.5 else ("GROWING" if c > 0.3 else "stable")
    )
    print("\nLuminos Dollar Share by Market (Y1 avg → Y2 avg):")
    print(pivot.sort_values("change_pp").to_string())

    print("\n✓ Data generation complete.")