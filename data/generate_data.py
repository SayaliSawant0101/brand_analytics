"""
generate_data.py  v7 — Luminos DTC Personal Care
Generates three raw event-grain tables:
  - events.csv            (GA4-style, full funnel)
  - orders.csv            (OMS/Shopify-style transactions)
  - ab_assignments.csv    (Optimizely-style, sticky assignment)

v7 vs v6:
  - Brand: Luminos personal care DTC
  - Traffic: paid social heavy (Instagram/TikTok), email CRM
  - Devices: mobile-first (65%)
  - SKUs: 12 Luminos products across 4 tiers ($17.99–$59.99)
  - CVR target: ~3.2% (personal care DTC benchmark)
  - AOV target: ~$48 (bundle-driven)
  - All funnel logic, intent system, behavioral signals unchanged from v6
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
np.random.seed(99)
random.seed(99)

# ── Config ──────────────────────────────────────────────────────────────
N_USERS       = 12_000
PRE_EXP_START = datetime(2023, 4, 1)
PRE_EXP_END   = datetime(2023, 12, 31)
EXP_START     = datetime(2024, 1, 8)
EXP_END       = datetime(2024, 4, 7)
N_EXP_USERS   = 8_000

# Personal care DTC traffic mix — paid social dominant
TRAFFIC_SOURCES = ["instagram",   "email",    "google",  "direct",  "tiktok",     "referral"]
TRAFFIC_WEIGHTS = [0.32,           0.18,       0.20,      0.12,      0.10,         0.08]
MEDIUMS         = ["paid_social",  "email",    "cpc",     "none",    "paid_social", "referral"]

# Mobile-first (social discovery)
DEVICES        = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.65,     0.28,      0.07]

CATEGORIES     = ["Shampoo", "Body Wash", "Conditioner", "Bundle", "Value Pack"]
# Bundle-heavy catalog to drive AOV
CAT_WEIGHTS    = [0.20,       0.18,        0.15,          0.30,      0.17]

PRICE_TIERS    = ["low",   "mid",    "high"]
PRICE_WEIGHTS  = [0.35,    0.45,     0.20]

SESSION_COUNTS  = [1, 2, 3, 4, 5]
SESSION_WEIGHTS = [0.25, 0.32, 0.22, 0.13, 0.08]


# ── Luminos Product Catalog ─────────────────────────────────────────────
# Realistic DTC personal care price points:
#   low  = regular single item 12–16oz ($17.99–$19.99)
#   mid  = large format 32oz or duo bundle ($26.99–$44.99)
#   high = full routine bundle / value pack ($49.99–$59.99)

LUMINOS_PRODUCTS = [
    # Singles — regular size
    {"item_id": "LUM-SH-01",  "item_name": "Hydra Boost Shampoo 12oz",            "category": "Shampoo",    "price_tier": "low",  "price": 21.99},
    {"item_id": "LUM-SH-02",  "item_name": "Scalp Restore Shampoo 12oz",           "category": "Shampoo",    "price_tier": "low",  "price": 23.99},
    {"item_id": "LUM-BW-01",  "item_name": "Moisture Surge Body Wash 16oz",        "category": "Body Wash",  "price_tier": "low",  "price": 21.99},
    {"item_id": "LUM-BW-02",  "item_name": "Gentle Calm Body Wash 16oz",           "category": "Body Wash",  "price_tier": "low",  "price": 22.99},
    {"item_id": "LUM-CD-01",  "item_name": "Repair & Strengthen Conditioner 12oz", "category": "Conditioner","price_tier": "low",  "price": 24.99},
    # Large format / duos — mid tier
    {"item_id": "LUM-SH-03",  "item_name": "Hydra Boost Shampoo 32oz",             "category": "Shampoo",    "price_tier": "mid",  "price": 39.99},
    {"item_id": "LUM-BDL-01", "item_name": "Shampoo + Conditioner Duo",            "category": "Bundle",     "price_tier": "mid",  "price": 42.99},
    {"item_id": "LUM-BDL-02", "item_name": "Shampoo + Body Wash Duo",              "category": "Bundle",     "price_tier": "mid",  "price": 39.99},
    {"item_id": "LUM-BDL-03", "item_name": "Scalp Care Complete Duo",              "category": "Bundle",     "price_tier": "mid",  "price": 44.99},
    # Full routine / value packs — high tier
    {"item_id": "LUM-VP-01",  "item_name": "Full Routine Bundle (Shampoo + Conditioner + Body Wash)", "category": "Value Pack", "price_tier": "high", "price": 59.99},
    {"item_id": "LUM-VP-02",  "item_name": "Family Care Value Pack (4 Full-Size Products)",           "category": "Value Pack", "price_tier": "high", "price": 64.99},
    {"item_id": "LUM-VP-03",  "item_name": "Luminos Starter Kit (6 Travel + 1 Full Size)",            "category": "Value Pack", "price_tier": "high", "price": 54.99},
]

catalog = pd.DataFrame(LUMINOS_PRODUCTS)

# Quantity distributions differ by tier
QUANTITY_DIST = {
    "low" : {"values": [1,2,3], "probs": [0.50, 0.33, 0.17]},
    "mid" : {"values": [1,2,3], "probs": [0.78, 0.18, 0.04]},
    "high": {"values": [1,2],   "probs": [0.92, 0.08]},
}


# ══════════════════════════════════════════════════════════════════════
# INTENT SYSTEM  (unchanged from v6)
# ══════════════════════════════════════════════════════════════════════

def assign_intent(source, price_tier, device):
    base_intent_map = {
        "direct"   : "very_high",
        "email"    : "very_high",
        "google"   : "high",
        "referral" : "high",
        "instagram": "low",
        "tiktok"   : "low",
    }
    intent_order = ["very_low", "low", "medium", "high", "very_high"]
    intent = base_intent_map[source]
    idx    = intent_order.index(intent)

    if price_tier == "high":
        idx = max(0, idx - 1)
    if device == "mobile" and price_tier == "high":
        idx = max(0, idx - 1)

    return intent_order[idx]


def assign_dropoff_reason(intent, session_num=0, has_purchased=False):
    reason_probs = {
        "very_high": {"converted": 0.45, "decision_friction": 0.15,
                      "price_barrier": 0.25, "out_of_stock": 0.10, "distraction": 0.05},
        "high":      {"converted": 0.28, "comparison_intent": 0.28,
                      "decision_friction": 0.25, "price_barrier": 0.14, "distraction": 0.05},
        "medium":    {"converted": 0.15, "comparison_intent": 0.35,
                      "ad_pdp_mismatch": 0.20, "price_barrier": 0.20, "decision_friction": 0.10},
        "low":       {"converted": 0.05, "impulse_faded": 0.30,
                      "price_shock": 0.28, "wrong_audience": 0.20, "distraction": 0.17},
        "very_low":  {"converted": 0.02, "price_shock": 0.38,
                      "impulse_faded": 0.28, "wrong_audience": 0.22, "distraction": 0.10},
    }
    probs = dict(reason_probs[intent])

    if has_purchased:
        probs = {"converted": 0.85, "decision_friction": 0.08,
                 "price_barrier": 0.05, "comparison_intent": 0.02}
    elif session_num > 0:
        probs["converted"]         = min(probs.get("converted", 0) * 1.8, 0.55)
        probs["comparison_intent"] = probs.get("comparison_intent", 0) * 1.2
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}

    return np.random.choice(list(probs.keys()), p=list(probs.values()))


def get_behavioral_signals(dropoff_reason, price_tier):
    signals = {
        "decision_friction": {"pdp_views": np.random.choice([2,3,4], p=[0.50,0.35,0.15]),
                              "session_dur": random.uniform(180,480), "scroll_depth": random.uniform(0.70,1.0), "cross_return": random.random()<0.45},
        "comparison_intent": {"pdp_views": np.random.choice([1,2], p=[0.40,0.60]),
                              "session_dur": random.uniform(60,240), "scroll_depth": random.uniform(0.50,0.85), "cross_return": random.random()<0.55},
        "price_barrier":     {"pdp_views": 1, "session_dur": random.uniform(15,60),
                              "scroll_depth": random.uniform(0.10,0.30), "cross_return": random.random()<0.10},
        "out_of_stock":      {"pdp_views": 1, "session_dur": random.uniform(10,40),
                              "scroll_depth": random.uniform(0.20,0.50), "cross_return": random.random()<0.30},
        "price_shock":       {"pdp_views": 1, "session_dur": random.uniform(10,45),
                              "scroll_depth": random.uniform(0.15,0.35), "cross_return": random.random()<0.05},
        "impulse_faded":     {"pdp_views": 1, "session_dur": random.uniform(5,30),
                              "scroll_depth": random.uniform(0.05,0.20), "cross_return": False},
        "wrong_audience":    {"pdp_views": 1, "session_dur": random.uniform(2,10),
                              "scroll_depth": random.uniform(0.01,0.05), "cross_return": False},
        "ad_pdp_mismatch":   {"pdp_views": 1, "session_dur": random.uniform(3,15),
                              "scroll_depth": random.uniform(0.02,0.10), "cross_return": False},
        "distraction":       {"pdp_views": 1, "session_dur": random.uniform(20,120),
                              "scroll_depth": random.uniform(0.20,0.60), "cross_return": random.random()<0.25},
        "converted":         {"pdp_views": np.random.choice([1,2,3], p=[0.60,0.30,0.10]),
                              "session_dur": random.uniform(90,360), "scroll_depth": random.uniform(0.60,1.0), "cross_return": False},
    }
    return signals.get(dropoff_reason, signals["distraction"])


# ══════════════════════════════════════════════════════════════════════
# FUNNEL PROBABILITIES — calibrated for ~3.2% CVR
# ══════════════════════════════════════════════════════════════════════

def get_home_to_plp_prob(source, device):
    base = {
        "direct"   : 0.82,
        "email"    : 0.85,
        "google"   : 0.68,
        "referral" : 0.65,
        "instagram": 0.38,
        "tiktok"   : 0.35,
    }[source]
    dev_mod = {"mobile": -0.06, "desktop": 0.04, "tablet": 0.00}[device]
    return np.clip(base + dev_mod, 0.10, 0.95)


def get_plp_to_pdp_prob(intent, category, device):
    base = {"very_high": 0.72, "high": 0.55, "medium": 0.40,
            "low": 0.22, "very_low": 0.12}[intent]
    cat_mod = {"Shampoo": 0.05, "Body Wash": 0.04, "Conditioner": 0.03,
               "Bundle": -0.05, "Value Pack": -0.08}[category]
    dev_mod = {"mobile": -0.05, "desktop": 0.03, "tablet": 0.00}[device]
    return np.clip(base + cat_mod + dev_mod, 0.05, 0.92)


def get_pdp_to_atc_prob(intent, dropoff_reason, price_tier,
                         device, is_treatment=False, has_purchased=False):
    # Calibrated down ~40% from v6 to achieve ~3.2% CVR
    base_by_reason = {
        "decision_friction": {"low": 0.17, "mid": 0.11, "high": 0.06},
        "comparison_intent": {"low": 0.14, "mid": 0.09, "high": 0.05},
        "price_barrier":     {"low": 0.04, "mid": 0.02, "high": 0.01},
        "out_of_stock":      {"low": 0.03, "mid": 0.02, "high": 0.01},
        "price_shock":       {"low": 0.02, "mid": 0.01, "high": 0.005},
        "impulse_faded":     {"low": 0.02, "mid": 0.01, "high": 0.005},
        "wrong_audience":    {"low": 0.01, "mid": 0.005,"high": 0.002},
        "ad_pdp_mismatch":   {"low": 0.01, "mid": 0.005,"high": 0.002},
        "distraction":       {"low": 0.03, "mid": 0.02, "high": 0.01},
        "converted":         {"low": 0.90, "mid": 0.82, "high": 0.68},
    }
    atc = base_by_reason.get(dropoff_reason, base_by_reason["distraction"])[price_tier]
    dev_mod = {"mobile": -0.05, "desktop": 0.03, "tablet": 0.00}[device]
    atc = np.clip(atc + dev_mod, 0.001, 0.85)

    if is_treatment:
        lift_by_reason = {
            "decision_friction": 1.28, "comparison_intent": 1.22,
            "price_barrier": 1.08,     "out_of_stock": 1.00,
            "price_shock": 1.03,       "impulse_faded": 1.01,
            "wrong_audience": 1.00,    "ad_pdp_mismatch": 1.01,
            "distraction": 1.05,       "converted": 1.10,
        }
        multiplier = lift_by_reason.get(dropoff_reason, 1.0)
        if price_tier == "high":
            multiplier *= 1.05
        atc = np.clip(atc * multiplier, 0.001, 0.85)

    if has_purchased:
        atc = np.clip(atc * 3.00, 0, 0.93)

    return atc


def get_atc_to_checkout_prob(price_tier, device, is_treatment=False):
    base = {"low": 0.60, "mid": 0.52, "high": 0.44}[price_tier]
    dev  = {"mobile": -0.07, "desktop": 0.04, "tablet": 0.00}[device]
    prob = np.clip(base + dev, 0.10, 0.90)
    if is_treatment:
        prob = np.clip(prob * 1.07, 0, 0.90)
    return prob


def get_checkout_to_purchase_prob(price_tier, source):
    base = {"low": 0.82, "mid": 0.76, "high": 0.68}[price_tier]
    src  = {"google": 0.02, "instagram": -0.04, "tiktok": -0.05,
            "email": 0.06,  "direct": 0.05,     "referral": 0.01}[source]
    return np.clip(base + src, 0.20, 0.95)


# ══════════════════════════════════════════════════════════════════════
# SESSION EVENT GENERATOR
# ══════════════════════════════════════════════════════════════════════

def generate_session_events(user_id, session_date, device, source, medium,
                             product, is_treatment=None,
                             session_num=0, has_purchased=False):
    events = []
    ts = session_date

    def next_ts(lo=5, hi=120):
        nonlocal ts
        ts = ts + timedelta(seconds=random.randint(lo, hi))
        return ts

    intent         = assign_intent(source, product["price_tier"], device)
    dropoff_reason = assign_dropoff_reason(intent, session_num=session_num,
                                            has_purchased=has_purchased)
    signals        = get_behavioral_signals(dropoff_reason, product["price_tier"])

    base = {
        "user_pseudo_id"  : user_id,
        "session_id"      : f"sess_{user_id}_{session_date.strftime('%Y%m%d%H%M%S')}",
        "device_category" : device,
        "traffic_source"  : source,
        "traffic_medium"  : medium,
        "item_id"         : product["item_id"],
        "item_name"       : product["item_name"],
        "product_category": product["category"],
        "price_tier"      : product["price_tier"],
        "item_price"      : product["price"],
        "ab_group"        : is_treatment,
        "intent_level"    : intent,
        "dropoff_reason"  : dropoff_reason,
    }

    # Stage 1: Home
    next_ts(1, 20)
    events.append({**base, "event_id": f"evt_{fake.uuid4()[:8]}",
                   "event_timestamp": ts, "event_name": "page_view",
                   "page_type": "home", "page_path": "/",
                   "category_viewed": None,
                   "scroll_depth": round(random.uniform(0.10, 0.60), 2),
                   "order_id": None, "order_revenue": None})

    if random.random() > get_home_to_plp_prob(source, device):
        return events

    # Stage 2: PLP
    next_ts(10, 60)
    cat_slug = product["category"].lower().replace(" ", "-").replace("&", "and")
    events.append({**base, "event_id": f"evt_{fake.uuid4()[:8]}",
                   "event_timestamp": ts, "event_name": "view_item_list",
                   "page_type": "plp",
                   "page_path": f"/collections/{cat_slug}",
                   "category_viewed": product["category"],
                   "scroll_depth": round(random.uniform(0.20, 0.80), 2),
                   "order_id": None, "order_revenue": None})

    if random.random() > get_plp_to_pdp_prob(intent, product["category"], device):
        return events

    # Stage 3: PDP
    for view_num in range(signals["pdp_views"]):
        next_ts(15, 90)
        scroll = min(round(signals["scroll_depth"] * random.uniform(0.85, 1.0), 2), 1.0)
        events.append({**base, "event_id": f"evt_{fake.uuid4()[:8]}",
                       "event_timestamp": ts, "event_name": "view_item",
                       "page_type": "pdp",
                       "page_path": f"/products/{product['item_id'].lower()}",
                       "category_viewed": product["category"],
                       "scroll_depth": scroll,
                       "order_id": None, "order_revenue": None})
        if view_num < signals["pdp_views"] - 1:
            next_ts(30, 200)

    atc_prob = get_pdp_to_atc_prob(
        intent, dropoff_reason, product["price_tier"],
        device, is_treatment=bool(is_treatment), has_purchased=has_purchased)

    if session_num > 0:
        atc_prob = np.clip(atc_prob * 1.60, 0, 0.88)

    if random.random() > atc_prob:
        return events

    # Stage 4: ATC
    next_ts(10, 45)
    events.append({**base, "event_id": f"evt_{fake.uuid4()[:8]}",
                   "event_timestamp": ts, "event_name": "add_to_cart",
                   "page_type": "pdp",
                   "page_path": f"/products/{product['item_id'].lower()}",
                   "category_viewed": product["category"],
                   "scroll_depth": None, "order_id": None, "order_revenue": None})

    if random.random() > get_atc_to_checkout_prob(
            product["price_tier"], device, bool(is_treatment)):
        return events

    # Stage 5: Checkout
    next_ts(20, 100)
    events.append({**base, "event_id": f"evt_{fake.uuid4()[:8]}",
                   "event_timestamp": ts, "event_name": "begin_checkout",
                   "page_type": "checkout", "page_path": "/checkout",
                   "category_viewed": None, "scroll_depth": None,
                   "order_id": None, "order_revenue": None})

    if random.random() > get_checkout_to_purchase_prob(
            product["price_tier"], source):
        return events

    # Stage 6: Purchase
    next_ts(30, 150)
    order_id  = f"ord_{fake.uuid4()[:6]}"
    qty_cfg   = QUANTITY_DIST[product["price_tier"]]
    quantity  = np.random.choice(qty_cfg["values"], p=qty_cfg["probs"])
    revenue   = round(product["price"] * quantity, 2)

    if is_treatment and product["price_tier"] in ("mid", "high"):
        revenue = round(revenue * random.uniform(1.04, 1.08), 2)

    events.append({**base, "event_id": f"evt_{fake.uuid4()[:8]}",
                   "event_timestamp": ts, "event_name": "purchase",
                   "page_type": "confirmation",
                   "page_path": "/order-confirmation",
                   "category_viewed": None, "scroll_depth": None,
                   "order_id": order_id, "order_revenue": revenue})

    return events


def random_ts(start, end):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta),
                              hours=random.randint(0, 23),
                              minutes=random.randint(0, 59))


# ══════════════════════════════════════════════════════════════════════
# PRE-EXPERIMENT
# ══════════════════════════════════════════════════════════════════════
print("Generating pre-experiment events (Luminos DTC, 9 months)...")
all_events = []

for i in range(N_USERS):
    user_id       = f"uid_{fake.uuid4()[:8]}"
    pref_device   = np.random.choice(DEVICES,         p=DEVICE_WEIGHTS)
    pref_source   = np.random.choice(TRAFFIC_SOURCES, p=TRAFFIC_WEIGHTS)
    n_sessions    = np.random.choice(SESSION_COUNTS,  p=SESSION_WEIGHTS)
    first_ts      = random_ts(PRE_EXP_START, PRE_EXP_END - timedelta(days=60))
    session_ts    = first_ts
    has_purchased = False

    for s in range(n_sessions):
        device = pref_device if random.random() < 0.75 else np.random.choice(DEVICES, p=DEVICE_WEIGHTS)
        source = pref_source if random.random() < 0.65 else np.random.choice(TRAFFIC_SOURCES, p=TRAFFIC_WEIGHTS)
        medium = MEDIUMS[TRAFFIC_SOURCES.index(source)]

        # Sample from catalog using category weights
        cat    = np.random.choice(CATEGORIES, p=CAT_WEIGHTS)
        cat_products = catalog[catalog["category"] == cat]
        product = cat_products.sample(1).iloc[0]

        if s > 0:
            session_ts = session_ts + timedelta(days=random.randint(7, 45),
                                                 hours=random.randint(0, 23))
        if session_ts > PRE_EXP_END:
            break

        evts = generate_session_events(user_id, session_ts, device, source, medium,
                                        product, is_treatment=None,
                                        session_num=s, has_purchased=has_purchased)
        all_events.extend(evts)
        if any(e["event_name"] == "purchase" for e in evts):
            has_purchased = True

print(f"  Pre-experiment: {len(all_events):,} events")


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ══════════════════════════════════════════════════════════════════════
print("Generating experiment events (Luminos PDP value messaging test)...")
ab_assignments = []
exp_events     = []

for i in range(N_EXP_USERS):
    user_id      = f"uid_{fake.uuid4()[:8]}"
    is_treatment = i >= N_EXP_USERS // 2
    group        = "treatment" if is_treatment else "control"
    pref_device  = np.random.choice(DEVICES,         p=DEVICE_WEIGHTS)
    pref_source  = np.random.choice(TRAFFIC_SOURCES, p=TRAFFIC_WEIGHTS)
    n_sessions   = np.random.choice(SESSION_COUNTS,  p=SESSION_WEIGHTS)
    first_ts     = random_ts(EXP_START, EXP_END - timedelta(days=14))

    ab_assignments.append({
        "assignment_id"    : f"asgn_{fake.uuid4()[:8]}",
        "user_pseudo_id"   : user_id,
        "experiment_id"    : "exp_luminos_pdp_v1",
        "experiment_name"  : "luminos_pdp_value_messaging_q1_2024",
        "variant"          : group,
        "assigned_at"      : EXP_START,
        "first_exposure_at": first_ts,
    })

    session_ts    = first_ts
    has_purchased = False

    for s in range(n_sessions):
        device = pref_device if random.random() < 0.75 else np.random.choice(DEVICES, p=DEVICE_WEIGHTS)
        source = pref_source if random.random() < 0.65 else np.random.choice(TRAFFIC_SOURCES, p=TRAFFIC_WEIGHTS)
        medium = MEDIUMS[TRAFFIC_SOURCES.index(source)]

        cat      = np.random.choice(CATEGORIES, p=CAT_WEIGHTS)
        cat_products = catalog[catalog["category"] == cat]
        product  = cat_products.sample(1).iloc[0]

        if s > 0:
            session_ts = session_ts + timedelta(days=random.randint(7, 30),
                                                 hours=random.randint(0, 23))
        if session_ts > EXP_END:
            break

        evts = generate_session_events(user_id, session_ts, device, source, medium,
                                        product, is_treatment=is_treatment,
                                        session_num=s, has_purchased=has_purchased)
        exp_events.extend(evts)
        if any(e["event_name"] == "purchase" for e in evts):
            has_purchased = True

all_events.extend(exp_events)
print(f"  Experiment: {len(exp_events):,} events")


# ══════════════════════════════════════════════════════════════════════
# ORDERS TABLE
# ══════════════════════════════════════════════════════════════════════
print("Building orders table...")
purchase_events = [e for e in all_events if e["event_name"] == "purchase"]
orders = []
for e in purchase_events:
    qty_cfg  = QUANTITY_DIST[e["price_tier"]]
    quantity = np.random.choice(qty_cfg["values"], p=qty_cfg["probs"])
    base_ret = {"low": 0.04, "mid": 0.07, "high": 0.10}[e["price_tier"]]
    if e["ab_group"]:
        base_ret = round(base_ret * 0.88, 3)
    orders.append({
        "order_id"        : e["order_id"],
        "user_pseudo_id"  : e["user_pseudo_id"],
        "session_id"      : e["session_id"],
        "order_created_at": e["event_timestamp"],
        "item_id"         : e["item_id"],
        "item_name"       : e["item_name"],
        "product_category": e["product_category"],
        "price_tier"      : e["price_tier"],
        "quantity"        : quantity,
        "item_price"      : e["item_price"],
        "order_revenue"   : e["order_revenue"],
        "payment_method"  : np.random.choice(
            ["credit_card", "paypal", "apple_pay", "shop_pay"],
            p=[0.42, 0.18, 0.24, 0.16]),
        "is_returned"     : random.random() < base_ret,
        "intent_level"    : e["intent_level"],
        "dropoff_reason"  : e["dropoff_reason"],
    })


# ══════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════
print("Saving CSVs...")
df_events = pd.DataFrame(all_events).drop(columns=["ab_group"])
df_orders = pd.DataFrame(orders)
df_ab     = pd.DataFrame(ab_assignments)

os.makedirs("data", exist_ok=True)
df_events.to_csv("data/events.csv",         index=False)
df_orders.to_csv("data/orders.csv",         index=False)
df_ab.to_csv    ("data/ab_assignments.csv", index=False)


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("DATA GENERATION COMPLETE  (v7 — Luminos DTC Personal Care)")
print("=" * 62)
print(f"  events.csv             {len(df_events):>8,} rows")
print(f"  orders.csv             {len(df_orders):>8,} rows")
print(f"  ab_assignments.csv     {len(df_ab):>8,} rows")

purchases  = df_events[df_events['event_name'] == 'purchase']
sessions   = df_events.drop_duplicates('session_id')
cvr        = len(purchases['session_id'].unique()) / len(sessions)
aov        = df_orders['order_revenue'].mean()

print(f"\n  Unique users           {df_events['user_pseudo_id'].nunique():>8,}")
print(f"  Unique sessions        {df_events['session_id'].nunique():>8,}")
print(f"  Overall CVR            {cvr:>8.2%}  (target ~3.2%)")
print(f"  AOV                    ${aov:>7.2f}  (target ~$48)")

user_order_counts = df_orders.groupby('user_pseudo_id')['order_id'].count()
repeat_purchasers = (user_order_counts > 1).sum()
total_purchasers  = len(user_order_counts)
print(f"\n  Total purchasers       {total_purchasers:>8,}")
print(f"  Repeat purchasers      {repeat_purchasers:>8,}  ({repeat_purchasers/total_purchasers:.1%})")

print(f"\n  Date range   {df_events['event_timestamp'].min()} →")
print(f"               {df_events['event_timestamp'].max()}")

print("\nFunnel breakdown:")
for stage in ['page_view','view_item_list','view_item','add_to_cart','begin_checkout','purchase']:
    n = df_events[df_events['event_name']==stage]['session_id'].nunique()
    print(f"  {stage:<20} {n:>6,}  ({n/len(sessions):.1%})")

print("\nTraffic source mix:")
print(sessions['traffic_source'].value_counts(normalize=True).round(3).to_string())

print("\nDevice mix:")
print(sessions['device_category'].value_counts(normalize=True).round(3).to_string())

print("\nProduct category mix (orders):")
print(df_orders['product_category'].value_counts(normalize=True).round(3).to_string())

print(f"\nAvg item price by tier:")
print(df_orders.groupby('price_tier')['item_price'].mean().round(2).to_string())

print("\nFiles saved to data/")