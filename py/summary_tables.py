import numpy as np
import pandas as pd
import os

# Scripts must be run from root directory.
if "py" not in os.listdir("."):
    os.chdir("..")

from make_dfs import get_dfs


p, h = get_dfs()


def pct_chg(base, new):
    return (new - base) / base


def group(data, group_cols, agg_cols=[], compare_cols=None, mean=False):
    """ Workaround for current microdf groupby issue.
    """
    # Replicate agg cols for base and reform.
    all_compare_cols = compare_cols + [i + "_base" for i in compare_cols]
    res_g = data[group_cols + agg_cols + all_compare_cols].groupby(group_cols)
    if mean:
        res = res_g.mean()
    else:
        res = res_g.sum()
    for i in compare_cols:
        res[i + "_pct_chg"] = pct_chg(res[i + "_base"], res[i])
    return res


# Reform level summary.
INEQS = ["gini", "top_10_pct_share", "top_1_pct_share"]
POVS = ["in_poverty_bhc", "in_deep_poverty_bhc"]
ineq_base = h.groupby("reform").equiv_household_net_income_base.agg(INEQS)
ineq_base.columns = [i + "_base" for i in ineq_base.columns]
ineq_reform = h.groupby("reform").equiv_household_net_income.agg(INEQS)
ineq_reform.columns = [i + "_reform" for i in ineq_reform.columns]
budget_impact = group(h, ["reform"], compare_cols=["household_net_income"])
budget_impact["household_net_income_chg"] = (
    budget_impact.household_net_income
    - budget_impact.household_net_income_base
)
pov = group(p, ["reform"], compare_cols=POVS, mean=True)
p_agg = p.groupby("reform")[["household_net_income_pl", "winner"]].mean()
r = (
    p_agg.join(ineq_base)
    .join(ineq_reform, on="reform")
    .join(budget_impact)
    .join(pov)
)
r["reform"] = r.index  # Easier for plotting.
for i in INEQS:
    r[i + "_pc"] = pct_chg(r[i + "_base"], r[i + "_reform"])

# Per reform per decile (by household).
decile = (
    group(h, ["reform", "decile"], ["people"], ["household_net_income"],)
).reset_index()
decile["chg"] = decile.household_net_income - decile.household_net_income_base
decile["weekly_chg_pp"] = (decile.chg / decile.people) / 52


G = ["reform", "region"]

region_pov = group(p, G, compare_cols=["in_poverty_bhc"])
region_inc = group(h, G, compare_cols=["household_net_income"])
region_r = region_pov.join(region_inc).reset_index()

# Poverty by age group.

p["age_group"] = np.where(
    p.age < 18, "0 to 17", np.where(p.age < 65, "18 to 64", "65 and older")
)
POV_COLS = [
    "in_poverty_bhc",
    "in_poverty_bhc_base",
    "in_deep_poverty_bhc",
    "in_deep_poverty_bhc_base",
]
GROUPS = ["reform", "age_group"]
# Use group function.
pov_age = (p[GROUPS + POV_COLS].groupby(GROUPS).mean() / 52).reset_index()

cur_pov = pov_age[pov_age.reform == "1: Foundational"][
    ["age_group", "in_poverty_bhc_base", "in_deep_poverty_bhc_base"]
]
cur_pov_long = cur_pov.melt(id_vars="age_group")

pov_age["pov_chg"] = pov_age.in_poverty_bhc / pov_age.in_poverty_bhc_base - 1
pov_age["deep_pov_chg"] = (
    pov_age.in_deep_poverty_bhc / pov_age.in_deep_poverty_bhc_base - 1
)
pov_chg_long = pov_age.melt(
    ["reform", "age_group"], ["pov_chg", "deep_pov_chg"]
)

# Within-decile.
BUCKETS = [
    "Lose more than 5%",
    "Lose less than 5%",
    "Gain less than 5%",
    "Gain more than 5%",
]
h["household_net_income_pc_group"] = pd.cut(
    h.household_net_income_pc,
    [-np.inf, -0.05, 0, 0.05, np.inf],
    labels=BUCKETS,
)


def group(groupby, name="people"):
    return (
        h[h.household_net_income_base > 0]
        .groupby(groupby)
        .people.sum()
        .reset_index()
        .rename({0: name}, axis=1)
    )


chg_bucket = group(["reform", "decile", "household_net_income_pc_group"])
chg_bucket_decile_total = group(["reform", "decile"], "total_people")
chg_bucket_total = group(["reform", "household_net_income_pc_group"])
reform_total = group("reform")
# Calculate share of decile.
chg_bucket = chg_bucket.merge(chg_bucket_decile_total, on=["reform", "decile"])
chg_bucket["share_of_decile"] = chg_bucket.people / chg_bucket.total_people

# Sort for correct stack order.
chg_bucket["order"] = chg_bucket.household_net_income_pc_group.map(
    dict(zip(BUCKETS, range(len(BUCKETS))))
)
chg_bucket.sort_values("order", ascending=False, inplace=True)

# Export.
def csv(df, f):
    df.to_csv("data/" + f + ".csv", index=False)


csv(r, "reform")
csv(region_r, "region_reform")
csv(decile, "decile")
csv(cur_pov_long, "cur_pov_long")
csv(pov_chg_long, "pov_chg_long")
csv(chg_bucket, "within_decile")
