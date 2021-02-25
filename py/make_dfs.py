from openfisca_uk import PopulationSim, IndividualSim
from openfisca_uk.reforms.modelling import reported_benefits
import numpy as np
import pandas as pd
import microdf as mdf
from py.calc_ubi import ubi_reform

REGIONS = [
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
    "NORTHERN_IRELAND",
]

region_map = dict(zip(range(len(REGIONS)), REGIONS))

optimal_params = pd.read_csv("optimal_params.csv")  # Up a folder.


def reform(i):
    row = optimal_params.iloc[i].round()
    return ubi_reform(
        adult=row.adult,
        child=row.child,
        senior=row.senior,
        dis_base=row.dis_base,
        geo=row[REGIONS],
    )


reforms = [reform(i) for i in range(3)]

baseline_sim = PopulationSim(reported_benefits)
reform_sims = [PopulationSim(reported_benefits, reform) for reform in reforms]

REFORM_NAMES = ["1: Foundational", "2: Disability", "3: Disability + geo"]

BASELINE_PERSON_COLS = [
    "household_weight",
    "age",
    "region",
    "is_disabled_for_ubi",
    "household_net_income",
    "equiv_household_net_income",
    "poverty_line_bhc",
    "poverty_line_ahc",
]

BASELINE_HH_COLS = [
    "household_weight",
    "poverty_gap_bhc",
    "poverty_gap_ahc",
    "household_net_income",
    "equiv_household_net_income",
    "people_in_household",
]

REFORM_PERSON_COLS = ["basic_income", "household_net_income"]

REFORM_HH_COLS = [
    "household_net_income",
    "equiv_household_net_income",
    "poverty_gap_bhc",
    "poverty_gap_ahc",
]

p_base = mdf.MicroDataFrame(baseline_sim.df(BASELINE_PERSON_COLS, map_to="person"), weights="household_weight")
p_base.rename(
    {
        "household_net_income": "household_net_income_base",
        "equiv_household_net_income": "equiv_household_net_income_base",
    },
    axis=1,
    inplace=True,
)

hh_base = mdf.MicroDataFrame(baseline_sim.df(BASELINE_HH_COLS, map_to="household"), weights="household_weight")
# TODO: Do this without hard-coding.
# TODO: Add persons to calculate person weight.
hh_base.rename(
    {
        "household_net_income": "household_net_income_base",
        "equiv_household_net_income": "equiv_household_net_income_base",
        "poverty_gap_bhc": "poverty_gap_bhc_base",
        "poverty_gap_ahc": "poverty_gap_ahc_base",
    },
    axis=1,
    inplace=True,
)
hh_base["person_weight"] = (
    hh_base.household_weight * hh_base.people_in_household
)
# mdf.add_weighted_quantiles(
#     # Cast to pd.DataFrame because it might not currently work with mdf.
#     pd.DataFrame(hh_base), "equiv_household_net_income_base", "person_weight"
# )


def reform_p(i):
    p = reform_sims[i].df(REFORM_PERSON_COLS, map_to="person")
    p["reform"] = REFORM_NAMES[i]
    return mdf.concat([p_base, p], axis=1)


def reform_hh(i):
    hh = reform_sims[i].df(REFORM_HH_COLS, map_to="household")
    hh["reform"] = REFORM_NAMES[i]
    return mdf.concat([hh_base, hh], axis=1)


def pct_chg(base, new):
    return (new - base) / base


def reform_stats(df):
    # For applying over a groupby(reform) or a .
    gini = df.equiv_household_net_income_base.aggregate(
        ["gini", "top_10pct_share"]
    )
    p_agg = df[["household_net_income_pl", "winner"]].mean()


def get_dfs():
    p = mdf.concat([reform_p(i) for i in range(3)])
    h = mdf.concat([reform_hh(i) for i in range(3)])

    # Process.
    p["region_name"] = p.region.map(region_map)

    def chg(df, col):
        df[col + "_chg"] = df[col] - df[col + "_base"]
        # Percentage change, only defined for positive baselines.
        df[col + "_pc"] = np.where(
            df[col + "_base"] > 0,
            df[col + "_chg"] / df[col + "_base"],
            np.nan,
        )
        # Percentage loss. NB: np.minimum(np.nan, 0) -> np.nan.
        df[col + "_pl"] = np.minimum(0, df[col + "_pc"])

    chg(p, "household_net_income")
    chg(h, "household_net_income")
    p["winner"] = p.household_net_income_chg > 0
    h["winner"] = h.household_net_income_chg > 0
    # Per-reform.
    INEQS = ["gini", "top_10_pct_share", "top_1_pct_share"]
    ineq_base = h.groupby("reform").equiv_household_net_income_base.agg(INEQS)
    ineq_base.columns = [i + "_base" for i in ineq_base.columns]
    ineq_reform = h.groupby("reform").equiv_household_net_income.agg(INEQS)
    ineq_reform.columns = [i + "_reform" for i in ineq_reform.columns]
    p_agg = p.groupby("reform")[["household_net_income_pl", "winner"]].mean()
    r = p_agg.join(ineq_base).join(ineq_reform, on="reform")
    r["reform"] = reform.index  # Easier for plotting.
    for i in INEQS:
        r[i + "_pc"] = pct_chg(r[i + "_base"], r[i + "_reform"])
    
    # Per reform per decile (by household).

    # decile = (
    #     h.groupby(["reform", "equiv_household_net_income_base_decile"])[
    #         [
    #             "household_net_income_base",
    #             "household_net_income",
    #             "household_weight",
    #         ]
    #     ]
    #     .sum()
    #     .reset_index()
    # )
    # decile["chg"] = (
    #     decile.household_net_income - decile.household_net_income_base
    # )
    # decile["chg_per_hh"] = decile.chg / decile.household_weight
    # decile["pc"] = decile.chg / decile.household_net_income_base
    return p, h, r#, decile
