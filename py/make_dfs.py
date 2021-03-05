import microdf as mdf
import numpy as np
import openfisca_uk as o
import pandas as pd
from openfisca_uk import IndividualSim, PopulationSim
from openfisca_uk.reforms.modelling import reported_benefits
from calc_ubi import ubi_reform

REGION_CODES = [
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

REGION_NAMES = [
    "North East",
    "North West",
    "Yorkshire and the Humber",
    "East Midlands",
    "West Midlands",
    "East of England",
    "London",
    "South East",
    "South West",
    "Wales",
    "Scotland",
    "Northern Ireland",
]

region_code_map = dict(zip(range(len(REGION_CODES)), REGION_CODES))
region_name_map = dict(zip(range(len(REGION_NAMES)), REGION_NAMES))

optimal_params = pd.read_csv("../optimal_params.csv")  # Up a folder.


def reform(i):
    row = optimal_params.iloc[i].round()
    return ubi_reform(
        adult=row.adult,
        child=row.child,
        senior=row.senior,
        dis_base=row.dis_base,
        geo=row[REGION_CODES],
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
]

# Extract these for baseline too.
REFORM_PERSON_COLS = [
    "household_net_income",
    "in_poverty_bhc",
    "in_deep_poverty_bhc",
]

BASELINE_HH_COLS = ["household_weight", "people_in_household", "region"]

# Extract these for baseline too.
REFORM_HH_COLS = [
    "household_net_income",
    "equiv_household_net_income",
    "poverty_gap_bhc",
    "poverty_gap_ahc",
]

p_base = mdf.MicroDataFrame(
    baseline_sim.df(
        BASELINE_PERSON_COLS + REFORM_PERSON_COLS, map_to="person"
    ),
    weights="household_weight",
)
p_base.rename(
    dict(zip(REFORM_PERSON_COLS, [i + "_base" for i in REFORM_PERSON_COLS])),
    axis=1,
    inplace=True,
)

hh_base = mdf.MicroDataFrame(
    baseline_sim.df(BASELINE_HH_COLS + REFORM_HH_COLS, map_to="household"),
    weights="household_weight",
)
hh_base.rename(
    dict(zip(REFORM_HH_COLS, [i + "_base" for i in REFORM_HH_COLS])),
    axis=1,
    inplace=True,
)
hh_base["person_weight"] = (
    hh_base.household_weight * hh_base.people_in_household
)

# Add region code and names
hh_base["region_code"] = hh_base.region.map(region_code_map)
hh_base["region_name"] = hh_base.region.map(region_name_map)
p_base["region_code"] = p_base.region.map(region_code_map)
p_base["region_name"] = p_base.region.map(region_name_map)

# Change weight column to represent people for decile groups.
hh_base.set_weights(hh_base.person_weight)
hh_base["decile"] = np.ceil(
    hh_base.equiv_household_net_income_base.rank(pct=True) * 10
)

# Change weight back to household weight for correct calculation of totals.
hh_base.set_weights(hh_base.household_weight)


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


def get_dfs():
    p = mdf.concat([reform_p(i) for i in range(3)])
    h = mdf.concat([reform_hh(i) for i in range(3)])
    # Error with duplicate indexes. See microdf#179.
    p = p.reset_index(drop=True)
    h = h.reset_index(drop=True)

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
    return p, h
