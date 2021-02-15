from openfisca_uk import PopulationSim, IndividualSim
from openfisca_uk.reforms.modelling import reported_benefits
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

REFORM_NAMES = ["Foundational", "Disability", "Disability + geo"]

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
]

REFORM_PERSON_COLS = ["basic_income", "household_net_income"]

REFORM_HH_COLS = ["household_net_income", "poverty_gap_bhc", "poverty_gap_ahc"]

p_base = baseline_sim.df(BASELINE_PERSON_COLS, map_to="person")
p_base.rename(
    {
        "household_net_income": "household_net_income_base",
        "equiv_household_net_income": "equiv_household_net_income_base",
    },
    axis=1,
    inplace=True,
)

hh_base = baseline_sim.df(BASELINE_HH_COLS, map_to="household")
hh_base.rename(
    {
        "household_net_income": "household_net_income_base",
        "poverty_gap_bhc": "poverty_gap_bhc_base",
        "poverty_gap_ahc": "poverty_gap_ahc_base",
    },
    axis=1,
    inplace=True,
)


def reform_p(i):
    p = reform_sims[i].df(REFORM_PERSON_COLS, map_to="person")
    p["reform"] = REFORM_NAMES[i]
    return pd.concat([p, p_base], axis=1)


def reform_hh(i):
    hh = reform_sims[i].df(REFORM_HH_COLS, map_to="household")
    hh["reform"] = REFORM_NAMES[i]
    return pd.concat([hh, hh_base], axis=1)


def get_dfs():
    p_all = pd.concat([reform_p(i) for i in range(3)])
    p_all["region_name"] = p_all.region.map(region_map)

    hh_all = pd.concat([reform_hh(i) for i in range(3)])
    return p_all, hh_all
