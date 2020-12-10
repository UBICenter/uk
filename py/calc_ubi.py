"""
Example usage:

baseline_df, base_reform_df, budget = get_data()
ubi_df = set_ubi(base_reform_df, budget, 0, 0, 0, 0, 0, np.zeros((12)), verbose=True)

"""

from openfisca_uk.tools.simulation import PopulationSim
import frs
import pandas as pd
import numpy as np
from rdbl import gbp
from openfisca_uk.tools.general import add
from openfisca_core.model_api import *
from openfisca_uk.entities import *


def calc2df(
    sim: PopulationSim, cols: list, map_to: str = "person"
) -> pd.DataFrame:
    """Make a DataFrame from an openfisca-uk PopulationSim.

    :param sim: PopulationSim object to extract from.
    :type sim: PopulationSim
    :param cols: List of simulation attributes.
    :type cols: list
    :param map_to: Entity type to return: 'person', 'benunit', or 'household'.
        Defaults to 'person'.
    :type map_to: str, optional
    :return: DataFrame with each attribute of sim as a column.
    :rtype: pd.DataFrame
    """
    d = {}
    for i in cols:
        d[i] = sim.calc(i, map_to=map_to)
    return pd.DataFrame(d)


BASELINE_COLS = [
    "household_id",
    "is_SP_age",
    "is_child",
    "is_WA_adult",
    "is_disabled",
    "is_enhanced_disabled",
    "is_severely_disabled",
    "region",
    "household_weight",
    "household_net_income",
    "people_in_household"
]

CORE_BENEFITS = [
    "child_benefit",
    "income_support",
    "JSA_contrib",
    "JSA_income",
    "child_tax_credit",
    "working_tax_credit",
    "universal_credit",
    "state_pension",
    "pension_credit",
    "ESA_income",
    "ESA_contrib",
    "housing_benefit",
]


def ubi_reform(senior, adult, child, dis_1, dis_2, dis_3, geo):
    """Create an OpenFisca-UK reform class.

    Args:
        senior (float): Pensioner UBI amount per week
        adult (float): Adult UBI amount per week
        child (float): Child UBI amount per week
        dis_1 (float): Disabled (Equality Act+) supplement per week
        dis_2 (float): Enhanced disabled supplement per week
        dis_3 (float): Severely disabled supplement per week
        geo (ndarray): Numpy float array of 12 UK regional supplements per week

    Returns:
        DataFrame: Person-level DataFrame with columns mapped and yearlyised
    """

    class income_tax(Variable):
        value_type = float
        entity = Person
        label = "Income tax paid per year"
        definition_period = YEAR

        def formula(person, period, parameters):
            return 0.5 * person("taxable_income", period)

    class basic_income(Variable):
        value_type = float
        entity = Person
        label = "Amount of basic income received per week"
        definition_period = WEEK

        def formula(person, period, parameters):
            region = person.household("region", period)
            return (
                senior * person("is_SP_age", period.this_year)
                + adult * person("is_WA_adult", period.this_year)
                + child * person("is_child", period.this_year)
                + dis_1 * person("is_disabled", period.this_year)
                + dis_2 * person("is_enhanced_disabled", period.this_year)
                + dis_3 * person("is_severely_disabled", period.this_year)
                + geo[person.household("region").astype(int)]
            )

    class gross_income(Variable):
        value_type = float
        entity = Person
        label = "Gross income"
        definition_period = YEAR

        def formula(person, period, parameters):
            COMPONENTS = [
                "basic_income",
                "earnings",
                "profit",
                "state_pension",
                "pension_income",
                "savings_interest",
                "rental_income",
                "SSP",
                "SPP",
                "SMP",
                "holiday_pay",
                "dividend_income",
                "total_benefits",
                "benefits_modelling",
            ]
            return add(person, period, COMPONENTS, options=[MATCH])

    class reform(Reform):
        def apply(self):
            for changed_var in [income_tax, gross_income]:
                self.update_variable(changed_var)
            for added_var in [basic_income]:
                self.add_variable(added_var)
            for removed_var in CORE_BENEFITS + ["NI"]:
                self.neutralize_variable(removed_var)

    return reform


REGIONS = np.array(
    [
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
)


def get_data(path=None):
    """Generate key datasets for UBI reforms.

    Returns:
        DataFrame: Baseline DataFrame with core variables.
        DataFrame: UBI tax reform DataFrame with core variables.
        float: Yearly revenue raised by the UBI tax reform.
    """
    if path is not None:
        person = pd.read_csv(path + "/person.csv")
        benunit = pd.read_csv(path + "/benunit.csv")
        household = pd.read_csv(path + "/household.csv")
    else:
        person, benunit, household = frs.load()
    baseline = PopulationSim(frs_data=(person, benunit, household))
    baseline_df = calc2df(baseline, BASELINE_COLS)
    FRS_DATA = (person, benunit, household)
    reform_no_ubi = ubi_reform(0, 0, 0, 0, 0, 0, np.array([0] * 12))
    reform_no_ubi_sim = PopulationSim(reform_no_ubi, frs_data=FRS_DATA)
    reform_base_df = calc2df(reform_no_ubi_sim, BASELINE_COLS)
    budget = -np.sum(
        baseline.calc("household_weight")
        * (
            reform_no_ubi_sim.calc("household_net_income")
            - baseline.calc("household_net_income")
        )
    )
    return baseline_df, reform_base_df, budget


def set_ubi(
    base_df, budget, senior, child, dis_1, dis_2, dis_3, regions, verbose=False
):
    """Calculate budget-neutral UBI amounts per person.

    Args:
        base_df (DataFrame): UBI tax reform household-level DataFrame
        budget (float): Total budget for UBI spending
        senior (float): Pensioner UBI amount per week
        child (float): Child UBI amount per week
        dis_1 (float): Disabled (Equality Act+) supplement per week
        dis_2 (float): Enhanced disabled supplement per week
        dis_3 (float): Severely disabled supplement per week
        regions (ndarray): Numpy float array of 12 UK regional supplements per week
        verbose (bool, optional): Whether to print the calibrated adult UBI amount. Defaults to False.

    Returns:
        DataFrame: Reform household-level DataFrame
    """
    basic_income = (
        base_df["is_SP_age"] * senior
        + base_df["is_child"] * child
        + base_df["is_disabled"] * dis_1
        + base_df["is_enhanced_disabled"] * dis_2
        + base_df["is_severely_disabled"] * dis_3
    ) * 52
    for i, region_name in zip(range(len(regions)), REGIONS):
        basic_income += (
            np.where(REGIONS[base_df["region"]] == region_name, regions[i], 0)
            * 52
        )
    total_cost = np.sum(basic_income * base_df["household_weight"])
    adult_amount = (budget - total_cost) / np.sum(
        base_df["is_WA_adult"] * base_df["household_weight"]
    )
    if verbose:
        print(f"Adult amount: {gbp(adult_amount / 52)}/week")
    basic_income += base_df["is_WA_adult"] * adult_amount
    reform_df = base_df
    reform_df["basic_income"] = basic_income
    reform_df["household_net_income"] += basic_income
    return reform_df
