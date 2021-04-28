"""
Example usage:

baseline_df, base_reform_df, budget = get_data()
ubi_df = set_ubi(base_reform_df, budget, 0, 0, 0, np.zeros((12)),
                 verbose=True)
"""

from openfisca_uk import Microsimulation
import frs
import pandas as pd
import numpy as np
from rdbl import gbp
from openfisca_uk.tools.general import add
from openfisca_core.model_api import *
from openfisca_uk.entities import *


BASELINE_COLS = [
    "household_id",
    "age_over_64",
    "age_under_18",
    "age_18_64",
    "is_disabled_for_ubi",
    "region",
    "net_income",
    "people",
    "household_equivalisation_bhc",
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
    "PIP_DL",
    "PIP_M",
    "carers_allowance",
    "incapacity_benefit",
    "SDA",
    "AA",
    "DLA_M",
    "DLA_SC",
]


class is_disabled_for_ubi(Variable):
    value_type = float
    entity = Person
    label = (
        "Whether person is classified as disabled for the purposes of UBI supplements"
    )
    definition_period = YEAR

    def formula(person, period, parameters):
        QUALIFYING_BENEFITS = [
            "incapacity_benefit_reported",
            "SDA_reported",
            "AA_reported",
            "DLA_M_reported",
            "DLA_SC_reported",
            "IIDB_reported",
            # Given to a single person at benunit level
            "PIP_DL_reported",
            "PIP_M_reported",
        ]
        return add(person, period, QUALIFYING_BENEFITS, options=[MATCH]) > 0


def ubi_reform(
    senior: float, adult: float, child: float, dis_base: float, geo: np.array,
):
    """Create an OpenFisca-UK reform class.

    Args:
        senior (float): Pensioner UBI amount per week
        adult (float): Adult UBI amount per week
        child (float): Child UBI amount per week
        dis_base (float): Supplement per week for people claiming any
            disability benefit.
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
            return 0.5 * person("total_income", period)

    class basic_income(Variable):
        value_type = float
        entity = Person
        label = "Amount of basic income received per week"
        definition_period = WEEK

        def formula(person, period, parameters):
            def ubi_piece(value, flag):
                return value * person(flag, period.this_year)

            region = person.household("region", period).decode_to_str()
            return (
                ubi_piece(senior, "age_over_64")
                + ubi_piece(adult, "age_18_64")
                + ubi_piece(child, "age_under_18")
                + ubi_piece(dis_base, "is_disabled_for_ubi")
                + geo[region]
            )

    class gross_income(Variable):
        value_type = float
        entity = Person
        label = "Gross income, including benefits"
        definition_period = YEAR

        def formula(person, period, parameters):
            COMPONENTS = [
                "employment_income",
                "pension_income",
                "self_employment_income",
                "property_income",
                "savings_interest_income",
                "dividend_income",
                "miscellaneous_income",
                "benefits",
                "basic_income",
            ]
            return add(person, period, COMPONENTS, options=[MATCH])

    class reform(Reform):
        def apply(self):
            for changed_var in [income_tax, gross_income]:
                self.update_variable(changed_var)
            for added_var in [basic_income, is_disabled_for_ubi]:
                self.add_variable(added_var)
            for removed_var in CORE_BENEFITS + ["national_insurance"]:
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


def get_data():
    """Generate key datasets for UBI reforms.

    Returns:
        DataFrame: Baseline DataFrame with core variables.
        DataFrame: UBI tax reform DataFrame with core variables.
        float: Yearly revenue raised by the UBI tax reform.
    """
    baseline = Microsimulation()
    baseline_df = baseline.df([var for var in BASELINE_COLS if var != "is_disabled_for_ubi"], map_to="household")
    reform_no_ubi = ubi_reform(0, 0, 0, 0, pd.Series([0] * 12, index=REGIONS))
    reform_no_ubi_sim = Microsimulation(reform_no_ubi)
    reform_base_df = reform_no_ubi_sim.df(BASELINE_COLS, map_to="household")
    budget = (
        baseline.calc("net_income", map_to="household").sum()
        - reform_no_ubi_sim.calc("net_income", map_to="household").sum()
    )
    baseline_df_pd = pd.DataFrame(baseline_df)
    baseline_df_pd["household_weight"] = baseline_df.weights
    reform_base_df_pd = pd.DataFrame(reform_base_df)
    reform_base_df_pd["household_weight"] = reform_base_df.weights
    return baseline_df_pd, reform_base_df_pd, budget


def get_adult_amount(
    base_df: pd.DataFrame,
    budget: float,
    senior: float,
    child: float,
    dis_base: float,
    regions: np.array,
    verbose: bool = False,
    individual: bool = False,
    pass_income: bool = False,
) -> pd.DataFrame:
    """Calculate budget-neutral UBI amounts per person.

    Args:
        base_df (DataFrame): UBI tax reform household-level DataFrame.
        budget (float): Total budget for UBI spending.
        senior (float): Pensioner UBI amount per week.
        child (float): Child UBI amount per week.
        dis_base (float): Supplement per week for people claiming any
            disability benefit.
        regions (ndarray): Numpy float array of 12 UK regional supplements per
            week.
        verbose (bool, optional): Whether to print the calibrated adult UBI
            amount. Defaults to False.

    Returns:
        DataFrame: Reform household-level DataFrame.
    """
    basic_income = (
        base_df["age_over_64"] * senior
        + base_df["age_under_18"] * child
        + base_df["is_disabled_for_ubi"] * dis_base
    ) * 52
    for i, region_name in zip(range(len(regions)), REGIONS):
        basic_income += (
            np.where(base_df.region == region_name, regions[i], 0)
            * base_df.people
            * 52
        )
    total_cost = basic_income.sum()
    adult_amount = (budget - total_cost) / base_df.age_18_64.sum()
    if verbose:
        print(f"Adult amount: {gbp(adult_amount / 52)}/week")
    if pass_income:
        return basic_income, adult_amount
    if individual:
        return adult_amount / 52
    else:
        return adult_amount


def set_ubi(
    base_df: pd.DataFrame,
    budget: float,
    senior: float,
    child: float,
    dis_base: float,
    regions: np.array,
    verbose: bool = False,
):
    """Calculate budget-neutral UBI amounts per person.

    Args:
        base_df (DataFrame): UBI tax reform household-level DataFrame.
        budget (float): Total budget for UBI spending.
        senior (float): Pensioner UBI amount per week.
        child (float): Child UBI amount per week.
        dis_base (float): Disabled supplement per week.
        regions (ndarray): Numpy float array of 12 UK regional supplements per
            week.
        verbose (bool, optional): Whether to print the calibrated adult UBI
            amount. Defaults to False.

    Returns:
        DataFrame: Reform household-level DataFrame
    """
    basic_income, adult_amount = get_adult_amount(
        base_df,
        budget,
        senior,
        child,
        dis_base,
        regions,
        pass_income=True,
        verbose=verbose,
    )
    basic_income += base_df.age_18_64 * adult_amount
    reform_df = base_df.copy()
    reform_df["basic_income"] = basic_income
    reform_df["net_income"] += basic_income
    return reform_df
