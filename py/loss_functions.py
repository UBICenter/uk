import numpy as np
import pandas as pd
from openfisca_uk.tools.simulation import Simulation
import microdf as mdf


DATA_DIR = "~/frs"

baseline_sim = Simulation(data_dir=DATA_DIR)


def calc2df(
    sim: Simulation, cols: list, map_to: str = "person"
) -> pd.DataFrame:
    """Make a DataFrame from an openfisca-uk Simulation.

    :param sim: Simulation object to extract from.
    :type sim: Simulation
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


# Predefine a DataFrame for speed.
BASELINE_COLS = [
    "is_SP_age",
    "is_child",
    "is_disabled",
    "is_enhanced_disabled",
    "is_severely_disabled",
    "region",
    "household_weight",
    "people_in_household",
    "household_net_income",
]

baseline_df = calc2df(baseline_sim, BASELINE_COLS, map_to="household")


def loss_metrics(
    reform_sim: Simulation,
    baseline_sim: Simulation = None,
    population: str = None,
) -> pd.Series:
    """Calculate each potential loss metric.

    :param reform_sim: Reform simulation object.
    :type reform_sim: Simulation
    :param baseline_sim: Baseline simulation object. Defaults to the baseline
        previously defined and extracted into baseline_df.
    :type baseline_sim: Simulation, optional
    :param population: Variable indicating the subpopulation to calculate
        losses for. Defaults to people_in_household, i.e. all people.
    :type population: str, optional
    :return: Series with five elements:
        loser_share: Share of the population who come out behind.
        losses: Total losses among losers in pounds.
        mean_pct_loss: Average percent loss across the population
            (including zeroes for people who don't experience losses).
        mean_pct_loss_pwd2: Average percent loss across the population, with
            double weight given to people with disabilities.
        reform_gini: Gini index of per-person household net income in the
            reform scenario, weighted by person weight at the household level.
    :rtype: pd.Series
    """
    reform_hh_net_income = reform_sim.calc("household_net_income")
    # If a different baseline is provided, make baseline_df.
    if baseline_sim is not None:
        baseline_df = calc2df(baseline_sim, BASELINE_COLS)
    change = reform_hh_net_income - baseline_df.household_net_income
    loss = np.maximum(-change, 0)
    if population:
        weight_var = baseline_sim.calc(population, map_to="household")
    else:
        weight_var = baseline_df.people_in_household
    weight = baseline_df.household_weight * weight_var
    # Calculate loser share.
    total_pop = np.sum(weight)
    losers = np.sum(weight * (loss > 0))
    loser_share = losers / total_pop
    # Calculate total losses in pounds.
    losses = np.sum(weight * loss)
    # Calculate average percent loss (including zero for non-losers).
    pct_loss = loss / baseline_df.household_net_income
    valid_pct_loss = np.isfinite(pct_loss)
    total_pct_loss = np.sum(weight[valid_pct_loss] * pct_loss[valid_pct_loss])
    mean_pct_loss = total_pct_loss / total_pop
    # Calculate average percent loss with double weight for PWD.
    pwd2_weight = weight * np.where(baseline_df.is_disabled, 2, 1)
    total_pct_loss_pwd2 = np.sum(
        pwd2_weight[valid_pct_loss] * pct_loss[valid_pct_loss]
    )
    total_pop_pwd2 = pwd2_weight.sum()  # Denominator.
    mean_pct_loss_pwd2 = total_pct_loss_pwd2 / total_pop_pwd2
    # Gini of income per person.
    reform_hh_net_income_pp = reform_hh_net_income / people_in_household
    # mdf.gini requires a dataframe.
    reform_df = pd.DataFrame(
        {"reform_hh_net_income_pp": reform_hh_net_income_pp, "weight": weight}
    )
    reform_gini = mdf.gini(reform_df, "reform_hh_net_income_pp", "weight")
    # Return Series of all metrics.
    return pd.Series(
        [loser_share, losses, mean_pct_loss, mean_pct_loss_pwd2, reform_gini],
        index=[
            "loser_share",
            "losses",
            "mean_pct_loss",
            "mean_pct_loss_pwd2",
            "reform_gini",
        ],
    )
