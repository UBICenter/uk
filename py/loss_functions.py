import numpy as np
import pandas as pd
from openfisca_uk.tools.simulation import Simulation


DATA_DIR = "~/frs"
baseline = Simulation(data_dir=DATA_DIR)

def calc2df(sim, cols):
    d = {}
    for i in cols:
        d[i] = sim.calc(i, map_to="household")
    return pd.DataFrame(d)

baseline = Simulation(data_dir=DATA_DIR)

# Predefine a DataFrame for speed.
BASELINE_COLS = ["is_SP_age",
                 "is_child",
                 "is_disabled",
                 "is_enhanced_disabled",
                 "is_severely_disabled",
                 "region",
                 "household_weight",
                 "people_in_household",
                 "household_net_income"]

baseline_df = calc2df(baseline, BASELINE_COLS)


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
    :return: Series with three elements:
        1) loser_share: Share of the population who come out behind.
        2) losses: Total losses among losers in pounds.
        3) mean_pct_loss: Average percent loss across the population
            (including zeroes for people who don't experience losses).
    :rtype: pd.Series
    """
    change = (
        reform_sim.calc("household_net_income")
        - baseline_df.household_net_income
    )
    loss = np.maximum(-change, 0)
    if population:
        weight_var = baseline_sim.calc(population, map_to="household")
    else:
        weight_var = baseline_df.people_in_household
    weight = baseline_df.household_weight * weight_var
    # Calculate loser share.
    total = np.sum(weight)
    losers = np.sum(weight * (loss > 0))
    loser_share = losers / total
    # Calculate total losses in pounds.
    losses = np.sum(weight * loss)
    # Calculate average percent loss (including zero for non-losers).
    pct_loss = loss / baseline_df.household_net_income
    valid_pct_loss = np.isfinite(pct_loss)
    # Not sure this is working:
    total_pct_loss = np.sum(weight[valid_pct_loss] * pct_loss[valid_pct_loss])
    mean_pct_loss = total_pct_loss / total
    # Not sure why this isn't working?
    # mean_pct_loss = np.mean(weight * pct_loss, where=~pct_loss.isnull())
    return pd.Series(
        [loser_share, losses, mean_pct_loss],
        index=["loser_share", "losses", "mean_pct_loss"],
    )
