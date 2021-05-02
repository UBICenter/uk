import numpy as np
import pandas as pd
import microdf as mdf

# File in repo.
from calc_ubi import get_data, set_ubi


def extract(x: list) -> tuple:
    # Extract parameters and generate reform DataFrame.
    senior, child, dis_base = x[:3]
    regions = np.array(x[3:])
    return senior, child, dis_base, regions


def loss_metrics(
    x: list, baseline_df: pd.DataFrame, reform_base_df: pd.DataFrame, budget: int,
) -> pd.Series:
    """Calculate each potential loss metric.

    :param x: List of optimization elements:
        [senior, child, dis_base, region1, region2, ..., region12]
    :type x: list
    :return: Series with five elements:
        loser_share: Share of the population who come out behind.
        losses: Total losses among losers in pounds.
        mean_pct_loss: Average percent loss across the population
            (including zeroes for people who don't experience losses).
        gini: Gini index of per-person household net income in the reform
            scenario, weighted by person weight at the household level.
    :rtype: pd.Series
    """
    senior, child, dis_base, regions = extract(x)
    reform_df = set_ubi(reform_base_df, budget, senior, child, dis_base, regions,)
    # Calculate loss-related loss metrics.
    change = reform_df.net_income - baseline_df.net_income
    loss = np.maximum(-change, 0)
    weight = baseline_df.household_weight * baseline_df.people
    # Calculate loser share.
    total_pop = np.sum(weight)
    losers = np.sum(weight * (loss > 0))
    loser_share = losers / total_pop
    # Calculate total losses in pounds.
    losses = np.sum(weight * loss)
    # Calculate average percent loss (including zero for non-losers).
    pct_loss = loss / baseline_df.net_income
    # Avoid infinite percent changes and backward changes due to negative
    # baseline income.
    valid_pct_loss = baseline_df.net_income > 0
    total_pct_loss = np.sum(weight[valid_pct_loss] * pct_loss[valid_pct_loss])
    mean_pct_loss = total_pct_loss / total_pop
    # Gini of income per person.
    reform_hh_net_income_pp = reform_df.net_income / baseline_df.people
    # mdf.gini requires a dataframe.
    reform_df = pd.DataFrame(
        {"reform_hh_net_income_pp": reform_hh_net_income_pp, "weight": weight}
    )
    gini = mdf.gini(reform_df, "reform_hh_net_income_pp", "weight")
    # Return Series of all metrics.
    return pd.Series(
        {
            "loser_share": loser_share,
            "losses": losses,
            "mean_pct_loss": mean_pct_loss,
            "gini": gini,
        }
    )
