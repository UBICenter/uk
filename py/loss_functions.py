import numpy as np
import pandas as pd
import microdf as mdf

# File in repo.
import calc_ubi


base, reform, budget = get_data("~/frs")


def loss_metrics(x: list) -> pd.Series:
    """Calculate each potential loss metric.

    :param x: List of optimization elements:
        [senior, child, dis_1, dis_2, dis_3, region1, region2, ..., region12]
    :type x: list
    :return: Series with five elements:
        loser_share: Share of the population who come out behind.
        losses: Total losses among losers in pounds.
        mean_pct_loss: Average percent loss across the population
            (including zeroes for people who don't experience losses).
        mean_pct_loss_pwd2: Average percent loss across the population, with
            double weight given to people with disabilities.
        poverty_gap_bhc: Poverty gap before housing costs.
        poverty_gap_ahc: Poverty gap after housing costs.
        gini: Gini index of per-person household net income in the reform
            scenario, weighted by person weight at the household level.
    :rtype: pd.Series
    """
    senior, child, dis_1, dis_2, dis_3 = x[:5]
    regions = np.array(x[5:])
    
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
    # Poverty gap.
    bhc_pov_gaps = np.maximum(
        baseline_df.absolute_poverty_bhc - reform_hh_net_income, 0
    )
    ahc_pov_gaps = np.maximum(
        baseline_df.absolute_poverty_ahc - reform_hh_net_income, 0
    )
    # TODO: Make this work with a filtered group.
    poverty_gap_bhc = np.sum(bhc_pov_gaps * baseline_df.household_weight)
    poverty_gap_ahc = np.sum(ahc_pov_gaps * baseline_df.household_weight)
    # Gini of income per person.
    reform_hh_net_income_pp = (
        reform_hh_net_income / baseline_df.people_in_household
    )
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
            "mean_pct_loss_pwd2": mean_pct_loss_pwd2,
            "poverty_gap_bhc": poverty_gap_bhc,
            "poverty_gap_ahc": poverty_gap_ahc,
            "gini": gini,
        }
    )
