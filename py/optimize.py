import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, OptimizeResult
from py.loss_functions import loss_metrics, extract
from py.calc_ubi import get_data, get_adult_amount


def optimize(
    input_dict: dict,
    loss_metric: str,
    reform: str,
    verbose: bool = True,
    path: str = None,
    **kwargs
) -> tuple:
    """Also accepts **kwargs passed to differential_evolution.

    :param input_dict: Dict with format {category: (min, max)} specifying the
        bounds for UBI amounts for each category. If min == max, the amount is
        fixed.
    :type input_dict: dict
    :param loss_metric: Type of loss metric to be used in optimization.
    :type loss_metric: str
    :param reform: Type of reform to apply.
    :type reform: str
    :param verbose: Bool specifying whether or not to print each function
        evaluation, defaults to True
    :type verbose: bool, optional
    :param path: Path to FRS files, defaults to None in which case the files
        are loaded via frs.load().
    :type path: str, optional
    :return: Tuple of OptimizeResult with the optimal solution and dict with
        solution for each UBI component.
        Also prints a dict with the optimal solution and loss metrics.
    :rtype: tuple
    """

    # Declare categories
    AGE_CATEGORIES = ["senior", "child"]
    DIS_CATEGORIES = ["dis_base", "dis_severe", "dis_enhanced"]
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

    # Set bounds according to chosen reform.
    ZERO = [(0, 0)]
    bounds = [input_dict[i] for i in AGE_CATEGORIES]
    if reform == "reform_1":  # Child/adult/senior only.
        bounds += ZERO * (len(DIS_CATEGORIES) + len(REGIONS))
    else:  # Add disability supplements to reforms 2 and 3.
        bounds += [input_dict[i] for i in DIS_CATEGORIES]
        if reform == "reform_2":
            bounds += ZERO * len(REGIONS)
        else:  # Reform 3.
            bounds += [input_dict[i] for i in REGIONS[:-1]]
            bounds += ZERO  # Last geo is a baseline.

    baseline_df, reform_base_df, budget = get_data(path=path)

    # Take the average value of each tuple to create array of starting values
    x = [((i[0] + i[1]) / 2) for i in bounds]

    # Create full list (in order) of categories.
    categories = ["adult"] + AGE_CATEGORIES + DIS_CATEGORIES + REGIONS

    def loss_func(x, args=(loss_metric)):
        loss_metric = args

        # Calculate loss with current solution (given the adult amount)
        loss = loss_metrics(x, baseline_df, reform_base_df, budget)[
            loss_metric
        ]

        if verbose:
            (
                senior,
                child,
                dis_base,
                dis_severe,
                dis_enhanced,
                regions,
            ) = extract(x)
            adult_amount = get_adult_amount(
                reform_base_df,
                budget,
                senior,
                child,
                dis_base,
                dis_severe,
                dis_enhanced,
                regions,
                individual=True,
            )
            x = np.insert(x, 0, adult_amount)
            output_dict = {
                categories[i]: x[i] for i in range(len(x))
            }

            # Print loss and corresponding solution set
            print("Loss: {}".format(loss))
            print(output_dict)

        return loss

    result = differential_evolution(func=loss_func, bounds=bounds, **kwargs)

    # Performance of optimal solutions on other loss metrics
    loss_dict = loss_metrics(
        result.x, baseline_df, reform_base_df, budget
    ).to_dict()
    print("Loss by all metrics:\n", loss_dict, "\n")

    # Get adult amount
    senior, child, dis_base, dis_severe, dis_enhanced, regions = extract(
        result.x
    )
    adult_amount = get_adult_amount(
        reform_base_df,
        budget,
        senior,
        child,
        dis_base,
        dis_severe,
        dis_enhanced,
        regions,
        individual=True,
    )

    # Construct pandas Series of optimal result and insert adult amount.
    optimal_x = pd.Series(
        np.insert(result.x, 0, adult_amount), index=categories
    )

    # Print optimal loss
    print("Optimal {}:".format(loss_metric), result.fun, "\n")

    # Make geo supplements positive by shifting negatives to child/adult/senior.
    min_region = min(optimal_x)
    optimal_x.loc[AGE_CATEGORIES + ["adult"]] += min_region
    optimal_x.loc[REGIONS] -= min_region

    optimal_x = optimal_x.round()
    # Print optimal solution.
    print("Optimal solution:\n", optimal_x)

    return result, optimal_x
