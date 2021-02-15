# THIS DOESN'T CURRENTLY WORK WHEN TRYING TO RUN FROM BASH.

# This all only works from the root level folder.
import os
if 'py' not in os.listdir("."):
    os.chdir("..")

from py.loss_functions import loss_metrics
from py.optimize import optimize

import pandas as pd

AGE_CATEGORIES = ["senior", "child"]
DIS_CATEGORIES = ["dis_base"]
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
categories = AGE_CATEGORIES + DIS_CATEGORIES + REGIONS

# Define bounds
AGE_BOUNDS = (40, 240)  # Child, working-age adult, senior.
DIS_BOUNDS = (0, 200)  # Base, severe, and enhanced disability UBI supplements.
GEO_BOUNDS = (-50, 50)  # Will be relative to a baseline geo.
# Skip adult which is calculated.
bounds = (
    [AGE_BOUNDS] * len(AGE_CATEGORIES)
    + [DIS_BOUNDS] * len(DIS_CATEGORIES)
    + [GEO_BOUNDS] * len(REGIONS)
)

input_dict = {category: bound for category, bound in zip(categories, bounds)}


def opt(reform):
    return optimize(
        input_dict,
        "mean_pct_loss",
        reform,
        verbose=False,
        seed=0,
        # Reforms don't always improve upon one another with the
        # default tolerance of 0.01.
        tol=0.0001,
    )


reform_1 = opt("reform_1")

reform_2 = opt("reform_2")

reform_3 = opt("reform_3")

# Check that iterations improve.
assert reform_2[0].fun < reform_1[0].fun, "Reform 2 doesn't improve on 1"
assert reform_3[0].fun < reform_2[0].fun, "Reform 3 doesn't improve on 2"

ubi_params = pd.DataFrame(reform_1[1]).T
ubi_params.loc[1] = reform_2[1]
ubi_params.loc[2] = reform_3[1]
ubi_params["mean_pct_loss"] = [
    reform_1[0].fun,
    reform_2[0].fun,
    reform_3[0].fun,
]
# Export to top-level folder.
ubi_params.to_csv("../optimal_params.csv", index=False)
