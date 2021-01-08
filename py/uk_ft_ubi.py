#!/usr/bin/env python
# coding: utf-8

# # UK UBI Analysis
#
# ## Contents
#
#
#
# 1.   Overall
#     1. Changes to government expenditure
#     2. Changes to marginal tax rates
# 2.   Reform comparisons
#     1. UBI specifications
#     2. Percent changes (% loss, % better off)
#     3. Changes by income decile
# 3.   Reform 2
#     1. Intra-decile changes
#     2. Intra-group changes
#     3. Poverty changes by subgroup
# 4.   Individual scenarios
#     1. Marginal tax schedule by claimant type
#     2. Income schedule by claimant type
#     3. Average changes to tax and benefits by claimant type
#
# All metrics are at the person-level unless otherwise stated.
#
#

# ## Overall

#
# ### Changes to government expenditure
#
# All reforms have a budget surplus - each woud generate revenue for the government by raising taxes and removing benefit programs. The surplus for each reform is slightly different due to rounding amounts to the nearest £1/week.

# In[1]:


# @title
from openfisca_uk import PopulationSim, IndividualSim
from openfisca_uk.reforms.modelling import reported_benefits
import numpy as np
import pandas as pd
import plotly

OUTPUT_TYPES = [
  "png",
  "jpeg",
  "webp",
  "svg",
  "eps"
]

def export(fig, name):
  RES_FACTOR = 4
  for extension in OUTPUT_TYPES:
    fig.write_image(f"images/{name}.{extension}", scale=4)

# plotly.io.renderers.default = "colab"
import plotly.express as px
from plotly import graph_objects as go
import microdf as mdf

# from openfisca_uk.reforms.ubi import ubi_reform
from calc_ubi import ubi_reform

REMOVED_BENEFITS = [
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

params = [
    [205, 124, 76, 0, np.zeros((12))],
    [196, 119, 76, 68, np.zeros((12))],
    [194, 115, 73, 69, np.array([5, 4, 0, 3, 3, 2, 7, 5, 3, 4, 2, 0])],
]


reform_1 = ubi_reform(*params[0])
reform_2 = ubi_reform(*params[1])
reform_3 = ubi_reform(*params[2])

reforms = [reform_1, reform_2, reform_3]

baseline_sim = PopulationSim(reported_benefits)
reform_sims = [PopulationSim(reported_benefits, reform) for reform in reforms]

reform_names = ["Reform 1", "Reform 2", "Reform 3"]

people = baseline_sim.calc("household_weight", map_to="person")
benunits = baseline_sim.calc("benunit_weight")
households = baseline_sim.calc("household_weight")

change_vars = [
    "total_tax",
    "state_pension",
    "benefits_modelling",
    "household_net_income"
]

reform_0 = ubi_reform(0, 0, 0, 0, np.zeros((12)))

reform_0_sim = PopulationSim(reported_benefits, reform_0)

changes = []

for var in change_vars:
    original = (baseline_sim.calc(var, map_to="household") * households).sum()
    reformed = (reform_0_sim.calc(var, map_to="household") * households).sum()
    changes += [reformed - original]



def net_cost(sim):
    gain = sim.calc("household_net_income") - baseline_sim.calc(
        "household_net_income"
    )
    return np.sum(gain * households)


fig = go.Figure()
fig.add_trace(go.Bar(x=reform_names, y=[net_cost(sim) for sim in reform_sims]))
fig.update_layout(
    title="Net cost of UBI reforms",
    xaxis_title="Reform",
    yaxis_title="Net cost per year",
    width=1000,
    height=800
)
# fig.show()
export(fig, "net_cost")


# ### Changes to marginal tax rates
#
# The marginal tax rates (calculated as the proportion of a hypothetical increase in earnings which would not increase a person's household's disposable income) for individuals are determined by the tax rates and benefit phase-out rates.
#
# The UK contains significant diversity in each person's receipt of targeted benefits and assessed taxable income. The first graph below shows the rolling average of effective marginal tax rates by income, grouped by family archetype. The graph broadly shows the high marginal tax rate schedules imposed on working families with children, as well as the tax increase on earnings between £100k-125k due to withdrawal of the personal allowance.
#

# In[2]:


# @title
mtr_sim = PopulationSim()
mtr = mtr_sim.calc_mtr()


# In[3]:


# @title
taxable_income = mtr_sim.calc("taxable_income")

group_names = [
    "Single, no children",
    "Couple, no children",
    "Single, with children",
    "Couple, with children",
]

single = mtr_sim.calc("is_single", map_to="person") > 0
couple = single == 0
children = mtr_sim.map_to(
    mtr_sim.calc("is_child", map_to="benunit"),
    entity="benunit",
    target_entity="person",
)

filters = [
    single * (children == 0),
    couple * (children == 0),
    single * (children > 0),
    couple * (children > 0),
]

mtr_groups = []

is_adult = mtr_sim.calc("is_adult")


def round(x, to_nearest=5000):
    return to_nearest * (x // to_nearest)


weight = mtr_sim.calc("adult_weight")
for filter_var in filters:
    subset = pd.DataFrame()
    condition = is_adult * filter_var > 0
    subset["income"] = taxable_income[condition]
    subset["mtr"] = mtr[condition]
    subset["weight"] = weight[condition]
    subset["income"] = subset["income"].apply(round)
    subset = subset[subset["income"] < 150000]
    grouped_subset = subset.groupby("income").mean()
    grouped_subset = grouped_subset[
        subset.groupby("income").sum()["weight"] > 10000
    ]
    mtr_groups += [grouped_subset]


# In[4]:


# @title
fig = go.Figure()

for group, name in zip(mtr_groups, group_names):
    fig.add_trace(
        go.Scatter(x=group.index, y=group["mtr"], mode="lines", name=name)
    )

fig.update_layout(
    title="Current effective marginal tax rates",
    xaxis_title="Taxable income",
    yaxis_title="Effective marginal tax rate",
    yaxis_tickformat="%",
)
export(fig, "current_mtr_lines")


# This graph shows the distribution of marginal tax rates, reflecting the fact that many face no marginal tax rates, a substantial number face between the basic and higher income tax rates alone, and a smaller but still significant minority pay between 60% and 100% effective marginal tax rates.

# In[5]:


# @title
mtr_df = pd.DataFrame()
mtr_df["income"] = mtr_sim.calc("taxable_income")
mtr_df["mtr"] = mtr
mtr_df = mtr_df[mtr_sim.calc("is_adult") > 0]

import plotly.figure_factory as ff

fig = ff.create_distplot(
    [mtr_df["mtr"][np.isfinite(mtr_df["mtr"])]],
    ["MTR"],
    show_hist=True,
    show_rug=False,
    bin_size=0.01,
)
fig.update_layout(
    title="Current effective marginal tax rates",
    xaxis_title="Effective marginal tax rate",
    yaxis_title="Relative frequency",
    xaxis_range=[0, 1],
    xaxis_tickformat="%",
    showlegend=False,
)

export(fig, "current_mtr_hist")


# ## Reform comparisons

#
# ### UBI Specifications
#
# The UBI reforms give the following amounts per week:

# In[22]:


# @title
summary = pd.DataFrame()
region_table = pd.DataFrame()
param_lst = np.array([lst[:-1] + list(lst[-1]) for lst in params]).T

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
cols = ["Senior", "Adult", "Child", "Disability"] + REGION_NAMES
for i in range(len(cols)):
    summary[cols[i]] = param_lst[i]
summary["Name"] = reform_names
summary = summary.set_index("Name").T
for col in summary:
    summary[col] = summary[col].apply(lambda x: "£" + str(int(x)))
summary


# ### Percent changes

# #### Mean percent loss
#
# The mean percent loss is the mean income loss as a percentage of net income among those who lose income from the reform.

# In[7]:


# @title
mean_pct_loss = []


def mean_pct_loss(sim):
    income_b = baseline_sim.calc("household_net_income", map_to="person")
    income_r = sim.calc("household_net_income", map_to="person")
    change = income_r - income_b
    loss = np.maximum(0, -change)
    pct_loss = loss / income_b
    is_valid = np.isfinite(pct_loss)
    pct_loss = pct_loss[is_valid]
    mean_pct_loss = np.average(pct_loss, weights=people[is_valid])
    return mean_pct_loss


fig = go.Figure()
fig.add_trace(
    go.Bar(x=reform_names, y=[mean_pct_loss(sim) for sim in reform_sims])
)
fig.update_layout(
    title="Mean percent loss",
    xaxis_title="Reform",
    yaxis_title="Mean percent loss",
    yaxis_tickformat="%",
)
export(fig, "mean_pct_loss")


# #### Percent better off
#
#

# The percentage better of the simply the number of people whose household disposable income increases. In each proposal a majority see an increase in net income.

# In[8]:


# @title
mean_pct_loss = []


def pct_better_off(sim):
    income_b = baseline_sim.calc("household_net_income", map_to="person")
    income_r = sim.calc("household_net_income", map_to="person")
    pct_better_off = np.average(income_r > income_b, weights=people)
    return pct_better_off


fig = go.Figure()
fig.add_trace(
    go.Bar(x=reform_names, y=[pct_better_off(sim) for sim in reform_sims])
)
fig.update_layout(
    title="Percent better off",
    xaxis_title="Reform",
    yaxis_title="Percent better off",
    yaxis_tickformat="%",
)
export(fig, "pct_better_off")


# ### Changes by income decile
#
# The following graph shows the percentage change to aggregate equivalised household net income for each decile per reform.

# In[9]:


# @title
from rdbl import gbp

df = pd.DataFrame()
df["household_weight"] = baseline_sim.calc("household_weight", map_to="person")
df["household_net_income"] = np.maximum(
    0, baseline_sim.calc("equiv_household_net_income", map_to="person")
)

df["household_net_income"] *= df["household_weight"]
mdf.add_weighted_quantiles(df, "household_net_income", "household_weight")
decile_df = df.groupby("household_net_income_decile").sum()

pct_changes = []
for reform_sim, reform_name in zip(reform_sims, reform_names):
    reform_df = pd.DataFrame()
    reform_df["household_weight"] = df["household_weight"]
    new_net_income = np.maximum(
        0, reform_sim.calc("equiv_household_net_income", map_to="person")
    )
    reform_df["household_net_income"] = new_net_income
    reform_df["household_net_income"] *= reform_df["household_weight"]

    mdf.add_weighted_quantiles(
        reform_df, "household_net_income", "household_weight"
    )
    reform_decile_df = reform_df.groupby("household_net_income_decile").sum()
    pct_changes += [
        (
            reform_decile_df["household_net_income"]
            - decile_df["household_net_income"]
        )
        / decile_df["household_net_income"]
    ]

fig = go.Figure()
for reform_name, pct_change in zip(reform_names, pct_changes):
    fig.add_trace(go.Bar(x=decile_df.index, y=pct_change, name=reform_name))
fig.update_layout(
    title="Effects on household income deciles",
    xaxis_title="Income decile",
    yaxis_title="Average change",
    yaxis_tickformat="%",
    xaxis=dict(tickvals=np.arange(1, 11)),
)
export(fig, "decile_bars")


# ## Reform details
#
# This section contains a more in-depth inspection of the changes to individuals under Reform 2, which is a suitable representative of the shared aspects of the three reforms.

# ### Intra-decile changes
#
# We have that the changes to income are broadly progressive along the income deciles, however there is significant diversity of outcomes within each decile due to differing compositions of income, family and individual characteristics such as disability. The following chart shows the distribution of outcomes within each income decile.

# In[26]:


# @title
import seaborn as sns
import matplotlib

N = 400
COLORS = list(
    map(
        lambda rgba_color: matplotlib.colors.to_hex(rgba_color),
        sns.diverging_palette(220, 20, as_cmap=True)(np.linspace(0, 1, num=N)),
    )
)[::-1]
MIN_PCT_CHANGE = -0.25
MAX_PCT_CHANGE = 0.25
interval_size = (MAX_PCT_CHANGE - MIN_PCT_CHANGE) / (N - 2)
change_bin_bands = pd.IntervalIndex.from_tuples(
    [(-np.inf, MIN_PCT_CHANGE)]
    + [
        (
            MIN_PCT_CHANGE + i * interval_size,
            MIN_PCT_CHANGE + (i + 1) * interval_size,
        )
        for i in range(N - 2)
    ]
    + [(MAX_PCT_CHANGE, np.inf)]
)
change_bin_names = np.array(
    [f"<{MIN_PCT_CHANGE * 100}%"]
    + [
        f"{round((MIN_PCT_CHANGE + i * interval_size) * 100)}%"
        for i in range(N - 2)
    ]
    + [f">{MAX_PCT_CHANGE * 100}%"]
)

reform_sim = reform_sims[1]
reform_name = reform_names[1]
df = pd.DataFrame()
df["household_weight"] = baseline_sim.calc("household_weight", map_to="person")
df["household_net_income"] = np.maximum(
    0, baseline_sim.calc("equiv_household_net_income", map_to="person")
)
# change_bin_names = [f"Decrease more than {interval}%", f"Decrease less than {interval}%", f"Gain less than {interval}%", f"Gain more than {interval}%"]
# change_bin_bands = pd.IntervalIndex.from_tuples([(-np.inf, - interval / 100), (- interval / 100, 0), (0, interval / 100), (interval / 100, np.inf)])

decile_names = list(map(str, range(1, 11))) + ["All"]
"""
COLORS = [
    "#004ba0",  # Dark blue.
    "#63a4ff",  # Light blue.
    "#ffc046",  # Light amber.
    "#c56000",  # Medium amber.
]
"""

df["income_abs_change"] = (
    np.maximum(
        0, reform_sim.calc("equiv_household_net_income", map_to="person")
    )
    - df["household_net_income"]
)
df["income_rel_change"] = df["income_abs_change"] / df["household_net_income"]
df["income_rel_change_band"] = pd.cut(
    df["income_rel_change"], change_bin_bands
)
df["is_disabled_for_ubi"] = baseline_sim.calc("is_disabled_for_ubi") > 0
df["is_sev_disabled_for_ubi"] = (
    baseline_sim.calc("is_severely_disabled_for_ubi") > 0
)
df["is_enh_disabled_for_ubi"] = (
    baseline_sim.calc("is_enhanced_disabled_for_ubi") > 0
)
df["lone_parent"] = baseline_sim.calc("is_lone_parent", map_to="person") > 0
df["lone_senior"] = (
    baseline_sim.calc("is_single_person", map_to="person")
    * baseline_sim.calc("is_SP_age")
    > 0
)
df["poverty"] = baseline_sim.calc("in_poverty_bhc", map_to="person") > 0
mdf.add_weighted_quantiles(df, "household_net_income", "household_weight")


def get_distr(df, condition=None):
    if condition is not None:
        filtered_df = df[condition]
    else:
        filtered_df = df
    grouped_by_outcome = filtered_df.groupby("income_rel_change_band").sum()[
        "household_weight"
    ]
    return grouped_by_outcome / grouped_by_outcome.sum()


fig = go.Figure()

outcomes = []
for subset in [
    df["household_net_income_decile"] == i for i in range(1, 11)
] + [None]:
    outcomes += [get_distr(df, subset)]

outcome_band_shares = np.array(outcomes).transpose()
for outcome_band, name, i in zip(
    outcome_band_shares, change_bin_names, range(N)
):
    labels = np.array([change_bin_names[i]] * 11)
    fig.add_trace(
        go.Bar(
            showlegend=i
            in [0, len(change_bin_bands) // 2, len(change_bin_bands) - 1],
            customdata=labels,
            orientation="h",
            y=decile_names,
            x=outcome_band,
            name=name,
            marker_color=COLORS[i],
            hovertemplate=["Outcome %{customdata}<extra></extra>"],
        )
    )

fig.update_traces(marker_line_width=0)
fig.update_layout(
    barmode="stack",
    yaxis_title="Category",
    xaxis_title="Distribution of outcomes",
    title=reform_name + " - outcome distribution by decile",
    yaxis_type="category",
    xaxis=dict(tickvals=[0.1 * x for x in range(1, 11)], tickformat="%"),
),
export(fig, "intra_decile_changes")


# ### Intra-group changes

# In[25]:


# @title
import seaborn as sns
import matplotlib

N = 400
COLORS = list(
    map(
        lambda rgba_color: matplotlib.colors.to_hex(rgba_color),
        sns.diverging_palette(220, 20, as_cmap=True)(np.linspace(0, 1, num=N)),
    )
)[::-1]
MIN_PCT_CHANGE = -0.25
MAX_PCT_CHANGE = 0.25
interval_size = (MAX_PCT_CHANGE - MIN_PCT_CHANGE) / (N - 2)
change_bin_bands = pd.IntervalIndex.from_tuples(
    [(-np.inf, MIN_PCT_CHANGE)]
    + [
        (
            MIN_PCT_CHANGE + i * interval_size,
            MIN_PCT_CHANGE + (i + 1) * interval_size,
        )
        for i in range(N - 2)
    ]
    + [(MAX_PCT_CHANGE, np.inf)]
)
change_bin_names = np.array(
    [f"<{MIN_PCT_CHANGE * 100}%"]
    + [
        f"{round((MIN_PCT_CHANGE + i * interval_size) * 100)}%"
        for i in range(N - 2)
    ]
    + [f">{MAX_PCT_CHANGE * 100}%"]
)

reform_sim = reform_sims[1]
reform_name = reform_names[1]
df = pd.DataFrame()
df["household_weight"] = baseline_sim.calc("household_weight", map_to="person")
df["household_net_income"] = np.maximum(
    0, baseline_sim.calc("equiv_household_net_income", map_to="person")
)
# change_bin_names = [f"Decrease more than {interval}%", f"Decrease less than {interval}%", f"Gain less than {interval}%", f"Gain more than {interval}%"]
# change_bin_bands = pd.IntervalIndex.from_tuples([(-np.inf, - interval / 100), (- interval / 100, 0), (0, interval / 100), (interval / 100, np.inf)])

decile_names = [
    "Disabled",
    "Lone parent families",
    "Lone seniors",
    "BHC poverty",
]
"""
COLORS = [
    "#004ba0",  # Dark blue.
    "#63a4ff",  # Light blue.
    "#ffc046",  # Light amber.
    "#c56000",  # Medium amber.
]
"""

df["income_abs_change"] = (
    np.maximum(
        0, reform_sim.calc("equiv_household_net_income", map_to="person")
    )
    - df["household_net_income"]
)
df["income_rel_change"] = df["income_abs_change"] / df["household_net_income"]
df["income_rel_change_band"] = pd.cut(
    df["income_rel_change"], change_bin_bands
)
df["is_disabled_for_ubi"] = baseline_sim.calc("is_disabled_for_ubi") > 0
df["is_sev_disabled_for_ubi"] = (
    baseline_sim.calc("is_severely_disabled_for_ubi") > 0
)
df["is_enh_disabled_for_ubi"] = (
    baseline_sim.calc("is_enhanced_disabled_for_ubi") > 0
)
df["lone_parent"] = baseline_sim.calc("is_lone_parent", map_to="person") > 0
df["lone_senior"] = (
    baseline_sim.calc("is_single_person", map_to="person")
    * baseline_sim.calc("is_SP_age")
    > 0
)
df["poverty_bhc"] = baseline_sim.calc("in_poverty_bhc", map_to="person") > 0
mdf.add_weighted_quantiles(df, "household_net_income", "household_weight")


def get_distr(df, condition=None):
    if condition is not None:
        filtered_df = df[condition]
    else:
        filtered_df = df
    grouped_by_outcome = filtered_df.groupby("income_rel_change_band").sum()[
        "household_weight"
    ]
    return grouped_by_outcome / grouped_by_outcome.sum()


fig = go.Figure()

outcomes = []
for subset in [
    df[x] > 0
    for x in [
        "is_disabled_for_ubi",
        "lone_parent",
        "lone_senior",
        "poverty_bhc",
    ]
]:
    outcomes += [get_distr(df, subset)]

outcome_band_shares = np.array(outcomes).transpose()
for outcome_band, name, i in zip(
    outcome_band_shares, change_bin_names, range(N)
):
    labels = np.array([change_bin_names[i]] * 11)
    fig.add_trace(
        go.Bar(
            showlegend=i
            in [0, len(change_bin_bands) // 2, len(change_bin_bands) - 1],
            customdata=labels,
            orientation="h",
            y=decile_names,
            x=outcome_band,
            name=name,
            marker_color=COLORS[i],
            hovertemplate=["Outcome %{customdata}<extra></extra>"],
        )
    )

fig.update_traces(marker_line_width=0)
fig.update_layout(
    hovermode=False,
    barmode="stack",
    yaxis_title="Category",
    xaxis_title="Distribution of outcomes",
    title=reform_name + " - outcome distributions for at-risk groups",
    yaxis_type="category",
    xaxis=dict(tickvals=[0.1 * x for x in range(1, 11)], tickformat="%"),
),
export(fig, "category_changes")


# ### Poverty changes by group
#
# Below we show the changes to two metrics of poverty. We define poverty in line with official UK statistics, as whether the equivalised household (before housing costs) income of a person falls below a threshold (£253 per week in this case). This gives us the metrics of the poverty rate, which is the number of individuals whose household fulfils this requirement, and the poverty gap, which is minimum spending necessary to eliminate poverty.

# In[12]:


# @title
from rdbl import gbp


def poverty_gap(sim, weight):
    return np.sum(
        sim.calc("poverty_gap_bhc", map_to="household")
        * sim.calc("household_weight", map_to="household")
        * weight
    )


def poverty_rate(sim, weight):
    return np.average(
        sim.calc("in_poverty_bhc", map_to="person") > 0,
        weights=sim.calc("household_weight", map_to="person") * weight,
    )


weights = [
    None,
    "is_WA_adult",
    "is_child",
    "is_SP_age",
    "tax_credits",
    "is_disabled_for_ubi",
    "is_lone_parent",
]

fig = go.Figure()

group_names = [
    "All",
    "Working-age adults",
    "Children",
    "Pension-age adults",
    "People receiving Tax Credits",
    "Disabled",
    "Lone parent families",
]
for reform_sim, reform_name in zip(reform_sims, reform_names):
    poverty_gap_reductions = []
    poverty_rate_reductions = []
    for weight in weights:
        if weight == "tax_credits":
            weight_arr = (
                baseline_sim.calc(
                    "working_tax_credit_reported", map_to="household"
                )
                + baseline_sim.calc(
                    "child_tax_credit_reported", map_to="household"
                )
                > 0
            )
            weight_arr_p = (
                baseline_sim.calc(
                    "working_tax_credit_reported", map_to="person"
                )
                + baseline_sim.calc(
                    "child_tax_credit_reported", map_to="person"
                )
                > 0
            )
        if weight is None:
            weight_arr = True
            weight_arr_p = True
        else:
            weight_arr = baseline_sim.calc(weight, map_to="household") > 0
            weight_arr_p = baseline_sim.calc(weight, map_to="person") > 0
        baseline_gap = poverty_gap(baseline_sim, weight_arr)
        reformed_gap = poverty_gap(reform_sim, weight_arr)
        baseline_rate = poverty_rate(baseline_sim, weight_arr_p)
        reformed_rate = poverty_rate(reform_sim, weight_arr_p)
        # print(f"{weight}: base = {gbp(baseline_gap)}, reformed = {gbp(reformed_gap)}")
        # print(f"{weight}: base = {round(baseline_rate * 100)}%, reformed = {round(reformed_rate * 100)}%")
        gap_reduction = (reformed_gap - baseline_gap) / baseline_gap
        rate_reduction = (reformed_rate - baseline_rate) / baseline_rate
        poverty_gap_reductions += [gap_reduction]
        poverty_rate_reductions += [rate_reduction]
    if reform_name == "Reform 2":
        visible = True
    else:
        visible = "legendonly"
    fig.add_trace(
        go.Bar(
            x=group_names,
            y=poverty_gap_reductions,
            name=reform_name + " poverty gap change",
            visible=visible,
        )
    )
    fig.add_trace(
        go.Bar(
            x=group_names,
            y=poverty_rate_reductions,
            name=reform_name + " poverty rate change",
            visible=visible,
        )
    )

fig.update_layout(
    title="Poverty gap reductions, BHC",
    xaxis_title="Group",
    yaxis_title="BHC Poverty gap change",
    yaxis_tickformat="%",
)

export(fig, "poverty_changes")


# ## Individual scenarios
#
# The graphs above show the outcomes on the population at large, but do not examine in detail the changes to individual outcomes.

# ### Changes to marginal tax schedule by claimant type
#
# We examine four hypothetical scenarios:
#
# 1. Single person living alone
# 2. Couple with two children
# 3. Lone parent with two children
# 4. Pensioner couple
#
# In each, we show the baseline and reformed marginal tax rate schedules.

# In[13]:


# @title
from openfisca_uk import IndividualSim


def single_person(sim):
    sim.add_person(
        name="head", age=23, is_benunit_head=True, is_household_head=True
    )
    sim.add_benunit(adults=["head"], universal_credit_reported=True)
    sim.add_household(adults=["head"])
    return sim


def couple_kids(sim):
    sim.add_person(
        name="head", age=23, is_benunit_head=True, is_household_head=True
    )
    sim.add_person(name="secondary", age=22)
    sim.add_person(name="child", age=6),
    sim.add_person(name="child2", age=4),
    sim.add_benunit(
        adults=["head", "secondary"],
        children=["child", "child2"],
        universal_credit_reported=True,
    )
    sim.add_household(
        adults=["head", "secondary"], children=["child", "child2"]
    )
    return sim


def lone_parent(sim):
    sim.add_person(
        name="head", age=23, is_benunit_head=True, is_household_head=True
    )
    sim.add_person(name="child", age=6),
    sim.add_person(name="child2", age=4),
    sim.add_benunit(
        adults=["head"],
        children=["child", "child2"],
        universal_credit_reported=True,
    )
    sim.add_household(adults=["head"], children=["child", "child2"])
    return sim


def lone_pensioner(sim):
    sim.add_person(
        name="head", age=68, is_benunit_head=True, is_household_head=True
    )
    sim.add_benunit(adults=["head"], pension_credit_reported=True)
    sim.add_household(adults=["head"])
    return sim


archetypes = [single_person, couple_kids, lone_parent, lone_pensioner]
archetype_names = [
    "Single",
    "Couple with children",
    "Lone parent",
    "Lone pensioner",
]

fig = go.Figure()

default_shown = "Couple with children"

for archetype, name in zip(archetypes, archetype_names):
    baseline = IndividualSim()
    ubi = IndividualSim(reforms[0])
    baseline = archetype(baseline)
    ubi = archetype(ubi)
    baseline.vary("earnings", min=0, max=200000, step=100)
    baseline.vary("hours", min=0, max=100, step=100 / 2000)
    ubi.vary("earnings", min=0, max=200000, step=100)
    ubi.vary("hours", min=0, max=100, step=100 / 2000)
    earnings = baseline.calc("earnings", target="head")
    mtr_b = baseline.calc_mtr(target="head")
    mtr_r = ubi.calc_mtr(target="head")
    if name == default_shown:
        visible = True
    else:
        visible = "legendonly"
    fig.add_trace(
        go.Scatter(
            x=earnings, y=mtr_b, name=name + " - baseline", visible=visible
        )
    )
    fig.add_trace(
        go.Scatter(
            x=earnings, y=mtr_r, name=name + " - reformed", visible=visible
        )
    )

fig.update_layout(
    title="Changes to marginal tax rates",
    yaxis_tickformat="%",
    xaxis_tickprefix="£",
    xaxis_title="Earnings",
    yaxis_title="Marginal tax rate",
)

export(fig, "hypothetical_mtrs")