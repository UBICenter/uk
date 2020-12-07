# Universal Basic Income in the United Kingdom

## Introduction

The British government has sought to simplify the benefit system by consolidating six benefits into a single payment, known as Universal Credit. Despite the name, Universal Credit is not universal, as factors such as income and number of children affect the amount of benefit a claimant will receive. The rollout of Universal Credit has been plagued with problems, potentially opening the door to a reconsideration of the UK’s benefit system. Additionally, the COVID-19 pandemic has possibly altered the British public’s perception of the current benefits system. In April 2020, 51% percent of British adults said they would be supportive of the introduction of a Universal Basic Income (YouGov, 2020).

Using the OpenFisca microsimulation framework, UK taxes and benefits have been modelled in OpenFisca-UK. Four basic income simulations were designed and simulated using this model, using data from the 2018/19 Family Resources Survey (FRS). The four static microsimulations implemented different types of basic income systems, funded by different tax reforms and benefit changes.

## Reforms

Four reforms were analysed, each with a different distribution of basic income payments, but all following a core strategy. In common between all reforms are:

- Funding from a 50% flat Income Tax (National Insurance is removed)
- Removal of means-tested benefits
  - Jobseeker’s Allowance (contribution-based and income-based)
  - Income Support
  - Pension Credit
  - State Pension
  - Working Tax Credit
  - Child Tax Credit
  - Child Benefit
  - Housing Benefit
- A basic income of £165 per week for citizens over State Pension age

From this, each of the reforms explores an intersection of two variables in the basic income:

1. Whether children should receive half of the adult UBI, or the full amount
2. Whether a disability supplement should exist

  - In these reforms, we reverse the adult-child ratio of the basic income for disability to equalise overall basic income among the disabled.

### Reform cross-sections

|                       |        | Adult-child ratio |              |   |   |
|-----------------------|--------|:-----------------:|:------------:|---|---|
|                       |        | 2:1               | 1:1          |   |   |
| Disability supplement | None   | half_ubi          | full_ubi     |   |   |
|                       | Exists | half_dis_ubi      | full_dis_ubi |   |   |

### Weekly amounts

We decide weekly amounts by applying the changes constant across all reforms, and then using the amounts that are constrained by the cross-sectional parameters and produce a budget neutral reform.

## Reform budgets
We can see the net cost of each reform to be negative - each reform generates a budget surplus.

from openfisca_uk.tools.simulation import model, entity_df
from openfisca_uk.tools.reforms import solve_ft_ubi_reform
import numpy as np
from plotly import express as px
from plotly import graph_objects as go
import pandas as pd
from IPython.display import display
from rdbl import gbp
import warnings
# warnings.simplefilter('ignore')

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
    "housing_benefit",
    "ESA_income",
    "ESA_contrib"
]

half_ubi, half_ubi_params = solve_ft_ubi_reform(pensioner_amount=165, wa_adult_coef=1, child_coef=0.5, abolish_benefits=CORE_BENEFITS)
half_ubi_dis, half_ubi_dis_params = solve_ft_ubi_reform(pensioner_amount=165, wa_adult_coef=1, child_coef=0.5, adult_disability_coef=0.5, child_disability_coef=1, abolish_benefits=CORE_BENEFITS)
full_ubi, full_ubi_params = solve_ft_ubi_reform(pensioner_amount=165, wa_adult_coef=1, child_coef=1, abolish_benefits=CORE_BENEFITS)
full_ubi_dis, full_ubi_dis_params = solve_ft_ubi_reform(pensioner_amount=165, wa_adult_coef=1, child_coef=1, adult_disability_coef=1, child_disability_coef=1, abolish_benefits=CORE_BENEFITS)

reforms = [half_ubi, half_ubi_dis, full_ubi, full_ubi_dis]
models = [model(reform) for reform in reforms]
reform_names = ["Half", "Half+Dis.", "Full", "Full+Dis."]

params = [half_ubi_params, half_ubi_dis_params, full_ubi_params, full_ubi_dis_params]
'''
# Convergence graph
fig = go.Figure()

for i in range(4):
    fig.add_trace(go.Scatter(name=reform_names[i], x=params[i][4], y=params[i][5]))

fig.update_layout(title="Core UBI amount by net cost of reform")
fig.show()
'''

info = pd.DataFrame()

def read_param(i):
    return np.array([np.round(params[j][i], 2) for j in range(4)])

info["Reform"] = reform_names
info["Abled Pensioners"] = [165.00] * 4
info["Disabled Pensioners"] = info["Abled Pensioners"] + read_param(2)
info["Abled WA Adult"] = read_param(0)
info["Disabled WA Adult"] = read_param(0) + read_param(2)
info["Abled Child"] = read_param(1)
info["Disabled Child"] = read_param(1) + read_param(3)

baseline = model()
period = "2020-10-10"

households = baseline.calculate("household_weight", period)

def net_cost(reform):
    return np.sum((reform.calculate("household_net_income_ahc", period) - baseline.calculate("household_net_income_ahc", period)) * households) * 52

net_costs = [gbp(net_cost(reform)) for reform in models]

info["Net cost"] = net_costs

info.set_index("Reform", inplace=True)

display(info)

## Distributional effects

### Effects on income deciles

We see that the outcomes of all proposed basic income reforms are progressive.

from plotly import graph_objects as go
bars = []

for reform_model, reform_name in zip(models, reform_names):
    reform_df = pd.DataFrame()
    reform_df["net_gain"] = (reform_model.calculate("household_net_income_ahc", period) - baseline.calculate("household_net_income_ahc", period))
    reform_df["weight"] = baseline.calculate("household_weight", period)
    reform_df["household_income"] = baseline.calculate("household_income", period)
    reform_df["decile"], bins = pd.qcut(reform_df["household_income"], 10, retbins=True, duplicates="drop")
    reform_df = reform_df.groupby(by="decile").mean()
    bars.append(go.Bar(name=reform_name, x=np.arange(1, 11), y=reform_df["net_gain"]))
    
fig = go.Figure(data=bars)
fig.update_layout(barmode='group', title="Average net gain per household income decile", xaxis_title="Income decile", yaxis_title="Average net gain per week", xaxis=dict(tickvals=np.arange(1, 11)), yaxis=dict(tickprefix="£"))
fig.show()

All reform simulations give an average net gain to the bottom six household deciles, with the burden shouldered primarily by the top two deciles. We see also the change to individual components of household income and expenditure - showing the sources of the overall changes shown above, in the basic income payments and National Insurance removal.

import seaborn as sns

bars = []

variables = [
    "benunit_income_tax",
    "benunit_NI",
    "benunit_earnings",
    "benunit_pension_income",
    "benunit_interest",
    "benunit_state_pension",
    "working_tax_credit",
    "child_tax_credit",
    "JSA_income",
    "pension_credit",
    "housing_benefit",
    "universal_credit",
    "income_support",
    "benunit_misc"
]

vars_full = [
    "Income Tax",
    "National Insurance",
    "Earnings",
    "Pension income",
    "Investment income",
    "State Pension",
    "Working Tax Credit",
    "Child Tax Credit",
    "Jobseeker's Allowance",
    "Pension Credit",
    "Housing Benefit",
    "Universal Credit",
    "Income Support",
    "Other income",
    "Basic Income"
]

i = 0
colours = sns.color_palette("deep", len(variables) + 1)
var_color = {var: (colour[0] * 255, colour[1] * 255, colour[2] * 255) for var, colour in zip(variables + ["basic_income"], colours)}

baseline = model()
baseline_df = pd.DataFrame()
baseline_df["benunit_income"] = baseline.calculate("benunit_income", period)
_, bins = pd.qcut(baseline_df["benunit_income"], 10, retbins=True, duplicates="drop")
for sim, reform_name in zip([model()] + models, ["Baseline"] + reform_names):
    reform_df = pd.DataFrame()
    reform_df["weight"] = sim.calculate("benunit_weight", period)
    reform_df["benunit_income"] = sim.calculate("benunit_income", period)
    reform_df["decile"] = pd.cut(baseline_df["benunit_income"], bins)
    widths = bins[1:] - bins[:-1]
    for var in variables:
        reform_df[var] = sim.calculate(var, period)
    if reform_name != "Baseline":
        reform_df["basic_income"] = sim.calculate("benunit_basic_income", period)
    else:
        reform_df["basic_income"] = np.zeros_like(reform_df["weight"])
    reform_df = reform_df.groupby(by="decile").mean()
    total_reform = - reform_df["benunit_income_tax"]  - reform_df["benunit_NI"]
    for var, full_name in zip(variables + ["basic_income"], vars_full):
        bars.append(go.Bar(name=full_name, marker_color=f"rgb{var_color[var]}", x=np.arange(1, 11), y=reform_df[var], offsetgroup=i, base=total_reform))
        total_reform += np.array(reform_df[var])
    i += 1
    
fig = go.Figure(data=bars)
fig.update_layout(barmode='group', title="Net change to family finance", xaxis_title="Income decile", yaxis_title="Average net change per week", xaxis=dict(tickvals=np.arange(1, 11)), yaxis=dict(tickprefix="£"))
fig.show()

### Intra-decile changes to household income
While we have seen the broadly progressive effect on average members of each decile, that does not imply that members of each decile are affected equally. Each decile has a range of outcomes, primarily due to different compositions of household members, income sources and other circumstances.

from ipywidgets.widgets import Tab

figs = []

for reform, reform_name in zip(models, reform_names):
    fig = go.FigureWidget()
    reform_df = pd.DataFrame()
    reform_df["baseline_household_income"] = np.maximum(0, baseline.calculate("equiv_household_net_income_bhc", period))
    reform_df["household_income_rel_change"] = (np.maximum(0, reform.calculate("equiv_household_net_income_bhc", period)) - reform_df["baseline_household_income"]) / reform_df["baseline_household_income"]
    reform_df["decile"] = pd.qcut(reform_df["baseline_household_income"], 10, duplicates="drop", labels=False)

    change_bin_names = ["Decrease more than 5%", "Decrease less than 5%", "Gain less than 5%", "Gain more than 5%"]
    change_bin_bands = [[-np.inf, -0.05], [-0.05, 0], [0, 0.05], [0.05, np.inf]]
    decile_names = list(map(str, range(1, 11)))
    COLORS = [
        "#004ba0",  # Dark blue.
        "#63a4ff",  # Light blue.
        "#ffc046",  # Light amber.
        "#c56000",  # Medium amber.
    ]
    i = 0
    for name, bands in zip(change_bin_names, change_bin_bands):
        percent_in_bin = []
        for decile in range(10):
            total_decile_weight = np.sum(np.where(reform_df["decile"] == decile, households, 0))
            decile_weight_in_change_band = np.sum(np.where((reform_df["decile"] == decile) & (reform_df["household_income_rel_change"] > bands[0]) & (reform_df["household_income_rel_change"] <= bands[1]), households, 0))
            if name is "Gain more than 5%":
                decile_weight_in_change_band += np.sum(np.where((reform_df["decile"] == decile) & (reform_df["household_income_rel_change"] > bands[0]) & np.isinf(reform_df["household_income_rel_change"]), households, 0))
            percent_in_bin += [100 * (decile_weight_in_change_band / total_decile_weight)]
        fig.add_trace(go.Bar(orientation="h", y=decile_names, x=percent_in_bin, name=name, marker_color=COLORS[3-i]))
        i += 1

    fig.update_layout(barmode="stack", yaxis_title="Household income decile", xaxis_title="Distribution of outcomes", title=reform_name + " - outcome distribution by decile", yaxis=dict(tickvals=np.arange(1, 11)))
    figs += [fig]

output = [fig.show() for fig in figs]
'''
plot = Tab(figs)
[plot.set_title(i, name) for i, name in zip(range(4), reform_names)]
plot'''

### Inequality
We see that all reforms substantially reduce inequality as measured by the Gini index.

import microdf as mdf

def percent_reduction(before, after):
    return (after - before) / before

gini_reductions = []

for reformed in models:
    household_net_bhc = pd.DataFrame()
    household_net_bhc["w"] = baseline.calculate(
        "household_weight", period
    ) * baseline.calculate("people_in_household", period)
    household_net_bhc["baseline"] = baseline.calculate(
        "equiv_household_net_income_bhc", period
    )
    household_net_bhc["reform"] = reformed.calculate(
        "equiv_household_net_income_bhc", period
    )
    baseline_gini = mdf.gini(household_net_bhc, "baseline", w="w")
    reform_gini = mdf.gini(household_net_bhc, "reform", w="w")
    gini_reduction = percent_reduction(baseline_gini, reform_gini)
    gini_reductions.append(gini_reduction * 100)

fig = px.bar(x=reform_names, y=gini_reductions, title="Gini reduction")
fig.update_yaxes(ticksuffix="%")
fig.update_layout(xaxis_title="Reform", yaxis_title="Percentage reduction")
fig.show()

## Poverty
### Effects on core demographics
We also see substantial reductions in elderly, working-age and child poverty, with child poverty receiving the strongest reduction. The disability supplements have a small effect on working-age and child poverty, but a larger effect on senior poverty - likely caused by the higher rate of disability among the elderly.

from plotly import graph_objects as go
from plotly.subplots import make_subplots

def poverty_rate(sim, cross_section_var, mode="bhc", period="2020-09-10"):
    return np.sum(
        sim.calculate("in_poverty_" + mode, period)
        * sim.calculate(cross_section_var, period)
        * sim.calculate("household_weight", period)
    ) / np.sum(
        sim.calculate(cross_section_var, period)
        * sim.calculate("household_weight", period)
    )

bars = []

poverty_vars = [
    "seniors",
    "working_age_adults",
    "children"
]

poverty_names = [
    "Senior citizens",
    "Working-age adults",
    "Children"
]

reform_sims = models

fig = make_subplots(rows=1, cols=2,
                    subplot_titles=("BHC Poverty", "AHC Poverty"))

for poverty_var, poverty_name in zip(poverty_vars, poverty_names):
    poverty_reductions = []
    baseline_rate = poverty_rate(baseline, f"{poverty_var}_in_household",
                                 period=period)
    for sim, name in zip(reform_sims, reform_names):
        new_rate = poverty_rate(sim, f"{poverty_var}_in_household",
                                period=period)
        relative_change = (new_rate - baseline_rate) / baseline_rate * 100
        poverty_reductions.append(relative_change)
    fig.add_trace(go.Bar(name=poverty_name + " BHC", x=np.arange(1, 5),
                         y=poverty_reductions), row=1, col=1)

for poverty_var, poverty_name in zip(poverty_vars, poverty_names):
    poverty_reductions = []
    baseline_rate = poverty_rate(baseline, f"{poverty_var}_in_household",
                                 period=period, mode="ahc")
    for sim, name in zip(reform_sims, reform_names):
        new_rate = poverty_rate(sim, f"{poverty_var}_in_household",
                                period=period, mode="ahc")
        relative_change = (new_rate - baseline_rate) / baseline_rate * 100
        poverty_reductions.append(relative_change)
    fig.add_trace(go.Bar(name=poverty_name + " AHC", x=np.arange(1, 5),
                         y=poverty_reductions),
                  row=1, col=2)
    
fig.update_layout(title_text="Changes to absolute poverty rates",
                  yaxis=dict(ticksuffix="%"))
    
fig.show()

### Poverty reduction by circumstance
We see the effects on specific demographics, for Reform 1:

from plotly import graph_objects as go
from plotly.subplots import make_subplots

def poverty_rate(sim, cross_section_var, period="2020-09-10"):
    return np.sum(
        sim.calculate("benunit_in_poverty_bhc", period)
        * sim.calculate(cross_section_var, period)
        * sim.calculate("benunit_weight", period)
        * sim.calculate("people_in_benunit", period)
    ) / np.sum(
        sim.calculate(cross_section_var, period)
        * sim.calculate("benunit_weight", period)
        * sim.calculate("people_in_benunit", period)
    )

bars = []

poverty_vars = [
    "benunit_receiving_disability_benefits",
    "benunit_receiving_income_support",
    "benunit_receiving_tax_credits",
    "couple_childless_benunit",
    "couple_parents_benunit",
    "self_emp_benunit",
    "capital_benunit"
]

period = "2020-10-28"

baseline = model()

fig = go.Figure()

for poverty_var in poverty_vars:
    poverty_reductions = []
    baseline_rate = poverty_rate(baseline, f"{poverty_var}", period=period)
    for sim, name in zip(reform_sims, reform_names):
        new_rate = poverty_rate(sim, f"{poverty_var}", period=period)
        relative_change = (new_rate - baseline_rate) / baseline_rate * 100
        poverty_reductions.append(relative_change)
    fig.add_trace(go.Bar(name=poverty_var.replace("_", " "),
                         x=np.arange(1, 5), y=poverty_reductions))
    
fig.update_layout(title_text="Changes to absolute poverty rates, BHC",
                  yaxis=dict(ticksuffix="%"))
    
fig.show()

## Marginal Tax Rates
### Baseline MTRs

We see the current marginal tax rates on individuals:

from openfisca_uk.tools.simulation import entity_df, model, calc_mtr
import warnings
warnings.filterwarnings('ignore')
mtr = calc_mtr()
sim = model()
adults = entity_df(sim, entity="person")
adults["gross_income"] *= 52
adults["MTR"] = mtr * 100
fig = px.scatter(data_frame=adults, x="gross_income", y="MTR", opacity=0.05)

fig.update_layout(xaxis_title="Annual earnings",
                  yaxis_title="Marginal tax rate",
                  xaxis=dict(range=[-5000, 200000], tickprefix="£"),
                  yaxis=dict(range=[-10, 110], ticksuffix="%"))
fig.show()

Note that annual income is composed of a number of sources including employment, self-employment, pensions, investment and benefits. We calculate the effective marginal tax rate for each individual by asking the following question: if an individual's (employer-paid) earnings increased by £25 per week, what percentage of that £25 would not appear in the individual's new family net income per week. There is a wide variety of reasons that the additional pound would not be fully preserved, whether part of it goes to Income Tax, National Insurance, or is cancelled out by a reduction in means-tested benefits such as the Tax Credits (41% withdrawal rate), Housing Benefit (65% withdrawal rate), Pension Credit (as high as 100% in low-income individuals) or other benefits and taxes. Where diagonal lines appear, these do not correlate to any specific policy but are instead the result of a non-infinitesimal increase being used: if the bonus takes the individual over a boundary, then parts of the bonus will have different marginal tax rates and therefore the individual MTR will be the average of these. In simulations of the UBI reforms, the marginal tax rate for all individuals with taxable income becomes 50%.

### Changes to MTR statistics

mtr = calc_mtr()
new_mtr = calc_mtr(reforms[0])

mtr = mtr[np.isfinite(mtr) * (baseline.calculate("is_adult", period) > 0)]
new_mtr = new_mtr[np.isfinite(new_mtr) *
                  (baseline.calculate("is_adult", period) > 0)]

mtr_info = pd.DataFrame()
mtr_info["System"] = ["Baseline", "Reform"]
mtr_info["1st Percentile"] = [np.percentile(mtr, 1),
                              np.percentile(new_mtr, 1)]
mtr_info["10th Percentile"] = [np.percentile(mtr, 10),
                               np.percentile(new_mtr, 10)]
mtr_info["Median"] = [np.median(mtr), np.median(new_mtr)]
mtr_info["90th Percentile"] = [np.percentile(mtr, 90),
                               np.percentile(new_mtr, 90)]
mtr_info["99th Percentile"] = [np.percentile(mtr, 99),
                               np.percentile(new_mtr, 99)]
mtr_info["Mean"] = [np.average(mtr), np.average(new_mtr)]

mtr_info.set_index("System", inplace=True)
display(mtr_info)