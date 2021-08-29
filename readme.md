# Blank Slate UBI model

This repo contains the code required to reproduce the figures in the UBI Center [Blank Slate UBI report](https://www.ubicenter.org/uk-blank-slate-ubi).


## Reproducing

Dependencies:

- anaconda

1. clone or download this repository
2. Create the anaconda environment with `conda env create -f environment.yml` from within the cloned directory
   <!-- TODO: fix versions to known-working ones. -->
3. `conda activate uk`

You can now recreate our results. The main calculations can be performed like so:

- Run `python py/run_optimisation.py` and wait
   - This will use a stochastic optimisation process to determine "optimal" UBI values for each of the three reform schemes. It may take a few minutes.
   - The loss and calculated values will be printed and written into `optimal_params.csv`.
   - The exact values you calculate will differ slightly from those in the report because the optimisation is stochastic. This is expected.

You can recreate our summary tables with:

- `python py/summary_tables.py`

And our tables and figures with the `jupyter` notebook `jb/uk_ft_ubi.ipynb`. To use the notebook you will need to install a few more packages:

  - `pip install "git+https://github.com/UBICenter/ubicenter.py@5acc40c70b5495f634c2d18399db22f04603a4b1"`
  - `conda install -c plotly plotly-orca`

Then run `jupyter notebook`, open the notebook and run its cells.

Note also that some of the marginal tax rate cells at the bottom of the notebook will not run correctly. They depend on an older version of `openfisca-uk`. <!-- this is a guess -->


## Modifying the model

The modifications to the OpenFisca-UK microsimulation model are described in the `ubi_reform()` function in `py/calc_ubi.py`.

The optimisation process and loss function are described in `py/optimize.py` and `py/loss_functions.py`.
