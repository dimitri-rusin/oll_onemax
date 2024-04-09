# Repository for generating policies in the OLL-OneMax environment

What this means is, we are creating dynamic parameter configurations for the `(1 + (lambda, lambda))` genetic algorithm with the aim of making it search OneMax landscapes faster.

To build the conda environment:
```sh
./.deploy/BUILD
```

To start generating
```sh
conda activate .deploy/conda_environment
python ./.deploy/range.py
./.deploy/RUN_GENERATE config/continuous/affront.yaml
```

The output will be under `./compute/<your_hostname>/continuous/`. The output is a collection of `.db` files, viewable with [DB Browser for SQLite](https://sqlitebrowser.org/).

You can visualize the `.db` files [using this repository](https://github.com/dimitri-rusin/oll_onemax_visualization.git).

One `.db` file corresponds to one setting, according to which the configurator is run. The configurator will output a configuration of the `(1 + (lambda, lambda))` algorithm. But the configurator itself must be adjusted. This is accomplished via `.yaml` files in the ./config/ subtree.

One such adjustment and run of the configurator is called an experiment. Therefore, one `.db` file is an experiment, in which the configurator is creating more and more policies for the `(1 + (lambda, lambda))` algorithm, as it trains searching OneMax landscapes. Currently, in one experiment, we only train on one OneMax landscape, given by a fixed dimensionality.

