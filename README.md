# sktidy 

![](https://github.com/UBC-MDS/sktidy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/sktidy/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/sktidy) ![Release](https://github.com/UBC-MDS/sktidy/workflows/Release/badge.svg) [![Documentation Status](https://readthedocs.org/projects/sktidy/badge/?version=latest)](https://sktidy.readthedocs.io/en/latest/?badge=latest)

Python package that produces tidy output for sklearn model evaluation!

## Summary

Sktidy implements a `tidy` and `augment` function for Scikit learn linear regression and kmeans clustering to ease model selection and assesment tasks. The `tidy`
family of functions will provide similar functionality to `tidy` in the pybroom package but for sklearn models, returning a neat pandas dataframe with important model information at the level of features or clusters for linear regression and kmeans clustering respectively. The `augment` function will provide information at the level of the orignal data point on how points were clustered and silhoutte scores for kmeans clustering and predicted values and residuals for linear regression in a neat pandas data frame. 

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ sktidy
```

## Features

* `tidy_kmeans()`: Returns inertia, cluster location, and number of associated points at the level of clusters in a tidy format.
* `tidy_lr()`: Returns coefficients and corresponding feature names in a tidy format.
* `augment_lr()` : Returns predictions and residuals for each point in the training data set in a tidy format.
* `augment_kmeans()` : Returns assigned cluster and distance from cluster center for the data the kmeans algorithm was fitted with in a tidy format.

## Dependencies

- TODO

## Usage

- TODO

## Documentation

The official documentation is hosted on Read the Docs: https://sktidy.readthedocs.io/en/latest/

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/UBC-MDS/sktidy/graphs/contributors).

The original contributors to the project were:
- Jacob McFarlane
- Asma Odaini
- Peter Yang
- Heidi Ye

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
