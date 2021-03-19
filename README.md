# sktidy 

![](https://github.com/UBC-MDS/sktidy/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/sktidy/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/sktidy) [![Deploy](https://github.com/UBC-MDS/sktidy/actions/workflows/deploy.yml/badge.svg)](https://github.com/UBC-MDS/sktidy/actions/workflows/deploy.yml) [![Documentation Status](https://readthedocs.org/projects/sktidy/badge/?version=latest)](https://sktidy.readthedocs.io/en/latest/?badge=latest)

Python package that produces tidy output for sklearn model evaluation!

## Summary

Sktidy implements a `tidy` and `augment` function for Scikit learn linear regression and kmeans clustering to ease model selection and assessment tasks. The `tidy` family of functions will provide similar functionality to `tidy` in the [Pybroom](https://pybroom.readthedocs.io/en/dev/) but for sklearn models, returning a tidy pandas dataframe with important model information at the level of features or clusters for linear regression and kmeans clustering respectively. The `augment` function will provide information at the level of the original data point on how points were clustered and silhoutte scores for kmeans clustering and predicted values and residuals for linear regression in a neat pandas data frame. 


## How `sktidy` fits into the Python ecosystem

The functions `tidy` and `augment` functions are inspired by the functions `tidy` and `augment` in the [Pybroom](https://pybroom.readthedocs.io/en/dev/) package which is inspired by the R library [broom](https://github.com/dgrtwo/broom). The current implementation of Pybroom support scipy and lmfit objects. Sklearn lacks a similar functionality that allows users to obtain model fitting results in a tidy dataframe that makes it easy to process and plot the data. Tidy dataframes allows plotting libraries to automatically generate plots to compare many variables without the need for lengthy data cleaning and wrangling. Plotting libraries supporting tidy DataFrames include seaborn, recent versions of matplotlib, bokeh and altair.


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

The dependencies for this package are:

- python = "^3.8"
- pandas = "^1.2.3"
- sklearn = "^0.0"
- statsmodels = "^0.12.2"
- python-semantic-release = "^7.15.0"

For more details, you can check out the [`pyproject.toml` file](https://github.com/UBC-MDS/sktidy/blob/main/pyproject.toml)

## Usage

### Linear Regression

```python
    # Importing packages
    from sklearn.linear_model import LinearRegression
    from sklearn import datasets
    import pandas as pd
    import sktidy
    # Load data and traning the linear regression model
    X = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    y = datasets.load_iris(return_X_y = True, as_frame = True)[1]
    lr_model = LinearRegression()
    lr_model.fit(X,y)
    # Get tidy output for the trained sklearn LinearRegression model
    tidy_lr(model = lr_model, X = X, y = y)
    # Getting predicted y values and residuals
    augment_lr(model = lr_model,X = X,y = y)
```

### KMeans

```python
    # Importing packages
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import datasets
    import pandas as pd
    import sktidy
    # Extracting data and training the clustering algorithm
    df = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    kmeans_clusterer = KMeans()
    kmeans_clusterer.fit(df)
    # Getting the tidy df of cluster information
    tidy_kmeans(model = kmeans_clusterer, X = df)
    # Getting cluster assignment for each data point
    augment_kmeans(model = kmeans_clusterer, X = df)
```

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
