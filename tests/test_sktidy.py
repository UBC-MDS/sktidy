from sktidy import __version__
from sktidy import sktidy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from pytest import raises
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.cluster import KMeans


def test_version():
    assert __version__ == "0.1.0"


def test_tidy_lr():
    X = datasets.load_iris(return_X_y=True, as_frame=True)[0]
    y = datasets.load_iris(return_X_y=True, as_frame=True)[1]
    z = np.random.rand(X.shape[1])

    my_lr = LinearRegression()
    my_lr.fit(X, y)

    my_lr_2 = LinearRegression()

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    assert (
        sktidy.tidy_lr(model=my_lr, X=X, y=y).shape[0] == X.shape[1] + 1
    ), "Output dataframe has number of rows not equal to the number of features in the input dataframe + 1"
    assert (
        sktidy.tidy_lr(model=my_lr, X=X, y=y).shape[1] == 3
    ), "Output dataframe does not have 3 columns"

    with raises(TypeError):
        sktidy.tidy_lr(model=clf, X=X, y=y)

    with raises(TypeError):
        sktidy.tidy_lr(model=my_lr, X=z, y=y)

    with raises(NotFittedError):
        sktidy.tidy_lr(model=my_lr_2, X=X, y=y)


def test_tidy_kmeans():
    pass


def test_augment_lr():
    X = datasets.load_iris(return_X_y=True, as_frame=True)[0]
    y = datasets.load_iris(return_X_y=True, as_frame=True)[1]
    z = np.random.rand(X.shape[1])

    my_lr = LinearRegression()
    my_lr.fit(X, y)

    my_lr_2 = LinearRegression()

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    assert (
        sktidy.augment_lr(model=my_lr, X=X, y=y).shape[0] == X.shape[0]
    ), "Output dataframe has the same number of rows as the input dataframe."
    assert (
        sktidy.augment_lr(model=my_lr, X=X, y=y).shape[1] == X.shape[1] + 3
    ), "Output dataframe the same number of columns as X + y + 2"
    assert (
        (
            sktidy.augment_lr(model=my_lr, X=X, y=y)["residuals"]
            + sktidy.augment_lr(model=my_lr, X=X, y=y)["predictions"]
            == sktidy.augment_lr(model=my_lr, X=X, y=y)["target"]
        )
        .eq(True)
        .all()
    ), "Predictions and residuals should sum to target"

    with raises(TypeError):
        sktidy.augment_lr(model=clf, X=X, y=y)

    with raises(TypeError):
        sktidy.augment_lr(model=my_lr, X=z, y=y)

    with raises(NotFittedError):
        sktidy.augment_lr(model=my_lr_2, X=X, y=y)


def test_augment_kmeans():
    # Extracting data and traning the clustering algorithm
    X = datasets.load_iris(return_X_y=True, as_frame=True)[0]
    y = datasets.load_iris(return_X_y=True, as_frame=True)[1]
    kmeans_clusterer = KMeans()
    kmeans_clusterer.fit(X)

    z = np.random.rand(X.shape[1])

    kmeans_clusterer_2 = KMeans()

    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    assert (
        sktidy.augment_kmeans(model=kmeans_clusterer, X=X).shape[0] == X.shape[0]
    ), "Output dataframe has the same number of rows as the input dataframe."
    assert (
        sktidy.augment_kmeans(model=kmeans_clusterer, X=X).shape[1] == X.shape[1] + 1
    ), "Output dataframe the same number of columns as X + 1"

    assert np.all(
        (sktidy.augment_kmeans(model=kmeans_clusterer, X=X)["cluster"]).values
        == kmeans_clusterer.labels_,
    ), "Cluster assignments are the same as labels"

    with raises(TypeError):
        sktidy.augment_kmeans(model=clf, X=X)

    with raises(TypeError):
        sktidy.augment_kmeans(model=kmeans_clusterer, X=z)

    with raises(NotFittedError):
        sktidy.augment_kmeans(model=kmeans_clusterer_2, X=X)
