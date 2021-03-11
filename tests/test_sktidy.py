from sktidy import __version__
from sktidy import sktidy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import datasets
from pytest import raises
import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError


@pytest.fixture
def KMeans_in_data():
    kmeans_input = pd.DataFrame(
        {
            "x": np.array([1, 1, 1, 1, 3, 3, 3, 3]),
            "y": np.array([1, 3, 7, 9, 1, 3, 7, 9]),
        }
    )
    return kmeans_input


@pytest.fixture
def KMeans_out_data():
    center_0 = pd.DataFrame(np.array([[2.0, 2.0]]), columns=["x", "y"])
    center_1 = pd.DataFrame(np.array([[2.0, 8.0]]), columns=["x", "y"])

    kmeans_tidy_output = pd.DataFrame(
        {
            "cluster_number": np.array([0, 1]),
            # "cluster_inertia" : np.array([8,8]),
            "cluster_center": [center_0, center_1],
            "n_points": np.array([4, 4]),
        }
    )
    return kmeans_tidy_output


def test_version():
    assert __version__ == "0.1.0"


def test_tidy_lr():

    # feature and target toy data
    X = datasets.load_iris(return_X_y=True, as_frame=True)[0]
    y = datasets.load_iris(return_X_y=True, as_frame=True)[1]

    # A numpy array to test erroneous input
    z = np.random.rand(X.shape[1])

    # fitted sklearn LinearRegression object
    my_lr = LinearRegression()
    my_lr.fit(X, y)

    # non-fitted sklearn LinearRegression object to test for NotFittedError
    my_lr_2 = LinearRegression()

    # decision tree object to test fot input TypeError
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # test output shape
    assert (
        sktidy.tidy_lr(model=my_lr, X=X, y=y).shape[0] == X.shape[1] + 1
    ), "Output dataframe should have number of rows that equal to the number \
        of features in the input dataframe + 1 (intercept)"
    assert (
        sktidy.tidy_lr(model=my_lr, X=X, y=y).shape[1] == 3
    ), "Output dataframe should have 3 columns corresponding to feature name, \
        coefficient and p value"

    # test whether erroneous input got catched
    with raises(TypeError):
        sktidy.tidy_lr(model=clf, X=X, y=y)

    with raises(TypeError):
        sktidy.tidy_lr(model=my_lr, X=z, y=y)

    with raises(TypeError):
        sktidy.tidy_lr(model=my_lr, X=X, y=z)

    with raises(NotFittedError):
        sktidy.tidy_lr(model=my_lr_2, X=X, y=y)


def test_tidy_kmeans(KMeans_in_data, KMeans_out_data):

    # Fitting a KMeans model for testing
    kmeans_test = KMeans(n_clusters=2)
    kmeans_test.fit(KMeans_in_data)

    # Getting our sktidy output
    tidy_output = sktidy.tidy_kmeans(model=kmeans_test,
                                     dataframe=KMeans_in_data)

    # Comparing it with our expected outputs
    tidy_output.equals(KMeans_out_data)

    # Checking that we raise a type error when the wrong model is input
    X = datasets.load_iris(return_X_y=True, as_frame=True)[0]
    y = datasets.load_iris(return_X_y=True, as_frame=True)[1]
    # z = np.random.rand(X.shape[1]) z is never used according to flake8

    my_lr = LinearRegression()
    my_lr.fit(X, y)
    with raises(TypeError):
        sktidy.tidy_kmeans(model=my_lr, dataframe=KMeans_in_data)

    # Creating an untrained model and checking that we raise a NotFittedError \
    # when we try to use tidy on it
    with raises(NotFittedError):
        sktidy.tidy_kmeans(model=KMeans(), dataframe=KMeans_in_data)


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
        sktidy.augment_kmeans(model=kmeans_clusterer, X=X)
              .shape[0] == X.shape[0]
    ), "Output dataframe has the same number of rows as the input dataframe."
    assert (
        sktidy.augment_kmeans(model=kmeans_clusterer, X=X)
              .shape[1] == X.shape[1] + 1
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
