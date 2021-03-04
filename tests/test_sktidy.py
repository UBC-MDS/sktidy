from sktidy import __version__
from sktidy import sktidy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from pytest import raises
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

def test_version():
    assert __version__ == '0.1.0'

def test_tidy_lr():
    X = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    y = datasets.load_iris(return_X_y = True, as_frame = True)[1]
    z = np.random.rand(X.shape[1])

    my_lr = LinearRegression()
    my_lr.fit(X,y)

    my_lr_2 = LinearRegression()

    clf = DecisionTreeClassifier()
    clf.fit(X,y)

    assert sktidy.tidy_lr(model = my_lr, X = X, y = y).shape[0] == X.shape[1]+1, "Output dataframe has number of rows not equal to the number of features in the input dataframe + 1"
    assert sktidy.tidy_lr(model = my_lr, X = X, y = y).shape[1] == 3, "Output dataframe does not have 3 columns"

    with raises(TypeError):
        sktidy.tidy_lr(model = clf, X = X, y = y)

    with raises(TypeError):
        sktidy.tidy_lr(model = my_lr, X = z, y = y)
    
    with raises(NotFittedError):
        sktidy.tidy_lr(model = my_lr_2, X = X, y = y)

def test_tidy_kmeans():
    pass


def test_augment_lr():
    pass


def test_augment_kmeans():
    pass
