def tidy_lr(model, df):
    """
    Returns a tidy dataframe for sklearn LinearRegression model with feature names, coefficients/intercept and p-values

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression object
        The fitted sklearn LinearRegression model

    df: pandas dataframe
        The feature dataframe to which the LinearRegression object was fitted

    Returns
    -------
    tidy_dataframe : pandas dataframe
        A dataframe with n+1 rows, where n is the number of features in the feature dataframe that was 
        fitted to the model and 3 columns, describing feature names, coefficients/intercept and p-values

    Examples
    --------
    from sklearn.linear_model import LinearRegression
    from sklearn import datasets
    import pandas as pd
    import sktidy

    # Load data and traning the linear regression model
    X = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    y = datasets.load_iris(return_X_y = True, as_frame = True)[1]
    my_lr = LinearRegression()
    my_lr.fit(X,y)

    # Get tidy output for the trained sklearn LinearRegression model
    tidy_lr(model = my_lr, df = X)
    """
    pass


def tidy_kmeans(model, dataframe):
    """
    Return a tidy df of cluster information for a kmeans clustering algorithm

    This function delivers diagnostic information about each cluster defined by an instance of
    scikit learn's implementation of kmeans clustering including total intertia in each cluster,
    cluster center, and total number of points associated with each cluster.

    Parameters
    ----------
    model : sklearn.cluster.KMeans
    The model to extract the cluster specific information from.

    dataframe : pandas dataframe
        The data to which the Kmeans object has been fitted

    Returns
    -------
    df : pandas dataframe
        A dataframe with k rows, where k is the number of clusters and 3 columns,
        describing respectively the center of the cluster, the sum of inertia of the
        cluster, and the number of associated data points in a cluster.

    Examples
    --------
    # Importing packages
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn import datasets
    import pandas as pd
    import sktidy

    # Extracting data and traning the clustering algorithm
    df = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    kmeans_clusterer = KMeans()
    kmeans_clusterer.fit(df)

    # Getting the tidy df of cluster information
    tidy_kmeans(model = kmeans_clusterer, dataframe = df)
    """
    pass


def augment_lr(X, y):
    """
    Adds two columns to the original data of the linear regression model. This includes predictions and residuals.

    Parameters
    ----------
    X : pandas.core.frame.DataFrame
        A dataframe of explanatory variables to predict on. Shaped n observations by m features.

    y : pandas.core.frame.DataFrame
        A dataframe of response variables to predict on. Shaped n observations by 1.

    Returns
    -------
    df : pandas.core.frame.DataFrame
        A dataframe with the original data plus two additional columns for predictions and residuals. Shaped n observations by m features + 2.

    """
    pass


def augment_kmeans(model, X):
    """
    This function returns a dataframe of the original samples with their assigned clusters based
    on predictions make by an instance of scikit learn's implementation of KMeans clustering.

    Parameters
    ----------
    model : sklearn.cluster.KMeans
        The model to extract the cluster specific information from

    X : pandas dataframe
        The data to which the Kmeans object has been fitted

    Returns
    -------
    augment_dataframe : pandas dataframe
        A dataframe with k rows, where k is the number of examples in X and 2 columns of the
        data points in X and their corresponding predicted label

    Examples
    --------
    # Importing packages
    from sklearn.cluster import KMeans
    from sklearn import datasets
    import pandas as pd
    import sktidy

    # Extracting data and traning the clustering algorithm
    df = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    kmeans_clusterer = KMeans()
    kmeans_clusterer.fit(df)

    # Getting cluster assignment for each data point
    augment_kmeans(model = kmeans_clusterer, X = df)
    """
    pass
