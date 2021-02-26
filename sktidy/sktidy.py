#Peter
def tidy_lr():
    pass

def tidy_kmeans(kmeans):
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

    # Extracting Data and Traning the clustering algorithm
    df = datasets.load_iris(return_X_y = True, as_frame = True)[0]
    keans_clusterer = KMeans()
    kmeans_clusterer.fit(df)

    # Getting the tidy df of cluster information 
    tidy_kmeans(model = kmeans_clusterer, dataframe = df)
    """
    pass

#Heidi
def augment_lr(X,y):
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

#Asma
def augment_kmeans():
    pass