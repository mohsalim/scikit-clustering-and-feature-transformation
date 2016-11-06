import time
from sklearn import metrics
from scipy import stats
import scipy as sp
import numpy as np
from sklearn.feature_selection import SelectFromModel

def bench_kmeans(estimator, name, k, x_train, y_train, x_test, y_test):
    bench_kmeans_format = '% 9s   %.2i    %.3f   %.2fs    %.3f   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'   #%.3f   %.3f'

    # Train.
    start = time.time()
    estimator.fit(x_train)
    train_time = time.time() - start

    # Test.
    start = time.time()
    predicted = estimator.predict(x_test)
    test_time = time.time() - start
    
    #print('% 9s' % 'init   k'
    #  '    time   acc  inertia    homo   compl  v-meas     ARI AMI  FMI')
    results = (name,
               k,
               train_time,
               metrics.accuracy_score(y_train, estimator.labels_),
               test_time,
               metrics.accuracy_score(y_test, predicted),
               estimator.inertia_,
               metrics.homogeneity_score(y_train, estimator.labels_),
               metrics.completeness_score(y_train, estimator.labels_),
               metrics.v_measure_score(y_train, estimator.labels_),
               metrics.adjusted_rand_score(y_train, estimator.labels_),
               metrics.adjusted_mutual_info_score(y_train,  estimator.labels_),
               metrics.fowlkes_mallows_score(y_train, estimator.labels_))
         #metrics.calinski_harabaz_score(x_train, estimator.labels_),
         #metrics.silhouette_score(x_train, estimator.labels_, metric='euclidean')))
    
    #print(bench_kmeans_format % (results))

    # TODO why does silhouette score throw an error about label size?
    # TODO all x_train, estimator.labels_, and y_train all have same size (and its not 1)
    # TODO should we pass explicit sample size to silhouette score? len(x_train)

    return list(results)

def bench_em(estimator, name, k, x_train, y_train, x_test, y_test):
    bench_em_format = '% 9s   %.2i    %.3f   %.3f    %.3f   %.3f   %.3f   %.3f '

    # Train.
    start = time.time()
    estimator.fit(x_train)
    train_time = time.time() - start

    # Test.
    start = time.time()
    predicted = estimator.predict(x_test)
    test_time = time.time() - start
    
    results = (name,
               k,
               train_time,
               estimator.score(x_train, y_train),
               test_time,
               metrics.accuracy_score(y_test, predicted),
               estimator.aic(x_train),
               estimator.bic(x_train))
    
    #print(bench_em_format % (results))

    return list(results)

def get_kurtosis(x):
    # Convert array to float (assuming it's a float!).
    x = x.astype(np.float)
    # Describe data.
    n, min_max, mean, var, skew, kurt = sp.stats.describe(x)
    
    #print('number of points: ' + str(n))
    #print('min/max: ' + str(min_max))
    #print('mean: ' + str(mean))
    #print('variance: ' + str(var))
    #print('skew: ' + str(skew))
    #print('kurtosis: ' + str(kurt))
    #print('median: ' + str(sp.median(x)))

    # We only care about kurtosis.
    return kurt

def get_min_max_average(x):
    x_min = min(x)
    x_max = max(x)
    x_avg = sum(x)/len(x)
    return (x_min, x_max, x_avg)

def safely_get_dimensionality_reduction(x_transformed):
    base = len(x_transformed[0]);  
    for xt in x_transformed:
        xtl = len(xt)
        if xtl != base:
            print('DR base = ' + str(base) + ', DR inconsistent value = ' + str(xtl))
    return base

def bench_pca(estimator, name, k, x_train, y_train, x_test, y_test):
    bench_pca_format = '% 9s   %.2i    %.3f   %.3f    %.3f'   #%.3f'

    # Train.
    start = time.time()   
    x_transformed = estimator.fit_transform(x_train)
    train_time = time.time() - start

    # Note, the length of any
    results = (name,
               k,
               train_time,
               #estimator.score(x_train, y_train),
               # The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop 1999.
               # See “Pattern Recognition and Machine Learning” by C. Bishop, 12.2.1 p. 574 or http://www.miketipping.com/papers/met-mppca.pdf.
               # It is required to computed the estimated data covariance and score samples.
               estimator.noise_variance_,
               safely_get_dimensionality_reduction(x_transformed),
               # Principal axes in feature space, representing the directions of maximum variance in the data. The components are sorted by explained_variance_.
               # Note the components are the eigen vectors.
               str(estimator.components_),
               # The amount of variance explained by each of the selected components.
               # Note the explained variance is the eigen values: http://stackoverflow.com/a/31941631/2498729
               str(estimator.explained_variance_),
               # Percentage of variance explained by each of the selected components.
               # If n_components is not set then all components are stored and the sum of explained variances is equal to 1.0.
               str(estimator.explained_variance_ratio_),
               # Per-feature empirical mean, estimated from the training set.
               # Equal to X.mean(axis=1).
               str(estimator.mean_),
               # The estimated number of components.
               # When n_components is set to ‘mle’ or a number between 0 and 1 (with svd_solver == ‘full’) this number is estimated from input data.
               # Otherwise it equals the parameter n_components, or n_features if n_components is None.
               estimator.n_components_)
               
    #print(bench_pca_format % (results))

    return list(results)

def bench_ica(estimator, name, k, x_train, y_train, x_test, y_test):
    bench_ica_format = '% 9s   %.2i   %.2i    %.3f   %.3f'

    # Train.
    start = time.time()   
    x_transformed = estimator.fit_transform(x_train)
    train_time = time.time() - start

    origianl_kurt = get_kurtosis(x_train)
    (ok_min, ok_max, ok_avg) = get_min_max_average(origianl_kurt)
    
    transformed_kurt = get_kurtosis(x_transformed)
    (tk_min, tk_max, tk_avg) = get_min_max_average(transformed_kurt)

    components_kurt = get_kurtosis(estimator.components_)
    (ck_min, ck_max, ck_avg) = get_min_max_average(components_kurt)

    mixing_kurt = get_kurtosis(estimator.mixing_)
    (mk_min, mk_max, mk_avg) = get_min_max_average(mixing_kurt)

    results = (name,
               k,
               train_time,
               # If the algorithm is “deflation”, n_iter is the maximum number of iterations run across all components.
               # Else they are just the number of iterations taken to converge.
               estimator.n_iter_,
               safely_get_dimensionality_reduction(x_transformed),
               # The unmixing matrix.
               str(estimator.components_),
               # The mixing matrix.
               str(estimator.mixing_),
               str(origianl_kurt),
               ok_min,
               ok_max,
               ok_avg,
               str(transformed_kurt),
               tk_min,
               tk_max,
               tk_avg,
               str(components_kurt),
               ck_min,
               ck_max,
               ck_avg,
               str(mixing_kurt),
               mk_min,
               mk_max,
               mk_avg)
               
    #print(bench_ica_format % (results))

    return list(results)

def bench_rca(estimator, name, k, iterations, x_train, y_train, x_test, y_test):
    # Train
    start = time.time()
    data = x_train
    for i in range(0, iterations):
        estimator.fit(data)
        data = estimator.components_
    train_time = time.time() - start
    
    results = (name,
               k,
               iterations,
               train_time,
               str(estimator.components_),
               estimator.n_components)

    return results

def bench_lda(estimator, name, k, x_train, y_train, x_test, y_test):
    # Train
    start = time.time()
    estimator.fit(x_train.astype(np.float), y_train.astype(int))
    train_time = time.time() - start

    results = (name,
               k,
               train_time,
               # Weight vector(s).
               str(estimator.coef_),
               # Intercept term.
               str(estimator.intercept_),
               # Covariance matrix (shared by all classes).
               #str(covariance_),
               # Percentage of variance explained by each of the selected components.
               # If n_components is not set then all components are stored and the sum of explained variances is equal to 1.0.
               # Only available when eigen or svd solver is used.
               str(estimator.explained_variance_ratio_),
               # Class means.
               str(estimator.means_),
               # Class priors (sum to 1).
               str(estimator.priors_),
               # Scaling of the features in the space spanned by the class centroids.
               str(estimator.scalings_),
               # Overall mean.
               str(estimator.xbar_),
               # Unique class labels.
               str(estimator.classes_))

    return results

def bench_etr(estimator, name, x_train, y_train, x_test, y_test):
    # Train
    x_train_float = x_train.astype(np.float)
    start = time.time()
    estimator.fit(x_train_float, y_train.astype(int))
    train_time = time.time() - start

    model = SelectFromModel(estimator, prefit=True)
    x_new = model.transform(x_train_float)

    results = (name,
               train_time,
               str(x_train_float.shape),
               str(x_new.shape),
               str(estimator.feature_importances_))

    return results
