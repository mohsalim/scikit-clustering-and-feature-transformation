import time
from sklearn import metrics

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
               estimator.noise_variance_,
               safely_get_dimensionality_reduction(x_transformed))
               
    #print(bench_pca_format % (results))

    return list(results)

def bench_ica(estimator, name, k, x_train, y_train, x_test, y_test):
    bench_ica_format = '% 9s   %.2i   %.2i    %.3f   %.3f'

    # Train.
    start = time.time()   
    x_transformed = estimator.fit_transform(x_train)
    train_time = time.time() - start

    # Note, the length of any
    results = (name,
               k,
               train_time,
               estimator.n_iter_,
               safely_get_dimensionality_reduction(x_transformed))
               
    #print(bench_ica_format % (results))

    return list(results)
