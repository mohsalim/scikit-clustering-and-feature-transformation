from bench_mark import bench_kmeans, bench_em, bench_pca
from openpyxl import Workbook
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn import mixture
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import numpy as np

def part1(data, target, x_train, y_train, x_test, y_test):
    # K-Means
    # TODO find best n_clusters range (1-50?)
    print('--- KMeans ---')
    wb = Workbook()
    ws = wb.active
    headers = ['algorithm', 'k', 'train wall time', 'train accuracy', 'test wall time', 'test accuracy', 'inertia', 'homogeneity', 'completeness', 'v measure', 'ARI', 'AMI', 'FMI']
    ws.append(headers)
    for n in range(1, 31):
        bench_mark = bench_kmeans(KMeans(n_clusters=n, random_state=0), 'KMeans', n, x_train, y_train, x_test, y_test)
        ws.append(bench_mark)
    wb.save('part-1-kmeans-bench.xlsx')

    # EM
    # Run each covariance_type (defaults to full).
    # full: each component has its own general covariance matrix.
    # tied: all components share the same general covariance matrix.
    # diag: each component has its own diagonal covariance matrix.
    # spherical: each component has its own single variance.
    print('--- EM ---')
    cv_types = ['spherical', 'tied', 'diag', 'full']
    wb = Workbook()
    ws = wb.active
    headers = ['algorithm', 'k', 'train wall time', 'train score', 'test wall time', 'test accuracy', 'AIC', 'BIC']
    ws.append(headers)
    for cv_type in cv_types:
        # For each n size components.
        for n in range(1, 31):
            # Run expectation maximization (EM) algorithm.
            # Note: Gaussian Mixture implements EM.
            bench_mark = bench_em(mixture.GaussianMixture(n_components=n, covariance_type=cv_type), 'EM ' + cv_type, n, x_train, y_train, x_test, y_test)
            ws.append(bench_mark)
    wb.save('part-1-em-bench.xlsx')

def part2(data, target, x_train, y_train, x_test, y_test):
    # PCA
    print('--- PCA ---')
    wb = Workbook()
    ws = wb.active
    headers = ['algorithm', 'k', 'train wall time', 'noise variance', 'dimension size after dimensionality reduction']
    ws.append(headers)
    for n in range(1, 28):
        bench_mark = bench_pca(PCA(n_components=n), 'PCA', n, x_train, y_train, x_test, y_test)
        ws.append(bench_mark)
    wb.save('part-2-pca-bench.xlsx')

    # ICA
    print('--- ICA ---')
    ica = FastICA(n_components=2)
    ica.fit(x_train)
    print(ica.components_) 
    print(ica.mixing_ ) 
    print(ica.n_iter_)

    # RCA
    print('--- RCA ---')
    rca = GaussianRandomProjection(n_components=2)
    rca.fit(x_train)
    print(rca.components_) 
    print(rca.n_components_)

    # LDA
    print('--- LDA ---')
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x_train.astype(np.float), y_train.astype(int))
    print(lda.coef_)
    print(lda.intercept_)
    #print(lda.covariance_)
    print(lda.explained_variance_ratio_)
    print(lda.means_)
    print(lda.priors_)
    print(lda.scalings_)
    print(lda.xbar_)
    print(lda.classes_)

def part3(data, target, x_train, y_train, x_test, y_test):
    # PCA -> KMeans
    print('--- PCA -> KMeans ---')
    n = 2
    pca = PCA(n_components=n).fit(x_train)
    # In this case the seeding of the centers is deterministic.
    # Hence we run the k-means algorithm only once with n_init=1.
    # Else wise it will give a warning and set n_init to 1 anyways.
    kmeans = KMeans(init=pca.components_, n_clusters=n, n_init=1, random_state=0).fit(x_train)
    #print(kmeans.labels_)
    predicted_labels = kmeans.predict(x_test)
    #print(predicted_labels)
    # For every array, get the last item. Then squeeze all them together so it's a 1D array instead of a 2D array.
    expected_labels = y_test
    #print(expected_labels)
    score = metrics.accuracy_score(expected_labels, predicted_labels)    
    print("kmeans with pca " + str(n) + ": " + str(score))

