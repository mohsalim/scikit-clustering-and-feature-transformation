from bench_mark import bench_kmeans, bench_em
from openpyxl import Workbook
from sklearn.cluster import KMeans
from sklearn.random_projection import GaussianRandomProjection
from sklearn import mixture

def part1(data, target, x_train, x_test, y_train, y_test):
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
