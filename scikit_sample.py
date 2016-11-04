from sklearn.cluster import KMeans
import arff, numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import mixture
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import time
from openpyxl import Workbook

BENCH_KMEANS_FORMAT = '% 9s   %.2i    %.3f   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f'   #%.3f   %.3f'

def bench_kmeans(estimator, name, k, x_train, y_train):
    t0 = time.time()
    estimator.fit(x_train)
    #print('% 9s' % 'init   k'
    #  '    time   acc  inertia    homo   compl  v-meas     ARI AMI  FMI')
    results = (name,
         k,
         (time.time() - t0),
         metrics.accuracy_score(y_train, estimator.labels_),
         estimator.inertia_,
         metrics.homogeneity_score(y_train, estimator.labels_),
         metrics.completeness_score(y_train, estimator.labels_),
         metrics.v_measure_score(y_train, estimator.labels_),
         metrics.adjusted_rand_score(y_train, estimator.labels_),
         metrics.adjusted_mutual_info_score(y_train,  estimator.labels_),
         metrics.fowlkes_mallows_score(y_train, estimator.labels_))
         #metrics.calinski_harabaz_score(x_train, estimator.labels_),
         #metrics.silhouette_score(x_train, estimator.labels_, metric='euclidean')))
    
    print(BENCH_KMEANS_FORMAT % (results))

    # TODO why does silhouette score throw an error about label size?
    # TODO all x_train, estimator.labels_, and y_train all have same size (and its not 1)
    # TODO should we pass explicit sample size to silhouette score? len(x_train)

    # Return results as an array instead of a tuple.
    return list(results)

def part1(data, target, x_train, x_test, y_train, y_test, wb):
    # Set up graph stuff.
    x = []
    accuracy_y = []
    train_time_y = []
    test_time_y = []
    legend = []

    # K-Means
    # TODO find best n_clusters range (1-50?)
    print('--- KMeans ---')
    ws = wb.active
    headers = ['algorithm', 'k', 'wall time', 'accuracy', 'inertia', 'homogeneity', 'completeness', 'v measure', 'ARI', 'AMI', 'FMI']
    ws.append(headers)

    for n in range(1, 31):
        x.append(n)
        # Run k-means algorithm.
        start = time.time()   
        kmeans = KMeans(n_clusters=n, random_state=0).fit(x_train, y_train)
        train_time = time.time() - start
        train_time_y.append(train_time)
        #print(kmeans.labels_)
        start = time.time()   
        predicted_labels = kmeans.predict(x_test)
        test_time = time.time() - start
        test_time_y.append(test_time)
        #print(predicted_labels)
        # For every array, get the last item. Then squeeze all them together so it's a 1D array instead of a 2D array.
        expected_labels = y_test
        #print(expected_labels)
        score = metrics.accuracy_score(expected_labels, predicted_labels)
        accuracy_y.append(score)
        bench_mark = bench_kmeans(KMeans(n_clusters=n, random_state=0), 'KMeans', n, x_train, y_train)
        ws.append(bench_mark)
        #print(kmeans.labels_)
        #print("kmeans " + str(n) + ": " + str(score))
        #print(kmeans.cluster_centers_)
    wb.save('part-1-kmeans-bench.xlsx')

    plt.figure(1)
    plt.plot(x, accuracy_y)

    plt.figure(2)
    plt.plot(x, train_time_y)

    plt.figure(3)
    plt.plot(x, test_time_y)

    legend.append('K-Means')

    # EM
    # Run each covariance_type (defaults to full).
    # full: each component has its own general covariance matrix.
    # tied: all components share the same general covariance matrix.
    # diag: each component has its own diagonal covariance matrix.
    # spherical: each component has its own single variance.
    print('--- EM ---')
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        # Restart all graph arrays.
        x = []
        accuracy_y = []
        train_time_y = []
        test_time_y = []
        # For each n size components.
        for n in range(1, 31):
            x.append(n)
            # Run expectation maximization (EM) algorithm.
            # Note: Gaussian Mixture implements EM.
            em = mixture.GaussianMixture(n_components=n, covariance_type=cv_type)
            start = time.time()
            em.fit(x_train, y_train)
            train_time = time.time() - start
            train_time_y.append(train_time)
            #print(em.labels_)
            start = time.time()   
            predicted_labels = em.predict(x_test)
            test_time = time.time() - start
            test_time_y.append(test_time)
            #print(predicted_labels)
            # For every array, get the last item. Then squeeze all them together so it's a 1D array instead of a 2D array.
            expected_labels = y_test
            #print(expected_labels)
            score = metrics.accuracy_score(expected_labels, predicted_labels)
            accuracy_y.append(score)
            #print("em " + cv_type + " " + str(n) + ": " + str(score))
            #print(kmeans.cluster_centers_)
            # TODO how to bench mark EM????
            #bench_kmeans(mixture.GaussianMixture(n_components=n, covariance_type=cv_type), 'EM', x_train, y_train)

        plt.figure(1)
        plt.plot(x, accuracy_y)
        plt.figure(2)
        plt.plot(x, train_time_y)
        plt.figure(3)
        plt.plot(x, test_time_y)
        legend.append('EM ' + cv_type)

    x_label = 'Number of Components/Clusters'
    plt.figure(1)
    plt.xlabel(x_label)
    plt.ylabel('Accuracy')
    plt.legend(legend, loc='upper right')

    plt.figure(2)
    plt.xlabel(x_label)
    plt.ylabel('Train Wall Time')
    plt.legend(legend, loc='upper left')

    plt.figure(3)
    plt.xlabel(x_label)
    plt.ylabel('Test Wall Time')
    plt.legend(legend, loc='upper center')

    plt.show()

# Get data from arff file.
dataset = arff.load(open('tic-tac-toe-split.arff', 'r'))
data = np.array(dataset['data'])
# 9 actual features * 3 binary classes for each feature = 27
feature_count = 27

# Excel workbook.
wb = Workbook()

# Format data. First 27 are features, 28th is label.
# For parts 1 and 2, we want to use all the data/target instead of train/test sets.
target = np.squeeze(data[:, feature_count:]).astype(int)
data = data[:, :feature_count]

# Randomly split data into 2/3 training set and 1/3 testing set.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# Run each homework part.
part1(data, target, x_train, x_test, y_train, y_test, wb)

# PCA
print('--- PCA ---')
x = []
noise_variance_y=[]
train_time_y = []
for n in range(1, 28):
    x.append(n)
    pca = PCA(n_components=n)
    start = time.time()   
    pca.fit(x_train)
    train_time = time.time() - start
    train_time_y.append(train_time)
    #print(pca.components_) 
    #print(pca.explained_variance_) 
    #print(pca.explained_variance_ratio_) 
    #print(pca.mean_)
    #print(pca.n_components_)
    noise_variance_y.append(pca.noise_variance_)


x_label = 'Number of Components/Clusters'
plt.figure(1)
plt.xlabel(x_label)
plt.ylabel('Noise Variance')
plt.plot(x, noise_variance_y)

plt.figure(2)
plt.xlabel(x_label)
plt.ylabel('Train Wall Time')
plt.plot(x, train_time_y)

plt.show()

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
