import arff, numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans
import time
from homework_parts import part1

# Get data from arff file.
dataset = arff.load(open('tic-tac-toe-split.arff', 'r'))
data = np.array(dataset['data'])
# 9 actual features * 3 binary classes for each feature = 27
feature_count = 27

# Format data. First 27 are features, 28th is label.
# For parts 1 and 2, we want to use all the data/target instead of train/test sets.
target = np.squeeze(data[:, feature_count:]).astype(int)
data = data[:, :feature_count]

# Randomly split data into 2/3 training set and 1/3 testing set.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# Run each homework part.
part1(data, target, x_train, x_test, y_train, y_test)
#part2(data, target, x_train, x_test, y_train, y_test)

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
