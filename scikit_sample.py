import arff, numpy as np
from sklearn.model_selection import train_test_split
from homework_parts import part1, part2, part3, part4, part5

# Get data from arff file.
dataset = arff.load(open('tic-tac-toe-split.arff', 'r'))
data = np.array(dataset['data'])
# 9 actual features * 3 binary classes for each feature = 27
feature_count = 27

# Chosen after analyzing data from part 1.
special_k = [
    # KMeans & EM: had highest training and testing accuracy.
    2,
    # KMeans: had highest FMI and 2nd highest train/test accuracy.
    3,
    # EM: 3rd lowest AIC and BIC score under tied covariance.
    18,
    # KMeans: highest homogeneity/completeness/v-measure/ARI/AMI
    # EM: lowest AIC and BIC score under tied covariance.
    # EM: 2nd lowest AIC and BIC score under diag covariance.
    23,
    # EM: 2nd lowest AIC and BIC score under tied covariance.
    25,
    # KMeans: 2nd highest homogeneity/completeness/v-measure/ARI/AMI
    # EM: lowest AIC and BIC score under diag covariance.
    26,
    # KMeans: lowest inertia.
    # EM: highest score (log likelihood).
    27
    ]

# Format data. First 27 are features, 28th is label.
# For parts 1 and 2, we want to use all the data/target instead of train/test sets.
target = np.squeeze(data[:, feature_count:]).astype(int)
data = data[:, :feature_count]

# Randomly split data into 2/3 training set and 1/3 testing set.
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# Run each homework part.
#part1(data, target, x_train, y_train, x_test, y_test)
#part2(data, target, x_train, y_train, x_test, y_test, feature_count)
#part3(data, target, special_k, x_train, y_train, x_test, y_test)
#part4(data, target, special_k, x_train, y_train, x_test, y_test)
part5(data, target, special_k, x_train, y_train, x_test, y_test)
