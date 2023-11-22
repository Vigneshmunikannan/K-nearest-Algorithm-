# K-nearest-Algorithm-

# Algorithm
1. Select the number K of the neighbors
2. Calculate the Euclidean distance of K number of neighbors
3. Take the K nearest neighbors as per the calculated Euclidean distance.
4. Among these k neighbors, count the number of the data points in each category.
5. Assign the new data points to that category for which the number of the neighbor is
maximum.
6. Our model is ready.

# code 
import math
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, x):
        distances = [euclidean_distance(x, xi) for xi in self.X_train]
        sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k])
        k_nearest_labels = [self.y_train[i] for i in sorted_indices[:self.k]]
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common_label

def euclidean_distance(x1, x2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(x1, x2)))

#Example usage:
#Sample training data
X_train = [
    [5.1, 3.5],
    [4.9, 3.0],
    [6.7, 3.1],
    [6.0, 3.0],
    [5.7, 2.8],
    [5.8, 2.8],
    [6.2, 2.9],
    [6.7, 3.3],
]

y_train = ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']

#Create and train the k-NN classifier
knn_classifier = KNNClassifier(k=3)
knn_classifier.train(X_train, y_train)

#Sample test data
X_test = [
    [6.0, 3.4],
    [5.0, 3.0],
]

#Make predictions
for instance in X_test:
    prediction = knn_classifier.predict(instance)
    print(f"For instance {instance}, predicted class: {prediction}")


# Link to run if this copy paste is not working
https://replit.com/@vigneshm2021csb/KNN

# input 
X_test = [
    [6.0, 3.4],
    [5.0, 3.0],
]
# output
For instance [6.0, 3.4], predicted class: B
For instance [5.0, 3.0], predicted class: A
