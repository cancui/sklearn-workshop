import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")

# Fabricate data for classification where X is the feature array and y is the label array
X, y = make_classification(400, 2, 2, 0, weights=[.5, .5], random_state=15)

# Standardize the data for better classifier performance
X = scale(X)

# Split available data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create a logstic regression object, which contains functions for fitting and predicting,
# as well as the model parameters that are learned
logistic_regression = LogisticRegression()

# Train the model on the training data
logistic_regression.fit(X_train, y_train)

# Visualize the resulting decision boundary
xx, yy = np.mgrid[-5:5:.01, -5:5:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = logistic_regression.predict_proba(grid)[:, 1].reshape(xx.shape)
plt.figure()
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
plt.xlim((-3,3))
plt.ylim((-3,3))
plt.xlabel('Feature A')
plt.ylabel('Feature B')
plt.figure()
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=50, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white", linewidth=1)
plt.xlim((-3,3))
plt.ylim((-3,3))
plt.xlabel('Feature A')
plt.ylabel('Feature B')

# Test classifier accuracy
predictions = logistic_regression.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1_score = f1_score(y_test, predictions)
print('Model achieved accuracy:{:.2f} and f1-score:{:.2f}'.format(accuracy, f1_score))

plt.show()