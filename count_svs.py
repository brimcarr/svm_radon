import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Based on https://scikit-learn.org/stable/auto_examples/svm/plot_separating_
# hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

# Initialize lists of support vectors (sv's) & non-linearly separable (nls) samples.
nls=[]
supportvecs=[]
# Run the process 1000 times
for a in range(1000):
# Pick a random distribution of points around 2 centers
# "make_blobs create multiclass datasets by allocating each class one or more
# normally-distributed clusters of points. )
    X, y = make_blobs(n_samples=50, centers=2, center_box=(-20.0, 20.0), random_state=None)
# Fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=10000)
    clf.fit(X, y)
# Calculate the number of sv's
    svs=len(clf.support_vectors_)
# Add that number to the sv list.
    supportvecs.append(svs)
# If we get an unexpected result,
    if svs>3:
# Check if it's nls, if it's nls, add that to the nls list.
        if clf.score(X, y, sample_weight=None) < 1:
            nls.append(1)
        else:
# If it is linearly separable, then plot the function.
            print("support vecs:", clf.support_vectors_)
            plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# Plot the decision function.
            ax = plt.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            plt.axis('equal')
# Create grid to evaluate model.
            xx = np.linspace(xlim[0], xlim[1], 30)
            yy = np.linspace(ylim[0], ylim[1], 30)
            YY, XX = np.meshgrid(yy, xx)
            xy = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = clf.decision_function(xy).reshape(XX.shape)
# Plot decision boundary and margins.
            ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
# Plot sv's.
            ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                s=100, linewidth=1, facecolors='none', edgecolors='k')
            plt.show()
    else:
        pass
# Calculate the number of examples with 2 sv's, 3 sv's, nls, and others.
twos=supportvecs.count(2)
threes=supportvecs.count(3)
nonlinsep=len(nls)
others=len(supportvecs)-twos-threes-nonlinsep
# Display the results.
print("2 sv's:",twos,"\n","3 sv's:",threes,"\n","non-linearly separable:",
        nonlinsep,"\n","other:",others)
