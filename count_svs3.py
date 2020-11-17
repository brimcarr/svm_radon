from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# Based on https://scikit-learn.org/stable/auto_examples/svm/plot_separating_
# hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py

# Initialize lists of support vectors (sv's) & non-linearly separable (nls) samples.
nls=[]
supportvecs=[]
threeones=[]
twotwos=[]
n=20
# Run the process 1000 times
for a in range(1000):
# Pick a random distribution of points around 2 centers
# "make_blobs create multiclass datasets by allocating each class one or more
# normally-distributed clusters of points. )
    X, y = make_blobs(n_features=3, n_samples=n, centers=2, center_box=(-20.0, 20.0), random_state=None)
# Fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel='linear', C=10000)
    clf.fit(X, y)
# Calculate the number of sv's
    svs=len(clf.support_vectors_)
# Add that number to the sv list.
    supportvecs.append(svs)
# If we get an unexpected result,

    if svs>4:
# Check if it's nls, if it's nls, add that to the nls list.
        if clf.score(X, y, sample_weight=None) < 1:
            nls.append(1)
        else:
# If it is linearly separable, then plot the function.
            print("support vecs:", clf.support_vectors_)
# Computes the separating hyperplane
            z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
            tmp = np.linspace(-20,20,51)
            x,yy = np.meshgrid(tmp,tmp)
# Plots the points, the hyperplane, and the sv's.
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
            ax.plot_surface(x, yy, z(x,yy))
            ax.scatter(X[:, 0], X[:, 1], X[:,2], c=y, s=30, cmap=plt.cm.Paired)
            # Plot support vectors
            ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], clf.support_vectors_[:,2],
                c='black')
            plt.show()
# Classifies the cases with 4 support vectors as a 2-2 or 3-1 split.
    elif svs==4:
        if clf.n_support_[1]==2:
            twotwos.append(1)
        else:
            threeones.append(1)
    else:
        pass
# Calculate the number of examples with 2 sv's, 3 sv's, 4 sv's, nls, and others.
twos=supportvecs.count(2)
threes=supportvecs.count(3)
fours22=twotwos.count(1)
fours13=threeones.count(1)
nonlinsep=len(nls)
others=len(supportvecs)-twos-threes-fours22-fours13-nonlinsep
# Display the results.
print("2 sv's:",twos,"\n","3 sv's:",threes,"\n","4 sv's 2-2",fours22,"\n",
        "4 sv's 3-1",fours13,"\n","non-linearly separable:",nonlinsep,"\n","other:",others)
if others==0:
    pass
# A warning that the machine precision isn't high enough to distinguish which
# points are support vectors and which aren't. Can resolve by hand if needed.
else: print("Warning: Higher level of precision needed to resolve whether the",
                others,"other cases are 2, 3, or 4 support vector cases.")
