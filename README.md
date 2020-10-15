# svm_radon

This code is based off of an example from https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py. 
This program looks at n samples of points from two dimensional real space and classifies them into four different categories. 
The categories are linearly separable with 2 support vectors, linearly separable with 3 support vectors, linearly separable with more than 3 support vectors, and not linearly separable. 
For the cases with more than three support vectors, the program displays a plot of the points and prints the coordinates for the support vectors. 
These points can be evaluated for proper classification as a 2 support vector case or a 3 support vector case. 
See paper: 
