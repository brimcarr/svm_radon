# svm_radon

This code is based off of an example from https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html#sphx-glr-auto-examples-svm-plot-separating-hyperplane-py. 
The program count_svs.py looks at n samples of points from two dimensional real space and classifies them into four different categories. 
The categories are linearly separable with 2 support vectors, linearly separable with 3 support vectors, linearly separable with more than 3 support vectors, and not linearly separable. 
For the cases with more than three support vectors, the program displays a plot of the points and prints the coordinates for the support vectors. 
These points can be evaluated for proper classification as a 2 support vector case or a 3 support vector case. 

The program count_svs3.py does something similar but in three dimensional real space. Due to the increase in dimension, count_svs3.py classifies each case into one of six categories. 
The categories are linearly separable with 2 support vectors, linearly separable with 3 support vectors, linearly separable with 4 support vectors (2-2 split), linearly separable with 4 support vectors (3-1 split), linearly separable with more than 4 support vectors, and not linearly separable. For cases with more than 4 support vectors, the program prints out the coordinates of the support vectors and a 3-d plot of the case so that the points can be classified correctly.

See paper: https://arxiv.org/abs/2011.00617
