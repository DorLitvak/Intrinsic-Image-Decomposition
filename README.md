# Intrinsic Image Decomposition 

This repository is an extanded implementation of [1]. 
We used hirarchical DPMM in order to cluster the reflectance over multiple images. 
And later extanded the GP to Markov Random Field for faster preformances. 

Some mathematical explanation about the repo:

We will work in the log domain where the log of the observed image, x, is as sumed to be generated from the sum of the log shading and the log reflectance
image.

X - the observed image

S - the shading image

R - the reflectance image

log(X) ∼ log(S) + log(R)

The different type of colors and shapes over the surface. 
We assume:

$log(R) = p(z|\pi)$

$p(z|\pi) = \Pi_{i} p(z_{i}|\pi) = \Pi_{i} Cat(z_{i};\pi)$ 

$p(\mu) = \Pi_{k} p(\mu_{k}) = N(\mu_{k};\theta,\Sigma^{\mu})$ 

$p(\pi) = GEM(\pi; 1, \alpha)$

What is the shading?

Is the type of hills and curves in the shape of the instance. 
$log(S) = g$ 

- The log shading image, denoted g, is generated from a zero-mean Gaussian process (GP) with a stationary co variance kernel, k. 
- We model g as a 3D Gaussian process with a co-variance kernel that is a function of location and color. 
- $p(g) = GP(g ; k) = N(g ; 0, \Sigma ^{g})$ 

$\Sigma ^{g}$ denotes the finite-dimensional co-variance matrix obtained by evaluating the kernel, k, at the grid points. 
The specific co-variance kernel parameters
govern the smoothness properties of g and are learned from training data. (l and sigma ?).

Finally:
Finally, we assume that the observed pixels in the log image are drawn independently from the following Gaussian distribution:

$p(x|\mu, z, g, \Sigma^{x}) = \Pi_{i} p(x_{i}|\mu, z_{i}, g_{i}, \Sigma^{x}) = \Pi_{i} N(x_{i} ; \mu_{z_{i}} + g_{i}, \Sigma_{x})$


To covert to MRF version:

Assume a small image I as described as a vector: 

$I_{vec}^{T} = [X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11, X_12]$

Mark the two dimensions of I as $size_{1} = 4$, $size_{2} = 3$. The neighborhood of 3X3 surrounding for each pixel in the I image are defined using the next following questions. Summing them would give us punishment over pixels that are far from one another (far in color sense).

Right Neighbors: 
$\sum_{i=1}^{size_{1}*(size_{2}-1)} (X_{i} - X_{i+size_{1}})^{2}$

Left Neighbors: 
$\sum_{i=1}^{size_{1}*(size_{2}-1)} (X_{i+size_{1}} - X_{i})^{2}$

Up Neighbors: 
$\sum_{j=0}^{size_{2}-1} \sum_{i=2}^{size_{1}} (X_{i-1+(j*size_{1})} - X_{i+(j*size_{1})})^{2}$

Down Neighbors:
$\sum_{j=0}^{size_{2}-1} \sum_{i=2}^{size_{1}} (X_{i+(j*size_{1})} - X_{i-1+(j*size_{1})})^{2}$

Up Left Diagonal:
$\sum_{j=0}^{size_{2}-2} \sum_{i=size_{1}+2}^{2*size_{1}} (X_{i+(j*size_{1})} - X_{i-1+((j-1)*size_{1})})^{2}$

Down right Diagonal:
$\sum_{j=0}^{size_{2}-2} \sum_{i=size_{1}+2}^{2*size_{1}} (X_{i-1+((j-1)*size_{1})} - X_{i+(j*size_{1})})^{2}$

Up Right Diagonal:
$\sum_{j=0}^{size_{2}-2} \sum_{i=2}^{size_{1}} (X_{i+(j*size_{1})} - X_{i-1+((j+1)*size_{1})})^{2} $

Down Left Diagonal:
$\sum_{j=0}^{size_{2}-2} \sum_{i=2}^{size_{1}} (X_{i-1+((j+1)*size_{1})} - X_{i+(j*size_{1})})^{2}$


[1] Chang, Jason, Randi Cabezas, and John W. Fisher. "Bayesian nonparametric intrinsic image decomposition." European conference on computer vision. Springer, Cham, 2014.‏
