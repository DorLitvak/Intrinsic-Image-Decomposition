# Intrinsic Image Decomposition 

This repository is an expanded implementation of [1]. We used a hierarchical DPMM to cluster the reflectance over multiple images. Later, we expanded the GP to a Markov Random Field for faster performance.

Some mathematical explanation about the repo:

We will work in the log domain, where the log of the observed image (x) is assumed to be generated from the sum of the log shading (S) and the log reflectance image (R).

log(X) ∼ log(S) + log(R)

We assume:

$log(R) = p(z|\pi)$

$p(z|\pi) = \Pi_{i} p(z_{i}|\pi) = \Pi_{i} Cat(z_{i};\pi)$ 

$p(\mu) = \Pi_{k} p(\mu_{k}) = N(\mu_{k};\theta,\Sigma^{\mu})$ 

$p(\pi) = GEM(\pi; 1, \alpha)$

What is shading?
The type of hills and curves that make up the shape of an object.

$log(S) = g$ 

- The log shading image, denoted as g, is generated from a zero-mean Gaussian process (GP) with a stationary covariance kernel, k.
- We model g as a 3D Gaussian process with a covariance kernel that is a function of location and color. 
- $p(g) = GP(g ; k) = N(g ; 0, \Sigma ^{g})$ 

$\Sigma ^{g}$ denotes a finite-dimensional co-variance matrix obtained by evaluating the kernel, k, at the grid points. 
The specific covariance kernel parameters govern the smoothness properties of g, and are learned from training data (l and sigma).

Finally:
Finally, we assume that the observed pixels in the log image are independently drawn from the following Gaussian distribution:

$p(x|\mu, z, g, \Sigma^{x}) = \Pi_{i} p(x_{i}|\mu, z_{i}, g_{i}, \Sigma^{x}) = \Pi_{i} N(x_{i} ; \mu_{z_{i}} + g_{i}, \Sigma_{x})$


[1] Chang, Jason, Randi Cabezas, and John W. Fisher. "Bayesian nonparametric intrinsic image decomposition." European conference on computer vision. Springer, Cham, 2014.‏
