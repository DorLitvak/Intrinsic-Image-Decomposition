# Intrinsic Image Decomposition 

Bayesian Non-parametric Intrinsic Image Decomposition
We will work in the log domain where the log of the observed image, x, is as sumed to be generated from the sum of the log shading and the log reflectance
image.
X - the observed image
S - the shading image
R - the reflectance image
log(X) ∼ log(S) + log(R)

The different type of colors and shapes over the surface. We assume:\\ 
$log(R) = p(z|\pi)$ \\
$p(z|\pi) = \Pi_{i} p(z_{i}|\pi) = \Pi_{i} Cat(z_{i};\pi)$ \\
$p(\mu) = \Pi_{k} p(\mu_{k}) = N(\mu_{k};\theta,\Sigma^{\mu})$ \\
$p(\pi) = GEM(\pi; 1, \alpha)$ \\\\
\textbf{What is the shading?} \\\\
Is the type of hills and curves in the shape of the instance. \\
$log(S) = g$ \\
- The log shading image, denoted g, is generated from a zero-mean Gaussian process (GP) with a stationary co variance kernel, k. \\
- We model g as a 3D Gaussian process with a co-variance kernel that is a function of location and color. \\
- $p(g) = GP(g ; k) = N(g ; 0, \Sigma ^{g})$ \\
$\Sigma ^{g}$ denotes the finite-dimensional co-variance matrix obtained by evaluating the kernel, k, at the grid points. \\
The specific co-variance kernel parameters
govern the smoothness properties of g and are learned from training data. (l and sigma ?).\\\\
\textbf{Finally:} \\\\
Finally, we assume that the observed pixels in the log image are drawn independently from the following Gaussian distribution: \\\\
$p(x|\mu, z, g, \Sigma^{x}) = \Pi_{i} p(x_{i}|\mu, z_{i}, g_{i}, \Sigma^{x}) = \Pi_{i} N(x_{i} ; \mu_{z_{i}} + g_{i}, \Sigma_{x})$
