Instead of selecting knots smoothing splines uses a maximal set of knots. For all two times continuously differentiable function $f(x)$ the aim is to identify the one that minimizes
$$\text{RSS}(f,\lambda)\sum_{i=1}^N\{y_i-f(x_i)\}^2+\lambda\int\{f^{\prime\prime}(t)\}^2dt$$
$\lambda$ is a fixed smoothing parameter:
- For small $\lambda$ $f$ can be any function the interpolates the data
- For large $\lambda$ the fit matches the least squares line fit.

There can be as many as $N$ notes for $f$, one at each of the unique $x_i$ for $i=1,\dots, N$

The solution to the above problem is a natural spline
$$f(x)=\sum_{j=1}^NN_j(x)\theta_j$$
where $N_j(x)$ is from an $N$-dimensional set of basis functions. Therefore, the problem can be reduced to
$$\text{RSS}(\theta,\lambda)=(\mathbf{y}-\mathbf{N}\theta)^T(\mathbf{y}-\mathbf{N}\theta)+\lambda\theta^T\mathbf{\Omega}_N\theta$$
where $\{\mathbf{N}\}_{ij}=N_j(x_i)$ and $\{\mathbf{\Omega_N}\}_{jk}=\int N^{\prime\prime}_j(t)N^{\prime\prime}_k(t)dt$
the solution to which is given by
$$\hat{\theta}=(\mathbf{N}^T\mathbf{N}+\lambda\mathbf{\Omega_N})^{-1}\mathbf{N}^T\mathbf{y}$$
resulting in the fitted smoothing spline
$$\hat{f}(x)=\sum_{j=1}^NN_j(x)\hat{\theta}_j$$