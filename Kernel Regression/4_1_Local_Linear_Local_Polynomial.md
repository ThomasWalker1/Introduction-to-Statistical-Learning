Kernel regression methods introduce flexibility into the model by fitting simple models at each query point separately. Points close to a target point, $x_0$, are used to fit a model. Localization is achieved using a kernel $K_{\lambda}(x_0,x_i)$, a weighting function that assigns weights to $x_i$ based on their distance to $x_0$. 

## Local Linear Regression

At each target point, $x_0$, the following weighted least squares problem is solved
$$\min_{\alpha(x_0),\beta(x_0)}\sum_{i=1}^NK_{\lambda}(x_0,x_i)[y_i-\alpha(x_0)-\beta(x_0)x_i]^2$$
The resulting fitted model at $x_0$ is given by
$$\hat{f}(x_0)=\hat{\alpha}(x_0)+\hat{\beta}(x_0)$$

## Local Polynomial Regression

Similarly, we can fit locally using polynomials of degree $d$, by solving the constraint problem
$$\min_{\begin{gather*}\alpha(x_0),\beta_{j}(x_0)\\j=1,\dots,d\end{gather*}}\sum_{i=1}^NK_{\lambda}(x_0,x_i)\left[y_i-\alpha(x_0)-\sum_{j=1}^d\beta_j(x_0)x_i^j\right]^2$$
resulting in the fitted model at $x_0$
$$\hat{f}(x_0)=\hat{\alpha}(x_0)+\sum_{j=1}^d\hat{\beta}_j(x_0)x_0^j$$
Fitting polynomials corrects the bias of fitting linear models at points of true curvature of the underlying function. However, there is a increased variance as a result of this reduction in bias. 

## Comparison between local linear and local polynomials
- Local linear fits reduce bias at boundary with low variance cost
- Quadratic fits also reduce bias at boundaries but increase the variance
- Quadratic fits reduce bias at points of curvature at the interior of the domain
- Asymptotically odd degree polynomials dominate even degree fits