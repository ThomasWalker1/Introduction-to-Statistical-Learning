Additive models is a more automatic flexible statistical method to characterize nonlinear effects. In case of regression a generalized additive model has the form
$$\mathbb{E}(Y\vert X_1,\dots, X_p)=\alpha+f_1(X_1)+\dots+f_p(X_p)$$
where:
- $X_1,\dots,X_p$ are the predictors
- $Y$ is the outcome
- The $f_j$s are smooth (nonparametric) functions

In the case of logistic regression (where we are performing two-class classification) the general additive model has the form
$$\log\left(\frac{\mu(X)}{1-\mu(X)}\right)=\alpha+f_1(X_1)+\dots+f_p(X_p)$$
where $\mu(X)=P(Y=1\vert X)$
The link function $g$ represents
$$g(\mu(X))=\alpha+f_1(X_1)+\dots+f_p(X_p)$$

The functions $f_j$ are estimated using a scatter plot smoother. 

## Fitting Additive Models
Let the additive model have the form 
$$Y=\alpha+\sum_{j=1}^pf_j(X_j)+\epsilon$$
where $\epsilon$ is an error term of mean $0$. For observations $x_i,y_i$ the penalized sum of squares criterion is given by
$$\text{PRSS}(\alpha,f_1,\dots,f_p)=\sum_{i=1}^N\left(y_i-\alpha-\sum_{j=1}^pf_j(x_{ij})\right)^2+\sum_{j=1}^p\lambda_j\int f_j^{\prime\prime}(t_j)^2dt_j$$
where $\lambda_j\geq0$ are tuning parameters. The solution of which has each of the $f_j$ are a cubic spline in $X_j$ with knots at the $x_{ij}, i=1,\dots, N$.

**Algorithm (Back-fitting Algorithm For Additive Models)**
1. Initialize $\hat{\alpha}=\frac{1}{N}\sum_{i=1}^Ny_i$ and $\hat{f}_j\equiv 0$ for all $i,j$
2. Cycle $j=1,2,\dots,p,\dots,1,2,\dots,p,\dots$ until the function $\hat{f}_j$ changes less than a specified threshold.

$$\begin{gather*}\hat{f}_j\leftarrow\mathcal{S}_j\left[\left\{y_i-\hat{\alpha}-\sum_{k\neq j}\hat{f}_k(x_{ik})\right\}_1^N\right]\\\hat{f}_j\leftarrow\hat{f}_j-\frac{1}{N}\sum_{i=1}^N\hat{f}_j(x_{ij})\end{gather*}$$
where $\mathcal{S}_j$ is the cubic smoothing spline applied to the targets in the set as a function of $x_{ij}$. $\hat{\alpha}$ is set to be $\text{ave}(y_i)$
