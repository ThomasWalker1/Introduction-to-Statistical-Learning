## Introduction

The aim here is to predict an output vector $Y$ from an input vector $X=(X_1,\dots,X_p)^T$. If we assume that $E(Y\vert X)$ is linear (or can at least be approximated linearly) in the inputs then the linear regression has the form
$$f(X)=\beta_0+\sum_{j=1}^pX_j\beta_j.$$
The $\beta_j$ are parameters that we can choose. The least squares methods chooses the $\beta_j$s such that they minimize the residual sum of squares,
$$\text{RSS}(\beta)=\sum_{i=1}^N(y_i-f(x_i))^2$$
where the tuples $(x_i,y_i)$ are our training data. Note each $x_i$ is a vector with value $(x_{i1},\dots,x_{ip})$

## Finding the parameters
Defining
$$\mathbf{X}=\begin{pmatrix}1&x_{11}&\dots&x_{1p}\\\colon&&&\colon\\1&x_{N1}&\dots&x_{Np}\end{pmatrix}$$
and 
$$\mathbf{Y}=\begin{pmatrix}y_1\\\colon\\y_N\end{pmatrix}$$
allows us to reformulate the least squares condition on $\beta=(\beta_0,\dots,\beta_{p})$ as
$$\text{RSS}(\beta)=(\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta)\geq\text{RSS}(\hat{\beta})\quad\forall\beta\in\mathbb{R}^{p+1}.$$
Solving this one obtains the unique (provided $\mathbf{X}$ is full rank) solution
$$\hat{\beta}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

## Geometrical Interpretation

Let $\mathbf{X}=(\mathbf{x}_0\vert\dots\vert\mathbf{x}_p)$ and $\hat{\mathbf{y}}=\mathbf{X}\hat{\beta}$, then as $\hat{\beta}$ minimizes $\Vert\mathbf{y}-\mathbf{X}\beta\Vert^2$ the vector $\hat{\mathbf{y}}$ is an orthogonal projection of $\mathbf{y}$ on to the span of the $\mathbf{x}_i$. Therefore, $\mathbf{H}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is called the projection matrix.

## Distributional Inferences

Assume $y_i$ are uncorrelated with a constant variance $\sigma^2$ and the $x_i$ are fixed. 

**Definition (Variance-Covariance Matrix)**
$$\text{Var}(\hat{\beta})=(\mathbf{X}^T\mathbf{X})^{-1}\sigma^2$$
which can be approximated by the unbiased estimator
$$\hat{\sigma}^2=\frac{1}{N-p-1}\sum_{i=1}^N(y_i-\hat{y}_i)^2$$

**Further Assumptions**:
- Assume the $f(X)=\beta_0+\sum_{j=1}^pX_j\beta_j$ is the correct model for $E(Y\vert X)$
- The deviation of $Y$ around its mean is Gaussian

$$Y=E(Y\vert X_1,\dots,X_p)+\epsilon\text{ where }\epsilon\sim N(0,\sigma^2)$$

From these assumptions
- $\hat{\beta}\sim N(\beta,(\mathbf{X}^T\mathbf{X})^{-1}\sigma^2)$
- $(N-p-1)\hat{\sigma}^2\sim\sigma^2\chi^2_{N-p-1}$
- $\hat{\beta}$ and $\hat{\sigma}^2$ are independent

## Hypothesis Testing

### Test if a parameter is $0$
$$H_0:\beta_j=0\quad H_1:\beta_j\neq 0$$
Define *Z-score*
$$z_j=\frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{v_j}}\text{ where }v_j=(\mathbf{X}^T\mathbf{X})^{-1}_{jj}$$
It can be shown that
$$z_j\sim t_{N-p-1}$$
when $\sigma^2$ is unknown and approximated by $\hat{\sigma}$. Therefore, $H_0$ is rejected for high absolute values of the *Z-score*

### Testing whether a collection of parameters are $0$

$$H_0:\left(\beta_{n(1)},\dots,\beta_{n(p_1)}\right)^T=\mathbf{0}\quad H_1:\left(\beta_{n(1)},\dots,\beta_{n(p_1)}\right)^T\neq\mathbf{0}$$

Use the $F$-statistics
$$F=\frac{\left(\text{RSS}_0-\text{RSS}_1\right)/(p_1-p_0)}{\text{RSS}_1/(N-p_1-1)}$$
- $\text{RSS}_0$: Residual sum-of-squares of fitted model excluding $\beta_{n(i)}$ (which has $p_0$+1 parameters)
- $\text{RSS}_1$: Residual sum-of-squares of fitted model including $\beta_{n(i)}$ (which has $p_1+1$) parameters

$$F\sim F_{p_1-p_0, N-p_1-1}$$

## Confidence Intervals

When use *Z-scores* the $1-\alpha$ confidence interval is given by
$$\left(\hat{\beta}_j-z^{1-\frac{\alpha}{2}}\sqrt{v_j}\hat{\sigma},\hat{\beta}_j+z^{1-\frac{\alpha}{2}}\sqrt{v_j}\hat{\sigma}\right)$$
where $z^{1-\frac{\alpha}{2}}$ is the $1-\frac{\alpha}{2}$ percentile of the normal distribution. 

The confidence set for the parameter vector $\beta$ is approximated by
$$C_{\beta}=\left\{\beta\big\vert\left(\hat{\beta}-\beta)^T\mathbf{X}^T\mathbf{X}\left(\hat{\beta}-\beta\right)\leq\hat{\sigma}(\chi^2_{p+1})^{(1-\frac{\alpha}{2})}\right)\right\}$$
where $(\chi^2_{p+1})^{(1-\frac{\alpha}{2})}$ is $1-\frac{\alpha}{2}$ percentile of $\chi_{p+1}^2$

**Gauss-Markov Theorem**
Suppose 
$$\mathbf{y}=\mathbf{X}\beta+\mathbf{\epsilon}$$
with the following assumptions:
- $E(\epsilon_i)=0$
- $\text{Var}(\epsilon_i)<\infty$
- $\text{Cov}(\epsilon_i,\epsilon_j)=0$ for all $i\neq j$

Then the least squares estimator for the parameters $\beta_j$ has the lowest sampling variance among all linear unbiased estimators.