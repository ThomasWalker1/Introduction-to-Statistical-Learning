## Introduction

The aim here is to predict an output vector $Y$ from an input vector $X=(X_1,\dots,X_p)^T$. If we assume that $E(Y\vert X)$ is linear (or can at least be approximated linearly) in the inputs then the linear regression has the form
$$f(X)=\beta_0+\sum_{j=1}^pX_j\beta_j.$$
The $\beta_j$ are parameters that we can choose. The least squares methods chooses the $\beta_j$s such that they minimize the residual sum of squares,
$$\text{RSS}(\beta)=\sum_{i=1}^N(y_i-f(x_i))^2$$
where the tuples $(x_i,y_i)$ are our training data. Note each $x_i$ is a vector with value $(x_{i1},\dots,x_{ip})

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

To test whether a particular parameter is $0$ we form the standardized coefficient
$$z_j=\frac{\hat{\beta}_j}{\hat{\sigma}\sqrt{v_j}}\text{ where }v_j=(\mathbf{X}^T\mathbf{X})^{-1}_{jj}$$