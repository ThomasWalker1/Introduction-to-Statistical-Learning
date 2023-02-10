These methods aims to reduce the complexity of the regressed coefficients in a less variable manner than subset selection. Subset selection is a discrete process whereas shrinkage methods are continuous. 

## Ridge Regression

In ridge regression we impose a penalty on the size of the coefficients. The modified residual sum of squares is given by
$$\hat{\beta}^{\text{ridge}}=\argmin_{\beta}\left\{\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2+\lambda\sum_{j=1}^p\beta_j^2\right\}$$

which can be reformulated as
$$\begin{gather*}\hat{\beta}^{\text{ridge}}=\argmin_{\beta}\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2\\\text{subject to}\sum_{j=1}^p\beta_j^2\leq t\end{gather*}$$
which highlights the specific constraint on size. 

Perform this optimization alleviates the problem where we have large coefficients of opposite sign for variables that is correlated with another. 

Usually, one centres the inputs. That is $x_{ij}$ is replaced by $x_{ij}-\bar{x}_j$. Then one estimates $\beta_0$ by $\bar{y}$. One can then use the ridge regression to find estimates for the other parameters.

Let $\mathbf{X}$ be the $N\times p$ input matrix (where centering of the inputs has been done). Then
$$\text{RSS}(\lambda)=(\mathbf{y}-\mathbf{X}\beta)^T(\mathbf{y}-\mathbf{X}\beta)+\lambda\beta^T\beta$$
and
$$\hat{\beta}^{\text{ridge}}=(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

## Lasso

The constraint for Lasso shrinkage is defined as
$$\begin{gather*}\hat{\beta}^{\text{lasso}}=\argmin_{\beta}\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2\\\text{subject to}\sum_{j=1}^p\vert\beta_j\vert\leq t\end{gather*}$$
Similar to above we re-parametrize the constant $\beta_0$ by standardizing the predictors, $\hat{\beta_0}=\bar{y}$. Then we fit the model without the intercept.

The different size penalty results in non-linear solutions in $y_i$. 