These methods aims to reduce the complexity of the regressed coefficients in a less variable manner than subset selection. Subset selection is a discrete process whereas shrinkage methods are continuous. 

## Ridge Regression

In ridge regression we impose a penalty on the size of the coefficients. The modified residual sum of squares is given by
$$\hat{\beta}^{\text{ridge}}=\argmin_{\beta}\left\{\sum_{i=1}^N\left(y_i-\beta_0-\sum_{j=1}^px_{ij}\beta_j\right)^2+\lambda\sum_{j=1}^p\beta_j^2\right\}$$

## Lasso