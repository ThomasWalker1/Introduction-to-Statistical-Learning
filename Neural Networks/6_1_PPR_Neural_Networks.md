## Projection Pursuit Regression

- $X$ be an input vector with $p$ component
- $Y$ be target vector
- $\omega_m, m=1,2,\dots, M$, unit $p$-vectors of unknown parameters. 

The Projection Pursuit Regression (PPR) model has form
$$f(X)=\sum_{m=1}^Mg_m(\omega_m^TX)$$

$g_m(\omega_m^TX)\in\mathbb{R}^p$ is the ridge function, and varies only in directions $\omega_m$. 

**Universal Approximation Theorem**
We can approximate any continuous function in $\mathbb{R}^p$ using the PPR model with $M\to\infty$.

### Fitting PPR Model