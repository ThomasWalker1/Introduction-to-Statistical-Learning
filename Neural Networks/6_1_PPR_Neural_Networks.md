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
Suppose we have training data $(x_i,y_i), i=1,\dots,N$. We aim to minimize
$$\sum_{i=1}^N\left[y_i-\sum_{m=1}^Mg_m(\omega_m^Tx_i)\right]^2$$
over functions $g_m$ and direction vectors $w_m$.

**Case M=1:**
1. Start with a direction vector $\omega$.
2. Estimate $g$ using $\omega$ through smoothing splines
3. Using a quasi-Newton method minimize the above expression of $\omega$ using the $g$ estimated in step 2.
4. Repeat step 2 using the $\omega$ from step 3 until you get convergence.

**Cade M>1:**

Build the model forward in a stage-wise manner, adding $(\omega_m, g_m)$ pairs at each stage.

## Neural Networks

A neural networks is a two-stage regression or classification model.

### $K$-Class Classification

