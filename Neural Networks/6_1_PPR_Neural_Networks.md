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

The set up is as follows. 
- Inputs, $X$
- Derived features $Z_m, m=1,2,\dots M$
- Target measurements $T_k, k=1,2,\dots, K$

The targets are modelled as linear combinations of the features, the features are modelled as linear combinations of the inputs.

$$\begin{gather*}Z_m=\sigma(\alpha_{0m}+\alpha_m^TX),\;m=1,\dots,M\\T_k=\beta_{0k}+\beta_k^TZ,\;k=1,\dots,K\\f_k(X)=g_k(T),\;k=1,\dots, K\end{gather*}$$
$\sigma(v)$ is a nonlinear activation function. The introduction of nonlinearity allows for a more detailed representation of the data to be obtained.  

$g_k(T)$ is an output function allowing for a transformation of the output vector $T$. Typically a softmax function
$$g_k(T)=\frac{e^{T_k}}{\sum_{l=1}^Ke^{T_l}}$$
is used so that the vector $T$ represents a probability distribution.

The $Z_m$ are often called hidden units.  Can be thought of a a basis expansions of the inputs $X$. 

### Fitting Neural Networks

The model outlined above has a set of parameters $\theta$ 
$$\begin{gather*}\{\alpha_{0m},\alpha_m:m=1,\dots, M\}\quad K(p+1)\text{ weights}\\\{\beta_{0k},\beta_{k}:k=1,\dots,K\}\quad K(M+1)\text{ weights}\end{gather*}$$

In the case of regression, to measure the fit of a set of parameters we can use sum-of-squares
$$R(\theta)=\sum_{k=1}^K\sum_{i=1}^N(y_{ik}-f_k(x_i))^2$$

In the case of classification, to measure the fit of a set of parameters we can use cross entropy:
$$R(\theta)=-\sum_{i=1}^N\sum_{k=1}^Ky_{ik}\log(f_k(x_i))$$
with the corresponding classifier being
$$G(x)=\argmax_kf_k(x)$$

Often the global minimize for each $R(\theta)$ may overfit the data.

To minimize $R(\theta)$ we use gradient descent, in particular we use back-propagation. 

**Back-propagation:**

1. Calculate derivatives
$$\frac{\partial R_i}{\partial\beta_{km}},\;\frac{\partial R_i}{\partial\alpha_{ml}}$$
2. Use the update rules
$$\begin{gather*}\beta_{km}^{(r+1)}=\beta_{km}^{(r)}-\gamma_r\sum_{i-1}^N\frac{\partial R_i}{\partial\beta^{(r)}_{km}}\\\alpha_{ml}^{(r+1)}=\alpha_{ml}^{(r)}-\gamma_r\sum_{i-1}^N\frac{\partial R_i}{\partial\alpha^{(r)}_{ml}}\end{gather*}$$
where $\gamma_r$ is the learning rate.

3. In the forward pass of the algorithm fix the parameters and calculate the predicted values $\hat{f}_k(x_i)$
4. In the backward pass calculate the gradients and use the update rules to update the parameters.

Batch learning is when the parameter updates occur as the sum over all of the training cases.


A training epoch refers to a full sweep through the training set.

$\gamma_r$ can be fixed or it can be varied. If $\gamma_r\to 0, $\sum_r\gamma_r=\infty$ and $\sum_r\gamma_r^2\leq\infty$, then convergence of back-propagation is guaranteed.
