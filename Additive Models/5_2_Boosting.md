## Boosting Methods

The idea here is to use the outputs of many weak classifiers to produce one powerful committee of classifiers. 

As an example consider the two-class classification problem, where the output is $Y\in\{-1,1\}$ and $X$ is the vector of predictor variables, with $G(X)$ being the classifier. Define the error rate on the training data to be
$$\overline{\text{err}}=\frac{1}{N}\sum_{i=1}^NI(y_i\neq G(x_i))$$
and the expected error rate to be
$$E_{XY}I(Y\neq G(X))$$
Let $G_m$ for $m=1,\dots, M$ be a weak classifier, a classifier that performs slightly better than random guessing. Use these classifiers to define the committee
$$G(x)=\text{sign}\left(\sum_{m=1}^M\alpha_mG_m(x)\right)$$
where the $\alpha_m$ are computed by the boosting algorithm. An example of such a boosting algorithm is AdaBoost.

**Algorithm (AdaBoost)**

Step 1:Initialize the observation weights $w_i=\frac{1}{N},i=1,\dots, N$

Step 2: For $m=1$ to $N$

Step 2.1: Fit classifier $G_m(x)$ to training data using weights $w_i$

Step 2.2:Compute

$$\text{err}_m=\frac{\sum_{i=1}^Nw_iI(y_i\neq G_m(x_i))}{\sum_{i=1}^Nw_i}$$
Step 2.3: compute $\alpha_m=\log((1-\text{err}_m)/\text{err}_m)$

Step 2.4: set $w_i\leftarrow w_i\cdot\exp(\alpha_m\cdot I(y_i\neq G_m(x_i))) \;i=1,\dots N$ 

Step 3: Output $G(x)=\text{sign}(\sum_{m=1}^M\alpha_m G_m(x))$

## Boosting Additive Models

Boosting aims to fit an additive expansion using a set of elementary basis functions. The expansion takes the form
$$f(x)=\sum_{m=1}^M\beta_m b(x;\gamma_m)$$
where 
- $\beta_m,m=1,\dots,M$ are the expansion coefficients
- $b(x;\gamma)\in\mathbb{R}$ are simple functions characterized by parameter $\gamma$

Typically additive models are fit by minimizing a loss function over the training data
$$\min_{\{\beta_m,\gamma_m\}_1^M}\sum_{i=1}^NL\left(y_i,\sum_{m=1}^M\beta_m b(x_i;\gamma)\right)$$
However, this can be numerically intensive.

### Forward Stagewise Additive Modelling

We can fit additive models using a boosting algorithm known as Forward Stagewise, which approximates the solution to the optimization problem by sequentially adding new basis functions to the expansion. The new function is fitted to the model, whilst the parameters of the previously added functions remain unchanged.

**Algorithm (Forward Stagewise)**

Step 1: Initialize $f_0(x)=0$

Step 2: For $m=1$ to $M$:

Step 2.1: Compute
$$(\beta_,\gamma_m)=\argmin_{\beta,\gamma}\sum_{i=1}^NL(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma))$$
Step 2.2 Set $f_m(x)=f_{m-1}(x)+\beta_m b(x;\gamma_m)$