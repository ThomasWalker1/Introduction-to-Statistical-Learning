## Model Set Up

The model here takes the form
$$\begin{gather*}\log\left(\frac{P(G=1\vert X=x)}{P(G=K\vert X=x)}\right)=\beta_{10}+\beta_1^Tx\\\colon\\\log\left(\frac{P(G=K-1\vert X=x)}{P(G=K\vert X=x)}\right)=\beta_{(K-1)0}+\beta_{K-1}^Tx\end{gather*}$$
From this one has that
$$\begin{gather*}P(G=k\vert X=x)=\frac{\exp(\beta_{k0}+\beta_k^Tx)}{1+\sum_{l=1}^{K-1}\exp(\beta_{l0}+\beta_l^Tx)}\\P(G=K\vert X=x)=\frac{1}{1+\sum_{l=1}^{K-1}\exp(\beta_{l0}+\beta_l^Tx)}\end{gather*}$$
Let $\theta=\left\{\beta_{10},\beta_1,\dots,\beta_{(K-1)0},\beta_{K-1}^T\right\}$

## Fitting the Model

To do this we use the maximum likelihood. The log-likelihood of $N$ observations is given by
$$l(\theta)=\sum_{i=1}^N\log(p_{g_i}(x_i\vert\theta))$$
where $p_{k}(x_i\vert\theta)=P(G=k\vert X=x\vert\theta)$

### Newton's Algorithm
We can use Newton's algorithm to repeatedly estimate $\beta=\{\beta_{10},\beta_1\}$. 

Starting with a value $\beta^{\text{old}}$ we use the update rule
$$\beta^{\text{new}}\leftarrow\beta^{\text{old}}-\left(\frac{\partial^2 l(\beta)}{\partial\beta\partial\beta^T}\right)^{-1}\frac{\partial l(\beta)}{\partial\beta}$$
where the derivatives are evaluated at $\beta^{\text{old}}$
