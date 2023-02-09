Let $f_k(x)$ be the class-conditional density of $X$ in class $G=k$. Let $\pi_k$ be prior probability of class $k$. By Bayes

$$P(G=k\vert X=x)=\frac{f_k(x)\pi_k}{\sum_{l=1}^Kf_l(x)\pi_l}$$

Let each class density be model by Gaussian
$$f_k(x)=\frac{1}{(2\pi)^{p/2}\vert\mathbf{\Sigma}_k\vert^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_k)^T\mathbf{\Sigma}_k^{-1}(x-\mu_k)\right)$$

## Linear Discriminant Analysis (LDA)

LDA deals with case where $\mathbf{\Sigma}_k=\mathbf{\Sigma}$ for all $k$. In this case we get that decision boundaries between classes are linear, therefore, classes are separated by hyperplanes in $\mathbb{R}^p$. 

The linear discriminant functions are
$$\delta_k(x)=x^T\mathbf{\Sigma}^{-1}\mu_k-\frac{1}{2}\mu_k^T\mathbf{\Sigma}^{-1}\mu_k+\log(\pi_k)$$
and $G(x)=\argmax_k\delta_k(x)$

In practice the parameters of $f_k$ are unknown but can be estimated by:
- $\hat{\pi}_k=\frac{N_k}{N}$, $N_k$ is number of class $k$ observations
- $\hat{\mu}_k=\sum_{g_i=k}\frac{x_i}{N_k}$
- $\hat{\mathbf{\Sigma}}=\sum_{k=1}^K\sum_{g_i=k}\frac{(x_i-\hat{\mu}_k)(x_i-\hat{\mu}_k)^T}{N-k}$

LDA requires a total of $(K-1)\times(p+1)$ parameters.

## Quadratic Discriminant Analysis (QDA)

QDA considers the case where the $\mathbf{\Sigma}_k$ are not equal. In this case we get quadratic discriminant functions
$$\delta_k(x)=-\frac{1}{2}\log\vert\mathbf{\Sigma}_k\vert-\frac{1}{2}(x-\mu_k)^T\mathbf{\Sigma}_k^{-1}(x-\mu_k)+\log\pi_k$$

QDA requires $(K-1)\times\left(\frac{p(p+3)}{2}+1\right)$ parameters

## Regularized Discriminant Analysis

This is a compromise method between LDA and QDA. The covariance matrix have the form
$$\hat{\mathbf{\Sigma}}(\alpha)=\alpha\hat{\mathbf{\Sigma}}+(1-\alpha)\hat{\mathbf{\Sigma}}$$
Where $\hat{\mathbf{\Sigma}}$ is the pooled covariance matrix. Often cross-validation is done to determine $\alpha\in[0,1]$

We can further generalize the above setting by letting $\hat{\mathbf{\Sigma}}$ be parameterised:
$$\hat{\mathbf{\Sigma}}(\gamma)=\gamma\hat{\mathbf{\Sigma}}+(1-\gamma)\hat{\sigma}^2\mathbf{I}\text{ for }\gamma\in[0,1]$$