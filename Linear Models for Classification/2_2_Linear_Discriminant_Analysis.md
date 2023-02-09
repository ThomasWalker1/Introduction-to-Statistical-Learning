Let $f_k(x)$ be the class-conditional density of $X$ in class $G=k$. Let $\pi_k$ be prior probability of class $k$. By Bayes

$$P(G=k\vert X=x)=\frac{f_k(x)\pi_k}{\sum_{l=1}^Kf_l(x)\pi_l}$$

Let each class density be model by Gaussian
$$f_k(x)=\frac{1}{(2\pi)^{p/2}\vert\mathbf{\Sigma}_k\vert^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_k)^T\mathbf{\Sigma}_k^{-1}(x-\mu_k)\right)$$

LDA deals with case where $\mathbf{\Sigma}_k=\mathbf{\Sigma}$ for all $k$. In this case we get that decision boundaries between classes are linear, therefore, classes are separated by hyperplanes in $\mathbb{R}^p$. 

The linear discriminant functions are
$$\delta_k(x)=x^T\mathbf{\Sigma}^{-1}\mu_k-\frac{1}{2}\mu_k^T\mathbf{\Sigma}^{-1}\mu_k+\log(\pi_k)$$
and $G(x)=\argmax_k\delta_k(x)$