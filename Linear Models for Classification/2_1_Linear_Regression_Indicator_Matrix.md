Let $\mathcal{G}$ contain $K$ classes, each can be encoded by indicators $Y_k$ for $k=1,\dots,K$. Let $Y=(Y_1,\dots, Y_K)$. 

Suppose we have $N$ training instances that we record in an indicator response matrix, $\mathbf{Y}$, where the $i^{\text{th}}$ row of this matrix encodes the class of instance $i$. 

We can fit a linear regression model to this with fit
$$\mathbf{Y}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$
where $\mathbf{X}$ has $p+1$ columns, $p$ are for the inputs and $1$ is for the intercept. 

To perform classification of observation $x$:
1. Compute $\hat{f}(x)^T=(1,x)^T\hat{\mathbf{B}}$
2. Classify as $\hat{G}(x)=\argmax_{k\in\mathcal{G}}\hat{f}_k(x)$