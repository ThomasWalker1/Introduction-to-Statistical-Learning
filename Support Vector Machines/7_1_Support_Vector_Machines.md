## Support Vector Classifier

Suppose the training set consists $(x_1,y_1),\dots,(x_N,y_N)$ with $x_i\in\mathbb{R}^p$ and $y_i\in\{-1,1\}$. For the hyperplane
$$\{x:f(x)=x^T\beta+\beta_0\}$$
define the classification rule
$$G(x)=\text{sign}\left[x^T\beta+\beta_0\right]$$
For the optimization problem we introduce $\xi=(\xi_1,\dots,\xi_N)$ and define the constraints to be
$$\begin{gather*}\max_{\beta,\beta_0,\Vert\beta\Vert=1}M\text{ subject to }\\y_i(x_i^T\beta+\beta_0)\geq M(1-\xi_i),i=1,\dots, N\\\text{and }\xi_i\geq0\;\forall i, \sum_{i=1}^N\xi_i\leq K\end{gather*}$$
This allows there to be some overlap to deal with the case where the data is not separable. 

Dropping norm constraint on $\beta$ by defining $M=\frac{1}{\Vert\beta\Vert}$ leads to the equivalent optimization problem
$$\min(\Vert\beta\Vert)\text{ subject to }\begin{cases}y_i(x_i^T\beta+\beta_0)\geq 1-\xi_i\;\forall i\\\xi\geq0,\sum_{i=1}^N\xi_i\leq K\end{cases}$$

One notes that points further within the class region play less significant role in the shaping of the class boundaries. 

## Support Vector Machine

The idea here is to generalize from linear boundaries to nonlinear boundaries. We chose a set of basis functions, $h_m(x)$ for $m=1,\dots, M$ and consider the same constraint problem as above but with the input features
$$h(x_i)=(h_1(x_i),\dots,h_M(x_i)), i=1,\dots N$$
to produce the function and classifier
$$h(x)^T\hat{\beta}+\hat{\beta}_0,\;\hat{G}(x)=\text{sign}\left(\hat{f}(x)\right)$$

