A hyperplane (or affine set) $L$ is defined by the equation
$$f(x)=\beta_0+\beta^Tx=0$$
1. For any two points $x_1,x_2\in L$, $\beta^T(x_1-x_2)=0$ and so $\beta^*=\frac{\beta}{\Vert\beta\Vert}$ is vector normal to $L$
2. For $x_0\in L$, $\beta^Tx_0=-\beta_0$
3. The signed distance of any point $x$ to $L$ is
$$\beta^{*T}(x-x_0)=\frac{f(x)}{\Vert f^{\prime}(x)\Vert}$$

## Rosenblatt's Perceptron Learning Algorithm

This algorithm tries to find a separating hyperplane by minimizing distance of misclassified points to the decision boundary.
$$D(\beta,\beta_0)=-\sum_{i\in\mathcal{M}}y_i(x_i^T\beta+\beta_0)$$
where $\mathcal{M}$ contains the indexes of misclassified points. The decision boundary is defined by $\beta^Tx+\beta_0=0$. One can compute the gradients
$$\begin{gather*}\frac{\partial D(\beta,\beta_0)}{\partial\beta}=-\sum_{i\in\mathcal{M}}y_ix_i\\\frac{\partial D(\beta,\beta_0)}{\partial\beta_0}=-\sum_{i\in\mathcal{M}}y_i\end{gather*}$$
Using this we can apply stochastic gradient descent, taking steps in negative gradient direction after each observation is visited. 
$$\begin{pmatrix}\beta\\\beta_0\end{pmatrix}\leftarrow\begin{pmatrix}\beta\\\beta_0\end{pmatrix}+\rho\begin{pmatrix}y_ix_i\\y_i\end{pmatrix}\text{ where }\rho\text{ is the learning rate}$$

### Problems
1. Solution found is dependent on the starting values
2. Number of steps required for convergence may be large
3. If the data is not separable the algorithm may get stuck in a loop

## Optimal Separating Hyperplanes

This method separates two classes and maximizes the distance to the closet point from either class. Providing a unique solution to the separating hyperplane problem. Consider,
$$\max_{\beta,\beta_0,\Vert\beta\Vert=1}M\text{ subject to }y_i(x_i^T\beta+\beta_0)\geq M, i=1,\dots,N$$

Upon setting $\Vert\beta\Vert=\frac{1}{M}$ the above problem can be reformulated as
$$\max_{\beta,\beta_0}\frac{1}{2}\Vert\beta\Vert^2\text{ subject to }y_i(x_i^T\beta+\beta_0)\geq 1, i=1,\dots,N$$

There is a margin of empty space around the decision boundary, with thickness $\frac{1}{\Vert\beta\Vert}$, we want to maximize this thickness.

The optimal separating hyperplane produces a function $\hat{f}(x)=x^T\hat{\beta}+\hat{\beta}_0$ and we classify new observations as
$$\hat{G}(x)=\text{sign}\hat{f}(x)$$

If the data is not separable then no feasible solution to this problem will be found. 
