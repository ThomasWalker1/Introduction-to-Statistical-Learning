Start with observations $x_1,\dots, x_N\in\mathbb{R}^p$, let $d_{ij}$ denote the distance between observations $i$ and $j$ (note this doesn't have to be the usual Euclidean distance).

Multidimensional scaling seeks to find values $z_1,\dots,z_N\in\mathbb{R}^k$ that minimizes the stress function
$$S_M(z_1,\dots,z_N)=\sum_{i\neq i^{\prime}}\left(d_{ii^{\prime}}-\Vert z_i-z_{i^{\prime}}\Vert\right)^2$$
called the least-squares scaling

Gradient descent can be used to minimize $S_M$.

A scaling that doesn't rely on a metric is the Shephard-Kruskal non-metric scaling. In this case one minimizes 
$$S_{NM}(z_1,\dots, z_n)=\frac{\sum_{i\neq i^{\prime}}\left(\Vert z_i-z_{i^{\prime}}\Vert-\theta(d_{ii^{\prime}})\right)^2}{\sum_{i\neq i^{\prime}}\Vert z_i-z_{i^{\prime}}\Vert^2}$$
where $\theta$ is a fixed increasing function and we minimize over $z_i$ by gradient descent. Then with $z_i$ fixed one can find the best monotonic approximation $\theta(d_{ii^{\prime}})$ and repeat until the solutions stabilize.

Multidimensional scaling represents high-dimensional data in a low-dimensional space while trying to preserve pairwise distances.