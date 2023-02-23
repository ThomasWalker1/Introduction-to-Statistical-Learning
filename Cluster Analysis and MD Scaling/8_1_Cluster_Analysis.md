The goal here is to group a collection of objects into subsets, such that the the objects within the subsets are closely related in some way. Furthermore, these methods can be used to impose a hierarchy to the subsets.

### Proximity Matrices

Sometimes data is structured with a notion of proximity between pairs of objects (using dissimilarity between objects as the "measure"). This data can be represented as a $N\times N$ matrix $\mathbf{D}$, where $N$ is the number of data points and $d_{ij}$ records the proximity between object $i$ and $j$. 
- $\mathbf{D}$ is assumed to have non-negative entries
- $\mathbf{D}$ has $0$ along the diagonal
- $\mathbf{D}$ is assumed to be symmetric

### Attributes

Suppose we have measurements $x_{ij}$ for $i=1,\dots,N$ and $j=1,\dots, p$. The $j$s index the attributes. If Dissimilarity between objects in attribute $j$ is given $d_{j}(x_{ij},x_{i^{\prime}j})$ then the dissimilarity between objects $x_i$ and $x_{i^{\prime}}$ is given by
$$D(x_i,x_{i^{\prime}})=\sum_{j=1}^pd_j(x_{ij},x_{i^{\prime}{j}})$$

**Quantitative Variables:**

$$d(x_i,x_{i^{\prime}})=l\left(\vert x_i-x_i^{\prime}\vert\right)$$
where $l$ is some loss function (usually squared-error).

**Ordinal Variables:**

For variables with $M$ different values the variables are generally redefined as
$$\frac{i=\frac{1}{2}}{M},\;i=1,\dots, M$$
and then treated as quantitative variables.

**Categorical Variables:**

Here the degree-of-difference between pairs of values must be defined explicitly. If there are $M$ categories one encodes the differences in a symmetric $M\times M$ matrix, $L$, where $L_{rr^{\prime}}=L_{r^{\prime}r}\geq 0$ and $L_{rr}=0$.