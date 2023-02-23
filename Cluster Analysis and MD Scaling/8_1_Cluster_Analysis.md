The goal here is to group a collection of objects into subsets, such that the the objects within the subsets are closely related in some way. Furthermore, these methods can be used to impose a hierarchy to the subsets.

### Proximity Matrices

Sometimes data is structured with a notion of proximity between pairs of objects (using dissimilarity between objects as the "measure"). This data can be represented as a $N\times N$ matrix $\mathbf{D}$, where $N$ is the number of data points and $d_{ij}$ records the proximity between object $i$ and $j$. 
- $\mathbf{D}$ is assumed to have non-negative entries
- $\mathbf{D}$ has $0$ along the diagonal
- $\mathbf{D}$ is assumed to be symmetric

### Attributes

Suppose we have measurements $x_{ij}$ for $i=1,\dots,N$ and $j=1,\dots, p$. The $j$s index the attributes. If dissimilarity between objects in attribute $j$ is given $d_{j}(x_{ij},x_{i^{\prime}j})$ then the dissimilarity between objects $x_i$ and $x_{i^{\prime}}$ is given by
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

The combination of dissimilarity $D(\cdot,\cdot)$ defined above can instead be made into a weighted sum
$$D(x_i,x_{i^{\prime}})=\sum_{j=1}^pw_j\cdot d_j(x_{ij},x_{i^{\prime}j}),\quad\sum_{j=1}^pw_j=1$$
to reflect the relative influence of each attribute in the overall dissimilarity between objects.

The influence of attribute $j$ on object dissimilarity $D(\cdot,\cdot)$ depends on its relative contribution to the average dissimilarity measure across all pairs in the data set
$$\bar{D}=\frac{1}{N^2}\sum_{i=1}^N\sum_{i^{\prime}=1}^ND(x_i,x_{i^{\prime}})=\sum_{j=1}^pw_j\cdot\bar{d}_j\\\text{ where }\bar{d}_j=\frac{1}{N^2}\sum_{i=1}^N\sum_{i^{\prime}=1}^Nd_j(x_{ij},x_{i^{\prime}j})$$

## Clustering Algorithms

**Combinatorial Algorithms:** Works directly on the data.

**Mixture Modeling:** Supposes data is an i.i.d sample. The density function of which is a parameterized mixture of component density functions

**Mode Seekers:** Nonparametric approach, directly estimating the pdf.

### Combinatorial Algorithms

Each observation is labeled uniquely $i\in\{1,\dots, N\}$. A pre-specified number of cluster $K<N$ is postulated, labeled $k\in\{1,\dots, K\}$. Observations are then assigned to clusters using a one-to-one mapping. The encoder $k=C(i)$ specifies this mapping. One aims to find a $C$ such that a "loss" function is minimized.

A natural loss used in practice is 
$$W(C)=\frac{1}{2}\sum_{k=1}^K\sum_{C(i)=k}\sum_{C(i^{\prime})=k}d(x_i,x_{i^{\prime}})$$
this characterizes the closeness between objects within a cluster. Which is equivalent to maximizing the distance between clusters
$$B(C)=\frac{1}{2}\sum_{k=1}^K\sum_{C(i)=k}\sum_{C(i^{\prime})\neq k}d(x_i,x_{i^{\prime}})$$

Exploring all possible $C$ to optimize the above is unfeasible for large $N,K$, therefore, in practice iterative greedy descent algorithms are used to determine optimal partitions. 

### K-means
In this setting all are variables are quantitative and 
$$d(x_i,x_{i^{\prime}})=\sum_{j=1}^p(x_{ij}-x_{i^{\prime}j})^2=\Vert x_i-x_{i^{\prime}}\Vert^2$$
An iterative algorithm for minimizing $W(C)$ is given by:

For a given cluster assignment function, $C$, let $N_k=\sum_{i=1}^NI(C(i)=k)$

Step 1: Given $C$, minimize
$$\min_{C,\{m_k\}_1^K}\sum_{k=1}^NN_k\sum_{C(i)=k}\Vert x_i-m_k\Vert^2\quad(\star)$$
with respect to $\{m_1,\dots, m_k\}$ yielding the means of the currently assigned clusters.

Step 2: Given a set of means $\{m_1,\dots, m_k\}$ minimize $(\star)$ by 
$$C(i)=\argmin_{1\leq k\leq K}\Vert x_i-m_k\Vert^2$$

Step 3: Iterate steps 1 and 2 until the assignments do not change.