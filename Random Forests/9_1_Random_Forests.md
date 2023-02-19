## Tree-Based Methods 

### Regression

The idea is to partition the feature space. Suppose our data consists of 
$$(x_i,y_i),\; i=1,2\dots, N\text{ where }x_i=(x_{i1},\dots,x_{ip})$$

Consider a partition of $M$ regions, $R_1,\dots, R_m$, which we use to model the response as
$$f(x)=\sum_{m=1}^Mc_m I(x\in R_m)$$
We can regress on these region using least squares to obtain the fitted parameters
$$\hat{c}_m=\text{ave}(y_i\vert x_i\in R_m)$$

However, how do we find the best regions to partition the feature space? To do this we perform binary partitions, that is we split a variable $j$ at a split point $s$ to form the regions
$$R_1(j,s)=\{X\vert X_j\leq s\}\text{ and }R_2(j,s)=\{X\vert X_j> s\}$$
We choose $j$ and $s$ so that they solve
$$\min_{j,s}\left[\sum_{x_i\in R_1(j,s)}(y_i-\hat{c}_1)^2+\sum_{x_i\in R_2(j,s)}(y_i-\hat{c}_1)^2\right]$$
In practice the $s$ is computed for each $j$, then the best $(j,s)$ pair is chosen. This process can be repeated on the subsequent regions about to develop the tree.

How many regions should the feature space be partition into to?

The preferred strategy is to grow a large tree $T_0$ and then prune it using cost-complexity pruning. 
Define a subtree $T\subset T_0$ to be any tree that can be obtained from $T_0$ through pruning (collapsing internal nodes). Index terminal nodes by $m$, with node $m$ corresponding to region $m$. Let $\vert T\vert$ denote number of terminal nodes in $T$, then define
$$N_m=\#\{x_i\in R_m\},\;\hat{c}_m=\frac{1}{N_m}\sum_{x_i\in R_m}y_i,\;Q_m(T)=\frac{1}{N_m}\sum_{x_i\in R_m}(y_i-\hat{c}_m)^2$$
Now define the cost complex criterion as
$$C_{\alpha}(T)=\sum_{m=1}^{\vert T\vert}N_mQ_m(T)+\alpha\vert T\vert$$
The aim is to find the subtree that minimizes this expression. $\alpha\geq 0$ is a tuning parameter and governs tradeoff between size and goodness of fit.

### Classification

The structure here is very similar to above, however, the criteria for splitting nodes and pruning are different. $Q_m$ was given by least squares in the regression case. In the classification setting we there are several options as to what we can replace this with:
1. Misclassification Error:
$$1-\hat{p}_{mk(m)}$$
2. Gini Index
$$\sum_{k=1}^K\hat{p}_{mk}(1-\hat{p}_{mk})$$
3. Cross-entropy 
$$-\sum_{k=1}^K\hat{p}_{mk}\log \hat{p}_{mk}$$

where
$$\hat{p}_{mk}=\frac{1}{N_m}\sum_{x_i\in R_m}I(y_i=k)$$

## Random Forests


**Algorithm (Random Forest for Regression/Classification):**
1. $b=1:B$
   1. Draw bootstrap sample $\mathbf{Z}^*$ of size $N$ from training data
   2. Grow random-forest tree $T_b$ to the bootstrapped data, by recursively repeating the following for each terminal node of the tree, until the minimum size node size $n_{\text{min}}$ is reached.
      1. Select $m$ variables at random from the $p$ variables
      2. Pick the best variable/split-point among the $m$
      3. Split the node into two daughter noes
2. Output the ensemble of trees $\{T_b\}_1^B$

To perform...
- ...regression: $\hat{f}_{\text{rf}}^B(x)=\frac{1}{B}\sum_{b=1}^BT_b(x)$
- ...classification: $\hat{C}_{\text{rf}}^B(x)=\text{majority vote }\{\hat{C}_b(x)\}_1^B$ where $\hat{C}_b(x)$ is the class prediction of the $b$th random-forest tree 