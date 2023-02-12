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