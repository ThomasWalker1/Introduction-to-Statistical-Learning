## Best Subset Selection
A method to find the subset of size $k$, for each $k=1,\dots,p$, that gives the smallest residual sum of squares.

There are efficient algorithms to compute these subsets, *leaps and bounds*, that are feasible to compute when $p$ is around $30$ or $40$. 

## Forward and Backward Stepwise Selection

Rather than sampling each possible subset we can iteratively seek a path through them

### Forward-Stepwise Selection
1. Start with an intercept
2. Add to model to improve its fit
A greedy algorithm producing a nested sequence of models. 
**Benefits**
- Computationally more efficient then best subset selection
- Has lower variance


**Drawbacks**
- Potentially higher bias
- Not guaranteed to select optimal subset

### Backward-Stepwise Selection
1. Start with fitted model
2. Sequentially remove predictors that have the least impact on the fit, determine this using *Z-scores*

**Drawbacks**
- Can only be used when $N>p$

### Forward-Stagewise Regression

1. Start with intercept equal to $\bar{y}$ and centered predictors with coefficients $0$
2. Identify variable most correlated with residual
3. Compute linear regression coefficient on the residual and this variable
4. Add this value to the current coefficient value
5. Continue until none of the variables have correlation with the residual (that is the model is fitted)

**Benefits:**
- Can prove more powerful in higher dimensional cases


**Drawbacks:**
- Requires many more steps to get to the least squares fit
