In this approach a complete orthonormal basis of functions are used to represent functions. 

The bases are generated by translations and dilations of a scaling function $\phi(x)$ (father). The coefficients for the representation are determined through least squares.

## Haar Wavelets

Suppose $\phi(x)=I(x\in[0,1])$. Define $\phi_{0,k}(x)=\phi(x-k)$ for $k$ an integer. This generates an orthonormal basis of functions with jumps at the integers. Denote this space (the reference space) as $V_0$.
More generally,
$$\phi_{j,k}=2^{j/2}\phi(2^jx-k)\text{ spanning }V_j$$
Note that 
$$\dots\supset V_1\supset V_0\supset V_{-1}\supset\dots$$

For a function in $V_{j+1}$ one can represent it using a function in $V_j$ plus a function in the orthogonal complement to $V_j$ to $V_{j+1}$, denoted $W_j$ and called the *detail*. For the Haar basis the functions $\psi_{j,k}=2^{j/2}\psi(2^jx-k)$, where $\psi(x)=\phi(2x)-\phi(2x-1)$ forms an orthonormal basis for $W_j$. $\psi$ is referred to as the mother wavelet. Note that
$$V_{j+1}=V_j\oplus W_j=V_{j-1}\oplus W_{j-1}\oplus W_{j}=\dots$$

## Wavelet Filtering

Consider a one-dimensional lattice with $N=2^J$ lattice points. 
- $\mathbf{y}$ the response vector
- $\mathbf{W}$, $N\times N$ orthonormal wavelet basis matrix evaluated at the $N$ observations. 

$\mathbf{y}^{*}=\mathbf{W}^T\mathbf{y}$ is the wavelet transform of $\mathbf{y}$ (the full least squares regression coefficient). 
### Shrinkage
Using the criterion 
$$\min_{\mathbf{\theta}}\Vert\mathbf{y}-\mathbf{W\theta}\Vert_2^2+2\lambda\Vert\theta\Vert_1$$
through orthogonality we get the solution
$$\hat{\theta}_j=\text{sign}(y^*_j)(\vert y_j^*-\lambda)_+$$
That is, the least squares coefficient is translated then truncated. The fitted function is given by
$$\hat{\mathbf{f}}=\mathbf{W}\hat{\mathbf{\theta}}$$