Suppose $X$ is one dimensional. A piecewise polynomial, $f(X)$, can be used to approximate outputs by using basis functions. 

An order-$M$ spline with knots (points of joining between the intervals) $\xi_j$, $j=1,\dots, K$ is a piecewise-polynomial of order $M$, and has up to $M-2$ continuous derivative. The set of basis functions include
$$\begin{gather*}h_j(X)=X^{j-1},j=1,\dots,M\\h_{M+l}(X)=(X-\xi_{l})_+^{M-1},l=1,\dots,K\end{gather*}$$
The second set of basis functions ensures continuity across the intervals.

We fit parameters $\beta_k$ using least squares to get
$$f(X)=\sum_{j=1}^M\beta_jh_j(X)+\sum_{l=1}^K\beta_{M+l}h_{M+l}(X)$$