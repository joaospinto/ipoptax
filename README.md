# ipoptax

This repository implements a mini IPOPT written in JAX.
It uses an interior point method to optimize nonlinear nonconvex optimization
problems of the form

$$\min\limits_{x} f(x) \qquad \mbox{s.t.} \quad c(x) = 0 \wedge g(x) <= 0$$.

The functions $f, c, g$ are required to be continuously differentiable,
to ensure that our line searches succeed.
