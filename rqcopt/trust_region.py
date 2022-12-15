import numpy as np
import warnings


def riemannian_trust_region_optimize(f, retract, gradfunc, hessfunc, x_init, **kwargs):
    """
    Optimization via the Riemannian trust-region (RTR) algorithm.

    Reference:
        Algorithm 10 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)
    """
    rho_trust   = kwargs.get("rho_trust", 0.125)
    radius_init = kwargs.get("radius_init", 0.01)
    maxradius   = kwargs.get("maxradius",   0.1)
    niter       = kwargs.get("niter", 20)
    gfunc       = kwargs.get("gfunc", None)
    # transfer keyword arguments for truncated_cg
    tcg_kwargs = {}
    for key in ["maxiter", "abstol", "reltol"]:
        if ("tcg_" + key) in kwargs.keys():
            tcg_kwargs[key] = kwargs["tcg_" + key]
    assert 0 <= rho_trust < 0.25
    x = x_init
    radius = radius_init
    f_iter = []
    g_iter = []
    if gfunc is not None:
        g_iter.append(gfunc(x))
    for k in range(niter):
        grad = gradfunc(x)
        hess = hessfunc(x)
        eta, on_boundary = truncated_cg(grad, hess, radius, **tcg_kwargs)
        x_next = retract(x, eta)
        fx = f(x)
        f_iter.append(fx)
        # Eq. (7.7)
        rho = (f(x_next) - fx) / (np.dot(grad, eta) + 0.5 * np.dot(eta, hess @ eta))
        if rho < 0.25:
            # reduce radius
            radius *= 0.25
        elif rho > 0.75 and on_boundary:
            # enlarge radius
            radius = min(2 * radius, maxradius)
        if rho > rho_trust:
            x = x_next
        if gfunc is not None:
            g_iter.append(gfunc(x))
    return x, f_iter, g_iter


def truncated_cg(grad, hess, radius, **kwargs):
    """
    Truncated CG (tCG) method for the trust-region subproblem:
        minimize   <grad, z> + 1/2 <z, H z>
        subject to <z, z> <= radius^2

    References:
      - Algorithm 11 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)
      - Trond Steihaug
        The conjugate gradient method and trust regions in large scale optimization
        SIAM Journal on Numerical Analysis 20, 626-637 (1983)
    """
    maxiter = kwargs.get("maxiter", 2 * len(grad))
    abstol  = kwargs.get("abstol", 1e-8)
    reltol  = kwargs.get("reltol", 1e-6)
    r = grad.copy()
    rsq = np.dot(r, r)
    stoptol = max(abstol, reltol * np.sqrt(rsq))
    z = np.zeros_like(r)
    d = -r
    for j in range(maxiter):
        Hd = hess @ d
        dHd = np.dot(d, Hd)
        t = _move_to_boundary(z, d, radius)
        alpha = rsq / dHd
        if dHd <= 0 or alpha > t:
            # return with move to boundary
            return z + t*d, True
        # update iterates
        r += alpha * Hd
        z += alpha * d
        rsq_next = np.dot(r, r)
        if np.sqrt(rsq_next) <= stoptol:
            # early stopping
            return z, False
        beta = rsq_next / rsq
        d = -r + beta * d
        rsq = rsq_next
    # maxiter reached
    return z, False


def _move_to_boundary(b, d, radius):
    """
    Move to the unit ball boundary by solving
    || b + t*d || == radius
    for t with t > 0.
    """
    dsq = np.dot(d, d)
    if dsq == 0:
        warnings.warn("input vector 'd' is zero")
        return b
    p = np.dot(b, d) / dsq
    q = (np.dot(b, b) - radius**2) / dsq
    t = solve_quadratic_equation(p, q)[1]
    if t < 0:
        warnings.warn("encountered t < 0")
    return t


def solve_quadratic_equation(p, q):
    """
    Compute the two solutions of the quadratic equation x^2 + 2 p x + q == 0.
    """
    if p**2 - q < 0:
        raise ValueError("require non-negative discriminant")
    if p == 0:
        x = np.sqrt(-q)
        return (-x, x)
    x1 = -(p + np.sign(p)*np.sqrt(p**2 - q))
    x2 = q / x1
    return tuple(sorted((x1, x2)))
