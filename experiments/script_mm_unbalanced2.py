def mm_unbalanced(a, b, M, reg_m, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / - reg_m / 2)
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)

    for i in range(numItermax):
        Gprev = G

        if div == 'kl':
            u = nx.sqrt(a / (nx.sum(G, 1) + 1e-16))
            v = nx.sqrt(b / (nx.sum(G, 0) + 1e-16))
            G = G * K * u[:, None] * v[None, :]
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G


def mm_unbalanced2(a, b, M, reg_m, div='kl', G0=None, numItermax=1000,
                   stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - :math:`\mathrm{div}` is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    ot_distance : array-like
        the OT distance between :math:`\mathbf{a}` and :math:`\mathbf{b}`
    log : dict
        log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 1, 'l2'),2)
    0.25
    >>> np.round(ot.unbalanced.mm_unbalanced2(a, b, M, 1, 'kl'),2)
    0.57

    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd2 : Unregularized OT loss
    ot.unbalanced.sinkhorn_unbalanced2 : Entropic regularized OT loss
    """
    _, log_mm = mm_unbalanced(a, b, M, reg_m, div=div, G0=G0,
                              numItermax=numItermax, stopThr=stopThr,
                              verbose=verbose, log=True)

    if log:
        return log_mm['cost'], log_mm
    else:
        return log_mm['cost']

def mm_unbalanced_revised(a, b, M, reg_m, l_rate=0.5, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / (- reg_m *l_rate))
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)

    for i in range(numItermax):
        Gprev = G

        if div == 'kl':
            u = nx.power(a / (nx.sum(G, 1) + 1e-16),l_rate)
            v = nx.power(b / (nx.sum(G, 0) + 1e-16),l_rate)
            G = G * K * u[:, None] * v[None, :]
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G

def mm_unbalanced_revised_scale(a, b, M, reg_m, l_rate=0.5, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / (- reg_m *l_rate))
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)

    for i in range(numItermax):
        Gprev = G

        if div == 'kl':
            u = nx.power(a / (nx.sum(G, 1) + 1e-16),l_rate)
            v = nx.power(b / (nx.sum(G, 0) + 1e-16),l_rate)
            G = G * K * u[:, None] * v[None, :]/G.sum()
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G

def mm_unbalanced_revised_nestrov(a, b, M, reg_m, l_rate=0.5, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / (- reg_m *l_rate))
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)
    tk = 1
    for i in range(numItermax):

        Gprev = G

        if div == 'kl':
            u = nx.power(a / (nx.sum(G, 1) + 1e-16),l_rate)
            v = nx.power(b / (nx.sum(G, 0) + 1e-16),l_rate)
            G = G * K * u[:, None] * v[None, :]
            t_knew = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
            G  = G + ((tk - 1) / t_knew) * (G - Gprev)
            tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G

def mm_unbalanced_revised_nestrov_scale(a, b, M, reg_m, l_rate=0.5, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / (- reg_m *l_rate))
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)
    tk = 1
    for i in range(numItermax):

        Gprev = G

        if div == 'kl':
            u = nx.power(a / (nx.sum(G, 1) + 1e-16),l_rate)
            v = nx.power(b / (nx.sum(G, 0) + 1e-16),l_rate)
            G = G * K * u[:, None] * v[None, :]
            G = G/ G.sum()
            t_knew = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
            G  = G + ((tk - 1) / t_knew) * (G - Gprev)
            tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G

def mm_unbalanced_revised_Enestrov(a, b, M, reg_m, l_rate=0.5, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / (- reg_m *l_rate))
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)
    tk = 1
    for i in range(numItermax):

        Gprev = G

        if div == 'kl':
            u = nx.power(a / (nx.sum(G, 1) + 1e-16),l_rate)
            v = nx.power(b / (nx.sum(G, 0) + 1e-16),l_rate)
            G = G * K * u[:, None] * v[None, :]
            t_knew = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
            G = G*(nx.exp((tk - 1) / t_knew)) * (G / Gprev)
            tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G

def mm_unbalanced_revised_Enestrov_scale(a, b, M, reg_m, l_rate=0.5, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / (- reg_m *l_rate))
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)
    tk = 1
    for i in range(numItermax):

        Gprev = G

        if div == 'kl':
            u = nx.power(a / (nx.sum(G, 1) + 1e-32),l_rate)
            v = nx.power(b / (nx.sum(G, 0) + 1e-32),l_rate)
            G = G * K * u[:, None] * v[None, :]
            G  = G/G.sum()
            t_knew = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
            G = G* (nx.exp((tk - 1) / t_knew)) * (G / Gprev)
            G  = G/G.sum()
            tk = (1 + math.sqrt(1 + 4 * tk ** 2)) / 2
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-32
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G

def mm_unbalanced_dynamic(a, b, M, reg_m, div='kl', G0=None, numItermax=1000,
                  stopThr=1e-15, verbose=False, log=False):
    r"""
    Solve the unbalanced optimal transport problem and return the OT plan.
    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{div}(\gamma^T \mathbf{1}, \mathbf{b})
        s.t.
             \gamma \geq 0

    where:

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target
      unbalanced distributions
    - div is a divergence, either Kullback-Leibler or :math:`\ell_2` divergence

    The algorithm used for solving the problem is a maximization-
    minimization algorithm as proposed in :ref:`[41] <references-regpath>`

    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    M : array-like (dim_a, dim_b)
        loss matrix
    reg_m: float
        Marginal relaxation term > 0
    div: string, optional
        Divergence to quantify the difference between the marginals.
        Can take two values: 'kl' (Kullback-Leibler) or 'l2' (quadratic)
    G0: array-like (dim_a, dim_b)
        Initialization of the transport matrix
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True
    Returns
    -------
    gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
    log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------
    >>> import ot
    >>> import numpy as np
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[1., 36.],[9., 4.]]
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'kl'), 2)
    array([[0.3 , 0.  ],
           [0.  , 0.07]])
    >>> np.round(ot.unbalanced.mm_unbalanced(a, b, M, 1, 'l2'), 2)
    array([[0.25, 0.  ],
           [0.  , 0.  ]])


    .. _references-regpath:
    References
    ----------
    .. [41] Chapel, L., Flamary, R., Wu, H., Févotte, C., and Gasso, G. (2021).
        Unbalanced optimal transport through non-negative penalized
        linear regression. NeurIPS.
    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.unbalanced.sinkhorn_unbalanced : Entropic regularized OT
    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if G0 is None:
        G = a[:, None] * b[None, :]
    else:
        G = G0

    if log:
        log = {'err': [], 'G': []}

    if div == 'kl':
        K = nx.exp(M / - reg_m / 2)
    elif div == 'l2':
        K = nx.maximum(a[:, None] + b[None, :] - M / reg_m / 2,
                       nx.zeros((dim_a, dim_b), type_as=M))
    else:
        warnings.warn("The div parameter should be either equal to 'kl' or \
                      'l2': it has been set to 'kl'.")
        div = 'kl'
        K = nx.exp(M / - reg_m / 2)

    for i in range(numItermax):
        Gprev = G

        if div == 'kl':
            K = nx.exp(M / - (reg_m* (i+1)/numItermax) / 2)
            u = nx.sqrt(a / (nx.sum(G, 1) + 1e-16))
            v = nx.sqrt(b / (nx.sum(G, 0) + 1e-16))
            G = G * K * u[:, None] * v[None, :]
        elif div == 'l2':
            Gd = nx.sum(G, 0, keepdims=True) + nx.sum(G, 1, keepdims=True) + 1e-16
            G = G * K / Gd

        err = nx.sqrt(nx.sum((G - Gprev) ** 2))
        if log:
            log['err'].append(err)
            log['G'].append(G)
        if verbose:
            print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['cost'] = nx.sum(G * M)
        return G, log
    else:
        return G



def sinkhorn_knopp_unbalanced(a, b, M, reg, reg_m, numItermax=1000,
                              stopThr=1e-6, verbose=False, log=False, **kwargs):
    r"""
    Solve the entropic regularization unbalanced optimal transport problem and return the loss

    The function solves the following optimization problem:

    .. math::
        W = \min_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F + \mathrm{reg}\cdot\Omega(\gamma) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma \mathbf{1}, \mathbf{a}) +
        \mathrm{reg_m} \cdot \mathrm{KL}(\gamma^T \mathbf{1}, \mathbf{b})

        s.t.
             \gamma \geq 0

    where :

    - :math:`\mathbf{M}` is the (`dim_a`, `dim_b`) metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - KL is the Kullback-Leibler divergence

    The algorithm used for solving the problem is the generalized Sinkhorn-Knopp matrix scaling algorithm as proposed in :ref:`[10, 25] <references-sinkhorn-knopp-unbalanced>`


    Parameters
    ----------
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,) or array-like (dim_b, n_hists)
        One or multiple unnormalized histograms of dimension `dim_b`
        If many, compute all the OT distances (a, b_i)
    M : array-like (dim_a, dim_b)
        loss matrix
    reg : float
        Entropy regularization term > 0
    reg_m: float
        Marginal relaxation term > 0
    numItermax : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshold on error (> 0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True


    Returns
    -------
    if n_hists == 1:
        - gamma : (dim_a, dim_b) array-like
            Optimal transportation matrix for the given parameters
        - log : dict
            log dictionary returned only if `log` is `True`
    else:
        - ot_distance : (n_hists,) array-like
            the OT distance between :math:`\mathbf{a}` and each of the histograms :math:`\mathbf{b}_i`
        - log : dict
            log dictionary returned only if `log` is `True`

    Examples
    --------

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> M=[[0., 1.],[1., 0.]]
    >>> ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, M, 1., 1.)
    array([[0.51122823, 0.18807035],
           [0.18807035, 0.51122823]])


    .. _references-sinkhorn-knopp-unbalanced:
    References
    ----------
    .. [10] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016).
        Scaling algorithms for unbalanced transport problems. arXiv preprint
        arXiv:1607.05816.

    .. [25] Frogner C., Zhang C., Mobahi H., Araya-Polo M., Poggio T. :
        Learning with a Wasserstein Loss,  Advances in Neural Information
        Processing Systems (NIPS) 2015

    See Also
    --------
    ot.lp.emd : Unregularized OT
    ot.optim.cg : General regularized OT

    """
    M, a, b = list_to_array(M, a, b)
    nx = get_backend(M, a, b)

    dim_a, dim_b = M.shape

    if len(a) == 0:
        a = nx.ones(dim_a, type_as=M) / dim_a
    if len(b) == 0:
        b = nx.ones(dim_b, type_as=M) / dim_b

    if len(b.shape) > 1:
        n_hists = b.shape[1]
    else:
        n_hists = 0

    if log:
        log = {'err': [],'u':[],'v':[],'time':[]}
    timestartagain = time.time()
    timeacc = 0
    # we assume that no distances are null except those of the diagonal of
    # distances
    if n_hists:
        u = nx.ones((dim_a, 1), type_as=M) / dim_a
        v = nx.ones((dim_b, n_hists), type_as=M) / dim_b
        a = a.reshape(dim_a, 1)
    else:
        u = nx.ones(dim_a, type_as=M) / dim_a
        v = nx.ones(dim_b, type_as=M) / dim_b

    K = nx.exp(M / (-reg))

    fi = reg_m / (reg_m + reg)

    err = 1.

    for i in range(numItermax):
        uprev = u
        vprev = v

        Kv = nx.dot(K, v)
        u = (a / Kv) ** fi
        Ktu = nx.dot(K.T, u)
        v = (b / Ktu) ** fi

        if (nx.any(Ktu == 0.)
                or nx.any(nx.isnan(u)) or nx.any(nx.isnan(v))
                or nx.any(nx.isinf(u)) or nx.any(nx.isinf(v))):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            warnings.warn('Numerical errors at iteration %s' % i)
            u = uprev
            v = vprev
            break

        err_u = nx.max(nx.abs(u - uprev)) / max(
            nx.max(nx.abs(u)), nx.max(nx.abs(uprev)), 1.
        )
        err_v = nx.max(nx.abs(v - vprev)) / max(
            nx.max(nx.abs(v)), nx.max(nx.abs(vprev)), 1.
        )
        err = 0.5 * (err_u + err_v)
        if log:
            log['err'].append(err)
            timewithoutsave = time.time() - timestartagain
            timeacc += timewithoutsave
            log['u'].append(u)
            log['v'].append(v)
            timestartagain = time.time()
            log['time'].append(timeacc)
            if verbose:
                if i % 50 == 0:
                    print(
                        '{:5s}|{:12s}'.format('It.', 'Err') + '\n' + '-' * 19)
                print('{:5d}|{:8e}|'.format(i, err))
        if err < stopThr:
            break

    if log:
        log['logu'] = nx.log(u + 1e-300)
        log['logv'] = nx.log(v + 1e-300)

    if n_hists:  # return only loss
        res = nx.einsum('ik,ij,jk,ij->k', u, K, v, M)
        if log:
            return res, log
        else:
            return res

    else:  # return OT matrix

        if log:
            return u[:, None] * K * v[None, :], log
        else:
            return u[:, None] * K * v[None, :]
