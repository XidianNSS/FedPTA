import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import loguru
from copy import deepcopy
from scipy.sparse.linalg import svds
from typing import Union, List
from sklearn.decomposition import PCA
import time


def pta_anomaly_detection(hsi_tensor_y: np.ndarray,
                          alpha: float = 1,
                          beta: float = 0.01,
                          tau: float = 1,
                          r: int = 20,  # truncate_rank
                          max_iter: int = 400,
                          tol: float = 0.01,
                          rho: float = 1.5,
                          mu: float = 1e-3,
                          mu_bar: float = 1e10,

                          logger: loguru._Logger = loguru.logger):
    """
    Main program (Algorithm 1) in 'Prior-Based Tensor Approximation for Anomaly Detection in Hyperspectral Imagery'
    Link: https://ieeexplore.ieee.org/abstract/document/9288702

    Args:
        hsi_tensor_y: input HSI tensor Y
        alpha: Hyperparameter, default 1
        beta: Hyperparameter, default 0.01
        tau: Hyperparameter, default 1
        r: Hyperparameter, default 20, this need to be finely tuned with HSI tensor
        max_iter: max iter round, default 400
        tol:
        rho:
        mu: Hyperparameter, default 1e-3, this need to be finely tuned with HSI tensor
        mu_bar:
        logger:

    Returns:

    """
    hsi_tensor_y = deepcopy(hsi_tensor_y)
    assert len(hsi_tensor_y.shape) == 3  # 3 dimension array
    H = hsi_tensor_y.shape[0]  # first dimension
    W = hsi_tensor_y.shape[1]  # second dimension
    D = hsi_tensor_y.shape[2]  # third dimension

    if not (0 < r < D and 0 < r < H * W):  # r need to be in range.
        logger.warning(f"Truncate rank is too high( {r} ), will replace it with {D - 1}.")
        r = D - 1

    normal_map = np.ones(H * W)  # For debug
    anomaly_map = np.ones(H * W)

    delta_h = np.zeros([H, H - 1]).astype('float64')  # create D_H, Eq.2
    np.fill_diagonal(delta_h[:, 0:], -1)
    np.fill_diagonal(delta_h[1:], 1)
    h_t_dot_h = delta_h @ delta_h.T  # D_H^T @ D_H

    delta_w = np.zeros([W, W - 1]).astype('float64')  # create D_H, Eq.2
    np.fill_diagonal(delta_w[:, 0:], -1)
    np.fill_diagonal(delta_w[1:], 1)
    w_t_dot_w = delta_w @ delta_w.T  # W_H^T @ W_H

    X = np.zeros([H, W, D]).astype('float64')  # X = 0
    S = deepcopy(X)  # S = 0
    V1 = deepcopy(X)  # M1 = 0, M2 = 0, ...
    V2 = deepcopy(X)
    V3 = deepcopy(X)
    V4 = deepcopy(X)

    D1 = deepcopy(X)  # D1 = 0, D2 = 0, ..., D4 = 0, Actually here D2 is D1 in Algorithm.1
    D2 = deepcopy(X)
    D3 = deepcopy(X)
    D4 = deepcopy(X)
    D5 = deepcopy(X)
    A = np.zeros([D, max(r, 1)])
    B = np.zeros([H * W, max(r, 1)])
    area0 = []
    res = np.inf * np.ones(5)
    RES = []
    sigma2 = 0  # init

    epsilon = 1e-6
    k = 0  # iter round
    convergence = False
    while convergence is False:
        k += 1
        # update V1  OK
        X = (hsi_tensor_y - S + D1 + D2 + D3 + D4 + V1 + V2 + V3) / 4
        X1 = tensor_unfold(X - D2, 0)
        M1 = mu * np.linalg.solve((tau * h_t_dot_h + mu * np.eye(H)).T, X1.T).T  # Eq.16
        V1 = tensor_fold(M1, np.array([H, W, D]), 0)

        # update V2  OK
        X2 = tensor_unfold(X - D3, 1)
        M2 = mu * np.linalg.solve((tau * w_t_dot_w + mu * np.eye(W)).T, X2.T).T  # Eq.17
        V2 = tensor_fold(M2, np.array([H, W, D]), 1)

        # update V3  OK
        temp = 0.5 * (X + V4 - D4 + D5)  # Eq.18 inside norm, right
        temp3 = tensor_unfold(temp, 2).T
        U, sigma, V = np.linalg.svd(temp3, full_matrices=False)
        V = V.T
        svp = len(sigma[sigma > alpha / (2 * mu)])
        if svp >= 1:
            sigma = sigma[:svp] - (alpha / (2 * mu))
        else:
            svp = 1
            sigma = 0

        V3_3 = U[:, :svp] @ np.diag(sigma) @ (V[:, :svp]).T
        V3 = tensor_fold(V3_3.T, np.array([H, W, D]), 2)

        # update V4  OK
        temp3 = tensor_fold(alpha * B @ A.T / mu, np.array([H, W, D]), 2)  # Eq.20
        V4 = V3 - D5 + temp3

        # update A and B
        temp3 = tensor_unfold(V4, 2).T
        if r >= 1:
            A, sigma2, B = svds(temp3, k=r)  # Eq.21
            B = B.T
            sigma2 = np.diag(sigma2)

        if r == 0 or sigma2[0, 0] == 0:
            A = np.zeros([D, max(r, 1)])
            B = np.zeros([H * W, max(r, 1)])

        # update S  OK
        be = l21_norm(tensor_unfold(S, axis=2)) * beta / mu + 0.5 * np.linalg.norm(hsi_tensor_y - X + D1 - S)
        S = solve_l1l1l2(hsi_tensor_y - X + D1, beta / mu)
        S_temp = solve_prox_l21_beta_with_axis(hsi_tensor_y - X + D1, beta / mu, axis=2)
        af = l21_norm(tensor_unfold(S, axis=2)) * beta / mu + 0.5 * np.linalg.norm(hsi_tensor_y - X + D1 - S)
        af2 = l21_norm(tensor_unfold(S_temp, axis=2)) * beta / mu + 0.5 * np.linalg.norm(hsi_tensor_y - X + D1 - S_temp)
        print(f"before solve prox: {be}, after: {af}, my: {af2}")

        # update D1-D5  OK
        D1 = D1 + (hsi_tensor_y - X - S)
        D2 = D2 - X + V1
        D3 = D3 - X + V2
        D4 = D4 - X + V3
        D5 = D5 - V3 + V4  # Eq.25

        if k % 10 == 1:
            t0 = (hsi_tensor_y - X - S).ravel(order='F')  # t0 - t4 OK
            res[0] = np.sqrt(np.dot(t0, t0))
            t1 = (X - V1).ravel(order='F')
            res[1] = np.sqrt(np.dot(t1, t1))
            t2 = (X - V2).ravel(order='F')
            res[2] = np.sqrt(np.dot(t2, t2))
            t3 = (X - V3).ravel(order='F')
            res[3] = np.sqrt(np.dot(t3, t3))
            t4 = (V3 - V4).ravel(order='F')
            res[4] = np.sqrt(np.dot(t4, t4))

            f_show = np.sqrt(np.sum(S ** 2, axis=2))  # f_show, r_max, taus OK
            r_max = f_show.max()
            taus = np.linspace(0, r_max, 5000)
            PF0 = np.zeros(len(taus))
            PD0 = np.zeros(len(taus))
            for index2 in range(len(taus)):
                tau1 = taus[index2]
                anomaly_map_rx = f_show.ravel() > tau1
                # PF0[index2] = np.sum(anomaly_map_rx & normal_map) / np.sum(normal_map)
                PF0[index2] = np.sum(np.logical_and(anomaly_map_rx, normal_map)) / np.sum(normal_map)
                # PD0[index2] = np.sum(anomaly_map_rx & anomaly_map) / np.sum(anomaly_map)
                PD0[index2] = np.sum(np.logical_and(anomaly_map_rx, anomaly_map)) / np.sum(anomaly_map)

            idx = (k - 1) // 10
            area0.append(np.sum((PF0[:-1] - PF0[1:]) * (PD0[1:] + PD0[:-1]) / 2))
            RES.append(res[0])
            print('iter =', k, '- res(1) =', res[0], ',res(2) =', res[1], ',res(3) =', res[2], ',res(4) =',
                  res[3], ',res(5) =', res[4], ',AUC=', area0[idx])
        convergence = not ((k <= max_iter) and (np.sum(np.abs(res)) > tol))
        mu = min(mu * rho, mu_bar)
    area0 = area0[-1]
    return X, S, area0


def robust_admm(Y: np.ndarray,
                beta: float = 0.01,
                tau: float = 1,
                mu: float = 1e-3,
                max_iter: int = 400,
                tol: float = 1e-3,
                dm_idx: Union[List[int], np.ndarray] = None,
                logger: loguru._Logger = loguru.logger,
                verbose: bool = False
                ):
    """
    Input tensor Y: shape = [d(dimension), m(clients), T(round)]


    """
    Y = deepcopy(Y)
    assert len(Y.shape) == 3  # 3 dimension array
    d = Y.shape[0]  # first dimension (Param)
    m = Y.shape[1]  # second dimension (Client)
    t = Y.shape[2]  # third dimension (Round)

    D_t = np.zeros([t, t - 1]).astype('float64').T  # create D_T
    np.fill_diagonal(D_t[:, :-1], 1)
    np.fill_diagonal(D_t[:, 1:], -1)
    D_t_T_dot_D_t = D_t.T @ D_t  # shape = [t, t]

    if dm_idx is None:
        D_m = -1 * np.ones([m, m]).astype('float64') + m * np.identity(m)  # create D_m
    else:
        D_m = (np.zeros([m, m]).astype('float64'))
        D_m[:, dm_idx] = -1
        D_m += len(dm_idx) * np.identity(m)

    D_m_T_dot_D_m = D_m.T @ D_m  # shape = [m, m]

    X = np.zeros([d, m, t]).astype('float64')  # X = 0
    S = deepcopy(X)  # S = 0
    V2 = deepcopy(X)  # M2 fold
    V3 = deepcopy(X)  # M3 fold

    lam = deepcopy(X)  # lambda
    D2 = deepcopy(X)  # D2
    D3 = deepcopy(X)  # D3

    k = 0  # iter round
    convergence = False

    conv = np.inf * np.ones(3)
    conv_float = 0

    start_time = time.time()
    while convergence is False:
        k += 1

        # update X
        X = (Y - S + D2 + D3 + V2 + V3 + lam) / 3

        # update M2, V2 is folded version of M2
        X2 = tensor_unfold(X - D2, 1)
        M2 = mu * np.linalg.solve((D_m_T_dot_D_m + mu * np.eye(m)).T, X2.T).T
        V2 = tensor_fold(M2, np.array([d, m, t]), 1)

        # update M3, V3 is folded version of M3
        X3 = tensor_unfold(X - D3, 2)
        M3 = mu * np.linalg.solve((D_t_T_dot_D_t + mu * np.eye(t)).T, X3.T).T
        V3 = tensor_fold(M3, np.array([d, m, t]), 2)

        # update X
        # X = mu * (V2 - D2) + mu * (V3 - D3) + tau * (Y - S) + lam
        # X = X / (mu * 2 + tau)

        # update S
        be = l21_norm(tensor_unfold(S, axis=1)) * beta / tau + 0.5 * np.linalg.norm(Y - X + lam - S)
        S = solve_prox_l21_beta_with_axis(Y - X + lam, beta / tau, axis=1)
        af = l21_norm(tensor_unfold(S, axis=1)) * beta / tau + 0.5 * np.linalg.norm(Y - X + lam - S)
        if verbose:
            logger.debug(f"before solve prox: {be}, after: {af}")

        # update auxiliary variables
        lam = lam + (Y - X - S)
        D2 = D2 - X + V2
        D3 = D3 - X + V3

        # convergence judgment
        conv[0] = np.linalg.norm(Y - X - S)
        conv[1] = np.linalg.norm(X - V2)
        conv[2] = np.linalg.norm(X - V3)
        conv_float = conv[0] + conv[1] + conv[2]

        if verbose and k % 10 == 1:
            # argmin target
            arg_target = 0
            arg_target += 0.5 * np.linalg.norm(D_m @ tensor_unfold(X, 1).T)
            arg_target += 0.5 * np.linalg.norm(D_t @ tensor_unfold(X, 2).T)
            arg_target += beta * l21_norm(tensor_unfold(S, 0))
            logger.debug(f'iter={k}, conv={conv_float}, arg min target={arg_target}')
        convergence = not ((k <= max_iter) and conv_float > tol)
    end_time = time.time()
    if verbose:
        logger.debug(f'Admm time used: {end_time-start_time}, iter: {k}')

    return X, S


def tensor_unfold(origin: np.ndarray, axis: int = 0):
    """
    Reshape a high dimension array to a 2-dimension array, and shape is like (, origin.shape[axis])
    Reshaped array will be a 'slice' of axis.

    This function has been verified, compared to its MATLAB version.
    Args:
        origin:
        axis:

    Returns:

    """

    assert len(origin.shape) > 2 and 0 <= axis < len(origin.shape)

    size_t = np.array(origin.shape)
    n_dim = len(size_t)
    num = 1
    num_pre = 1
    num_post = 1

    for i in range(n_dim):
        num *= size_t[i]
        if i < axis:
            num_pre *= size_t[i]
        elif i > axis:
            num_post *= size_t[i]

    out = np.zeros((num // size_t[axis], size_t[axis]))

    B = np.reshape(origin, (num_pre, size_t[axis], num_post), order='F')

    for i in range(size_t[axis]):
        B_temp = B[:, i, :]
        temp = np.squeeze(B[:, i, :])
        out[:, i] = temp.flatten(order='F')

    return out


def tensor_fold(origin: np.ndarray,
                target_size: Union[np.ndarray, List[int]],
                axis: int = 0):
    """
    Fix an axis (indicated by param axis) and make the array shape == target_size

    This function has been verified, compared to its MATLAB version.
    Args:
        origin:
        target_size:
        axis:

    Returns:

    """
    target_size = np.array(target_size)

    assert 0 <= axis < len(target_size)

    origin_size = np.array(origin.shape)
    origin_size_mul = 1
    target_size_mul = 1
    for d in origin_size:
        origin_size_mul *= d
    for d in target_size:
        target_size_mul *= d
    assert target_size_mul == origin_size_mul

    num_post = 1
    num_pre = 1
    num = 1
    dim = len(target_size)

    for i in range(dim):
        num *= target_size[i]
        if i < axis:
            num_pre *= target_size[i]
        elif i > axis:
            num_post *= target_size[i]

    out = np.zeros(num)

    num_b = num_pre * target_size[axis]

    for i in range(num_post):
        temp = origin[i * num_pre: (i + 1) * num_pre, :]
        out[i * num_b: (i + 1) * num_b] = temp.flatten(order='F')

    out = np.reshape(out, target_size, order='F')
    return out


def solve_prox_l21_beta(origin: np.ndarray, beta):
    """

    Usage: S_ver = solve_prox_l21_beta(tensor_unfold(hsi_tensor_y - X + D1, 2).T, beta / tau).T
    Args:
        origin:
        beta:

    Returns:

    """

    assert len(origin.shape) == 2
    lines = origin.shape[0]
    columns = origin.shape[1]

    result = np.zeros([lines, columns])
    for j in range(columns):
        y_j = origin[:, j]
        y_j_norm = np.linalg.norm(y_j)
        y_j_norm_beta = np.maximum(y_j_norm - beta, 0)
        if y_j_norm_beta != 0:
            result[:, j] = np.repeat(y_j_norm_beta / y_j_norm, lines) * y_j

    return result


def solve_prox_l21_beta_with_axis(origin: np.ndarray, beta, axis: int = 0):
    result = solve_prox_l21_beta(tensor_unfold(origin, axis).T, beta).T
    result = tensor_fold(result, origin.shape, axis=axis)
    return result


def l21_norm(origin: np.ndarray):
    assert len(origin.shape) == 2
    lines = origin.shape[0]
    columns = origin.shape[1]
    result = 0

    for i in range(columns):
        result += np.linalg.norm(origin[:, i])
    return result


def l12_norm(origin: np.ndarray):
    assert len(origin.shape) == 2
    lines = origin.shape[0]
    columns = origin.shape[1]
    result = 0

    for i in range(lines):
        result += np.linalg.norm(origin[i, :])
    return result


def min_max_normalize(arr: np.ndarray, axis: int = None) -> np.ndarray:
    """
    Get min max normalization of an array
    Args:
        arr:
        axis: if None or arr is a vector, run all normalization on the whole array
              else do normalization on axis.
              Examples:
                  a = np.arange(10).reshape(1, 5, 2)
                  b = min_max_normalize(a, axis=0)

                  a[0] is:
                  [[0 1]
                   [2 3]
                   [4 5]
                   [6 7]
                   [8 9]]

                  b[0] is:
                  [[0.         0.11111111]
                   [0.22222222 0.33333333]
                   [0.44444444 0.55555556]
                   [0.66666667 0.77777778]
                   [0.88888889 1.        ]]


    Returns: Normalization result

    """
    if axis is None or len(arr.shape) == 1:
        return (arr - arr.min()) / (arr.max() - arr.min())
    else:
        assert 0 <= axis < len(arr.shape)

        unfolded = tensor_unfold(arr, axis)  # (?, arr.shape[axis])
        for i in range(unfolded.shape[1]):
            if unfolded[:, i].max() != unfolded[:, i].min():  # no zero div!
                unfolded[:, i] = (unfolded[:, i] - unfolded[:, i].min()) / (unfolded[:, i].max() - unfolded[:, i].min())
            else:
                unfolded[:, i] = 1
        res = tensor_fold(unfolded, np.array(arr.shape), axis=axis)
        return res


def medium_analysis(origin: np.ndarray):
    assert len(origin.shape) == 1
    lines = origin.shape[0]
    median = np.median(origin)

    # 计算向量中所有元素与中位数之间的距离
    distances = np.abs(origin - median)

    # 根据距离排序，并选择距离最小的n/2个值对应的下标
    sorted_indices = np.argsort(distances)
    n = origin.size
    n_over_2 = n // 2
    nearest_indices = sorted_indices[:n_over_2]

    return nearest_indices


def solve_l1l1l2(origin: np.ndarray, lam):
    """

    This function has been verified, compared to its MATLAB version.
    Args:
        origin:
        lam:

    Returns:

    """
    assert len(origin.shape) == 3
    H = origin.shape[0]
    W = origin.shape[1]
    D = origin.shape[2]
    nm = np.sqrt(np.sum(origin ** 2, axis=2))
    nms = np.maximum(nm - np.ones((H, W)) * lam, 0)
    # temp = (nms / nm).reshape(-1)
    sw = np.repeat((nms / nm).reshape(-1), D).reshape([H, W, D])
    # temp_sw = sw[:, :, 0]
    E = sw * origin
    return E


def robust_admm_by_all_pca(hsi_y: np.ndarray,
                           pca_dim: int = None,
                           beta: float = 150,
                           mu: float = 10,
                           tau: float = 1,
                           tol: float = 1e-6,
                           logger: loguru._Logger = loguru.logger
                           ):
    d = hsi_y.shape[0]
    m = hsi_y.shape[1]
    T = hsi_y.shape[2]
    fit_data = tensor_unfold(hsi_y, axis=0)
    valid_lim = min(fit_data.shape[0], fit_data.shape[1])  # check pca n_components
    if pca_dim is None:
        pca_dim = valid_lim
    elif pca_dim < 0 or pca_dim > valid_lim:
        logger.warning(f"Using PCA but got invalid pca dimension. "
                       f"Using {valid_lim} instead of {pca_dim}.")
        pca_dim = valid_lim

    pca = PCA(n_components=pca_dim)
    res_data = pca.fit_transform(fit_data)
    hsi_y = tensor_fold(res_data, [pca_dim, m, T], axis=0)
    x, s = robust_admm(hsi_y, beta=beta, mu=mu, tau=tau, tol=tol)
    # x, s, temp = pta_anomaly_detection(hsi_y, beta=beta, mu=mu, tau=tau, tol=tol)
    return x, s, pca.explained_variance_


def softmax(f: np.ndarray, axis: int = None):
    if axis is None:
        f = deepcopy(f)
        f -= np.max(f)
        return np.exp(f) / np.sum(np.exp(f))  # safe to do, gives the correct answer

    else:
        assert 0 <= axis < len(f.shape)

        raise NotImplementedError
        # f = deepcopy(f)
        # for i in range(f.shape[axis]):
        #     f -= np.max(f)


