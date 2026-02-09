#改成cupy的形式
import cupy as cp
from cupyx.scipy.special import ndtr  # GPU加速的正态分布累积分布函数
from tqdm import tqdm  # 引入进度条库
from joblib import Parallel, delayed


# 计算回归指标
def mean_squared_error(y_true, y_pred):
    return cp.mean((y_true - y_pred) ** 2)
def root_mean_squared_error(y_true, y_pred):
    return cp.sqrt(mean_squared_error(y_true, y_pred))
def mean_absolute_error(y_true, y_pred):
    return cp.mean(cp.abs(y_true - y_pred))
def r_squared(y_true, y_pred):
    ss_total = cp.sum((y_true - cp.mean(y_true)) ** 2)
    ss_residual = cp.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)
def beta_difference_2_norm(beta1, beta2):
    difference = cp.array(beta1) - cp.array(beta2)
    norm = cp.linalg.norm(difference, 2)
    return norm
# 核平滑函数，支持不同核函数
def K_tilde(u, kernel_type='gaussian'):
    if kernel_type == 'gaussian':
        return cp.exp(-0.5 * u ** 2) / cp.sqrt(2 * cp.pi)
    elif kernel_type == 'epanechnikov':
        return 0.75 * (1 - u ** 2) * (cp.abs(u) <= 1)
    else:
        raise ValueError("不支持的核函数类型")
def K_h(u, h):
    return K_tilde(u / h) / h
def K_bar_scalar(u):
    return ndtr(u)
def K_bar(u):
    return cp.vectorize(K_bar_scalar)(u)
def compute_gradients(X_batch, y_batch, beta_h, b_tau, tau, h):
    n, K = X_batch.shape[0], len(tau)
    U = -(y_batch[:, cp.newaxis] - X_batch @ beta_h[:, cp.newaxis] - b_tau[cp.newaxis, :]) / h
    W = K_bar(U)
    grad_beta_h = X_batch.T @ (W - tau) @ cp.ones(K) / (n * K)
    grad_b_tau = cp.sum(W - tau, axis=0) / (n * K)
    return grad_beta_h, grad_b_tau
def gradient(beta_h, b_tau, X, y, tau, h):
    grad_beta_h, grad_b_tau = compute_gradients(X, y, beta_h, b_tau, tau, h)
    return grad_beta_h, grad_b_tau
def BB_step(delta, eta, max_step=10):
    lambda_1 = cp.dot(delta, delta) / cp.dot(delta, eta)
    lambda_2 = cp.dot(delta, eta) / cp.dot(eta, eta)
    if lambda_1 > 0:
        return min(lambda_1, lambda_2, max_step)
    else:
        return 1.0
def shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta0, b_tau0, X1, y1):
    grad_beta_h, grad_b_tau = gradient(beta0, b_tau0, X, y, tau, h)
    grad_beta_b, grad_b_taub = gradient(beta_h, b_tau, X1, y1, tau, b)
    grad_beta_b0, grad_b_taub0 = gradient(beta0, b_tau0, X1, y1, tau, b)
    grad_shift_beta = grad_beta_b - grad_beta_b0 + grad_beta_h
    grad_shift_b_tau = grad_b_taub - grad_b_taub0 + grad_b_tau
    return grad_shift_beta, grad_shift_b_tau
def solve_CSQR(X, y, tau, h, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau
    for t in range(1, max_iter + 1):
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau


def solve_D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_CSQR(X1, y1, tau, b, nu, max_iter)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_1D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="1D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_2D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_1D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="2D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_3D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_2D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="3D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_4D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_3D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="4D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_5D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_4D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="5D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_6D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)
    beta_h = cp.zeros(p)
    b_tau = cp.zeros(M)
    beta_0, b_tau0 = solve_5D_CSQR(X, y, tau, h, b, X1, y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()
    for t in tqdm(range(1, max_iter + 1), desc="6D-CSQR 迭代进度"):
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, y1)
        delta = cp.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_combined = cp.concatenate([grad_beta_h, grad_b_tau])
        step_size = BB_step(delta, grad_combined)
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        if cp.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"D-CSQR 在 {t} 次迭代后收敛。")
            break
    else:
        print("D-CSQR 达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau


if __name__ == "__main__":
    n, p, m = 300, 10, 100
    N = n * m
    nu = 1e-4
    mean = cp.zeros(p)
    tau = cp.linspace(0.05, 0.95, 19)
    b = ((p + cp.log(n)) / n) ** (1 / 2)
    h = ((p + cp.log(N)) / N) ** (1 / 2)
    def generate_covariance_matrix(size):
        covariance_matrix = cp.zeros((size, size))
        for j in range(size):
            for k in range(size):
                covariance_matrix[j, k] = 2 * (0.5 ** abs(j - k))
        return covariance_matrix
    cov = generate_covariance_matrix(p)
    beta_true = cp.ones(p)



#-------------------------------------------------------------------
    # 对于全局SCQR，重复执行100次并求平均值
    results_beta_h = []
    results_b_tau = []

    for _ in range(100):
        X_parts = [cp.random.multivariate_normal(mean, cov, size=N // m) for _ in range(m)]
        X = cp.vstack(X_parts)
        errors = cp.random.standard_t(3, size=N)
        y = X @ beta_true + errors

        shuffled_indices = cp.random.permutation(N)
        shuffled_X = X[shuffled_indices]
        shuffled_errors = errors[shuffled_indices]
        shuffled_dataY = shuffled_X @ beta_true + shuffled_errors

        beta_h, b_tau = solve_CSQR(shuffled_X, shuffled_dataY, tau, h, nu)
        results_beta_h.append(beta_h)
        results_b_tau.append(b_tau)

    # 计算平均值
    average_beta_h = cp.mean(cp.array(results_beta_h), axis=0)
    average_b_tau = cp.mean(cp.array(results_b_tau), axis=0)
    biasall = beta_difference_2_norm(average_beta_h, beta_true)



    # beta_hall, b_tauall = solve_CSQR(shuffled_X, shuffled_dataY, tau, h, nu)
    # biasall = beta_difference_2_norm(beta_hall, beta_true)


    # #计算每个机器的平均，使用快速算法
    # def solve_local_CSQR(i, X, y, tau, h, nu):
    #     beta_hloc, b_tauloc = solve_CSQR(X, y, tau, h, nu)
    #     return beta_hloc, b_tauloc
    # # 并行计算
    # results = Parallel(n_jobs=-1)(
    #     delayed(solve_local_CSQR)(i, shuffled_X[(i-1) * N // m:N // m + i * N // m], shuffled_dataY[(i-1) * N // m:N // m + i * N // m], tau, b, nu) for i in range(m))
    # # 分离结果
    # results_beta = cp.array([res[0] for res in results])
    # results_b_tau = cp.array([res[1] for res in results])
    # sums = cp.sum(results_beta, axis=0)
    # results_beta = sums / len(results_beta)
    # biasresults = beta_difference_2_norm(results_beta, beta_true)

    # 对每个部分进行求解并计算平均值，重复100次
    results_beta_h_parts = []
    results_b_tau_parts = []

    for _ in range(100):  # 重复100次
        results_beta_h_temp = []
        results_b_tau_temp = []
        X_parts = [cp.random.multivariate_normal(mean, cov, size=N // m) for _ in range(m)]
        X = cp.vstack(X_parts)
        errors = cp.random.standard_t(3, size=N)
        y = X @ beta_true + errors

        shuffled_indices = cp.random.permutation(N)
        shuffled_X = X[shuffled_indices]
        shuffled_errors = errors[shuffled_indices]
        shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
        for i in range(m):
            beta_h_part, b_tau_part = solve_CSQR(shuffled_X[i * (N // m):(i + 1) * (N // m)],
                                                 shuffled_dataY[i * (N // m):(i + 1) * (N // m)],
                                                 tau, h, nu)
            results_beta_h_temp.append(beta_h_part)
            results_b_tau_temp.append(b_tau_part)

        # 计算每个部分的平均值
        average_beta_h_part = cp.mean(cp.array(results_beta_h_temp), axis=0)
        average_b_tau_part = cp.mean(cp.array(results_b_tau_temp), axis=0)

        results_beta_h_parts.append(average_beta_h_part)
        results_b_tau_parts.append(average_b_tau_part)

    # 计算最终的平均值
    final_average_beta_h = cp.mean(cp.array(results_beta_h_parts), axis=0)
    final_average_b_tau = cp.mean(cp.array(results_b_tau_parts), axis=0)
    biasav = beta_difference_2_norm(final_average_beta_h, beta_true)
    # -------------------------------------------------------------------

    #一步通信迭代
    results_beta_1D = []
    results_b_tau_1D = []
    for _ in range(100):
        X_parts = [cp.random.multivariate_normal(mean, cov, size=N // m) for _ in range(m)]
        X = cp.vstack(X_parts)
        errors = cp.random.standard_t(3, size=N)
        y = X @ beta_true + errors

        shuffled_indices = cp.random.permutation(N)
        shuffled_X = X[shuffled_indices]
        shuffled_errors = errors[shuffled_indices]
        shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
        beta_1D, b_tau1D = solve_D_CSQR(X, y, tau, h, b, shuffled_X[:N // m], shuffled_dataY[:N // m], nu=1e-4,
                                        max_iter=1000)
        results_beta_1D.append(beta_1D)
        results_b_tau_1D.append(b_tau1D)
    # 计算平均值
    average_beta_1D = cp.mean(cp.array(results_beta_1D), axis=0)
    average_b_tau_1D = cp.mean(cp.array(results_b_tau_1D), axis=0)
    # 计算偏差
    bias1D = beta_difference_2_norm(average_beta_1D, beta_true)
    # -------------------------------------------------------------------
    #四步通信迭代
    results_beta_4D = []
    results_b_tau_4D = []
    for _ in range(100):
        X_parts = [cp.random.multivariate_normal(mean, cov, size=N // m) for _ in range(m)]
        X = cp.vstack(X_parts)
        errors = cp.random.standard_t(3, size=N)
        y = X @ beta_true + errors

        shuffled_indices = cp.random.permutation(N)
        shuffled_X = X[shuffled_indices]
        shuffled_errors = errors[shuffled_indices]
        shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
        beta_1D, b_tau1D = solve_3D_CSQR(X, y, tau, h, b, shuffled_X[:N // m], shuffled_dataY[:N // m], nu=1e-4,
                                        max_iter=1000)
        results_beta_4D.append(beta_1D)
        results_b_tau_4D.append(b_tau1D)
    # 计算平均值
    average_beta_4D = cp.mean(cp.array(results_beta_4D), axis=0)
    average_b_tau_4D = cp.mean(cp.array(results_b_tau_4D), axis=0)
    # 计算偏差
    bias4D = beta_difference_2_norm(average_beta_4D, beta_true)
    # -------------------------------------------------------------------
    #七步通信迭代
    results_beta_7D = []
    results_b_tau_7D = []
    for _ in range(100):
        X_parts = [cp.random.multivariate_normal(mean, cov, size=N // m) for _ in range(m)]
        X = cp.vstack(X_parts)
        errors = cp.random.standard_t(3, size=N)
        y = X @ beta_true + errors

        shuffled_indices = cp.random.permutation(N)
        shuffled_X = X[shuffled_indices]
        shuffled_errors = errors[shuffled_indices]
        shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
        beta_1D, b_tau1D = solve_6D_CSQR(X, y, tau, h, b, shuffled_X[:N // m], shuffled_dataY[:N // m], nu=1e-4,
                                        max_iter=1000)
        results_beta_7D.append(beta_1D)
        results_b_tau_7D.append(b_tau1D)
    # 计算平均值
    average_beta_7D = cp.mean(cp.array(results_beta_7D), axis=0)
    average_b_tau_7D = cp.mean(cp.array(results_b_tau_7D), axis=0)
    # 计算偏差
    bias7D = beta_difference_2_norm(average_beta_7D, beta_true)
    # print("全局SCQR:","平均SCQR:","一步通信:","四步通信:","七步通信:",biasall, biasresults, bias1D, bias4D, bias7D)
#-----------------------------------------
    # 局部通信迭代，重复100次
    results_beta_LD = []
    results_b_tau_LD = []
    for _ in range(100):
        X_parts = [cp.random.multivariate_normal(mean, cov, size=N // m) for _ in range(m)]
        X = cp.vstack(X_parts)
        errors = cp.random.standard_t(3, size=N)
        y = X @ beta_true + errors

        shuffled_indices = cp.random.permutation(N)
        shuffled_X = X[shuffled_indices]
        shuffled_errors = errors[shuffled_indices]
        shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
        beta_LD, b_tauLD = solve_CSQR(shuffled_X[:N // m], shuffled_dataY[:N // m], tau, b, nu=1e-4, max_iter=1000)
        results_beta_LD.append(beta_LD)
        results_b_tau_LD.append(b_tauLD)
    # 计算平均值
    average_beta_LD = cp.mean(cp.array(results_beta_LD), axis=0)
    average_b_tau_LD = cp.mean(cp.array(results_b_tau_LD), axis=0)
    # 计算偏差
    biasLD = beta_difference_2_norm(average_beta_LD, beta_true)
    print("全局SCQR:", "平均SCQR:", "单机器SCQR", "一步通信:", "四步通信:", "七步通信:", biasall, biasav, biasLD, bias1D, bias4D, bias7D)