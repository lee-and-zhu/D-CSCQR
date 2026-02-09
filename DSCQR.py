import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from tqdm import tqdm  # 引入进度条库
import statsmodels.api as sm
from joblib import Parallel, delayed
# 计算回归指标
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)
def beta_difference_2_norm(beta1, beta2):
    # 计算两个β向量的差
    difference = np.array(beta1) - np.array(beta2)
    # 计算差向量的二范数
    norm = np.linalg.norm(difference, 2)
    return norm
# 核平滑函数，支持不同核函数
def K_tilde(u, kernel_type='gaussian'):
    if kernel_type == 'gaussian':
        return np.exp(-0.5 * u ** 2) / np.sqrt(2 * np.pi)
    elif kernel_type == 'epanechnikov':
        return 0.75 * (1 - u ** 2) * (np.abs(u) <= 1)
    else:
        raise ValueError("不支持的核函数类型")
# 核函数 K_h(u) 的计算 (缩放核函数)
def K_h(u, h):
    """
    计算缩放后的核函数 K_h(u)
    :param u: 输入值
    :param h: 平滑参数 (带宽)
    :return: 缩放核函数值 K_h(u)
    """
    return K_tilde(u / h) / h
# 核函数的积分形式 (累积核函数)

# 原始的 K_bar 函数，接受单个标量 u
def K_bar_scalar(u):
    return norm.cdf(u)
# 修改后的 K_bar 函数，支持标量和数组
def K_bar(u):
    """
    对标量或数组计算核的积分形式 K_bar。
    :param u: 标量或数组
    :return: 与 u 形状一致的累计核值
    """
    return np.vectorize(K_bar_scalar)(u)


def compute_gradients(X_batch, y_batch, beta_h, b_tau, tau, h):
    n, K = X_batch.shape[0], len(tau)

    # 计算误差矩阵 U (n x K)
    U = -(y_batch[:, np.newaxis] - X_batch @ beta_h[:, np.newaxis] - b_tau[np.newaxis, :]) / h

    # 计算权重矩阵 W (n x K) 使用累积核函数 K_bar
    W = K_bar(U)

    # 计算梯度 grad_beta_h 和 grad_b_tau
    grad_beta_h = X_batch.T @ (W - tau) @ np.ones(K)
    grad_b_tau = np.sum(W - tau, axis=0)

    return grad_beta_h, grad_b_tau
def gradient(beta_h, b_tau, X, y, tau, h, n_jobs=-1):
    """
    计算梯度的向量化版本，并行化处理。
    """
    # 将数据分成小批次
    batch_size = X.shape[0] // 4  # 假设分成4个批次
    batches = [(X[i:i + batch_size], y[i:i + batch_size]) for i in range(0, X.shape[0], batch_size)]

    # 使用joblib并行计算每个批次的梯度
    results = Parallel(n_jobs=n_jobs)(delayed(compute_gradients)(X_batch, y_batch, beta_h, b_tau, tau, h) for X_batch, y_batch in batches)

    # 汇总结果
    grad_beta_h = np.sum([result[0] for result in results], axis=0)
    grad_b_tau = np.sum([result[1] for result in results], axis=0)

    # Normalize gradients
    n_total = X.shape[0]
    K = len(tau)
    grad_beta_h /= n_total * K
    grad_b_tau /= n_total * K
    return grad_beta_h, grad_b_tau
# 目标函数的梯度,快速算法 (平滑分位数回归)
def gradient1(beta_h, b_tau, X, y, tau, h):
    """
    计算梯度的向量化版本。
    """
    n, K = X.shape[0], len(tau)

    # 计算误差矩阵 U (n x K)
    U = -(y[:, np.newaxis] - X @ beta_h[:, np.newaxis] - b_tau[np.newaxis, :]) / h

    # 计算权重矩阵 W (n x K) 使用累积核函数 K_bar
    W = K_bar(U)

    # 计算梯度 grad_beta_h 和 grad_b_tau
    # grad_beta_h: sum over (W - tau) * X
    grad_beta_h = X.T @ (W - tau) @ np.ones(K)

    # grad_b_tau: sum over (W - tau)
    grad_b_tau = np.sum(W - tau, axis=0)

    # Normalize gradients
    grad_beta_h /= n * K
    grad_b_tau /= n * K

    return grad_beta_h, grad_b_tau
#初代梯度，强行循环计算
def gradient2(beta_h, b_tau, X, y, tau, h):
    n, K = X.shape[0], len(tau)
    grad_beta_h = np.zeros_like(beta_h)
    grad_b_tau = np.zeros_like(b_tau)

    for i in range(n):
        for k in range(K):#循环可能有问题
            u = -(y[i] - X[i] @ beta_h - b_tau[k])/ h
            weight = K_bar(u)  # 使用 K_bar_h(u, h) 而不是 K_tilde
            grad_beta_h += (weight -tau[k]) * X[i]
            grad_b_tau[k] += (weight -tau[k])

    grad_beta_h /= n*K
    grad_b_tau /= n*K#加了K后就能跑了
    return grad_beta_h, grad_b_tau
#定义移位损失的梯度
def shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta0, b_tau0, X1, y1):
    #grad_beta_h, grad_b_tau = gradient(beta0, b_tau0, X, y, tau, h)#全局梯度
    # 使用并行计算梯度
    results = Parallel(n_jobs=-1)(
        delayed(gradient)(beta, b_tau, X, y, tau, h) for beta in [beta0]
    )
    grad_beta_h, grad_b_tau = results[0]
    grad_beta_b, grad_b_taub = gradient(beta_h, b_tau, X1, y1, tau, b)
    grad_beta_b0, grad_b_taub0 = gradient(beta0, b_tau0, X1, y1, tau, b)#局部梯度
    grad_shift_beta = grad_beta_b - grad_beta_b0 + grad_beta_h
    grad_shift_b_tau = grad_b_taub - grad_b_taub0 + grad_b_tau
    return grad_shift_beta, grad_shift_b_tau
# Barzilai-Borwein 步长计算函数
def BB_step(delta, eta):
    lambda_1 = np.dot(delta, delta) / np.dot(delta, eta)
    lambda_2 = np.dot(delta, eta) / np.dot(eta, eta)
    if lambda_1 > 0:
        return min(lambda_1, lambda_2, 40)
    else:
        return 1.0
# 使用梯度下降和 BB 步长求解 CSQR 估计
def solve_CSQR(X, y, tau, h, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)

    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = gradient(beta_h, b_tau, X, y, tau, h)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        #print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    beta_true = np.ones(p)
    # print(beta_difference_2_norm(beta_h, beta_true))
    return beta_h, b_tau
#第一步迭代中使用梯度下降计算估计量
def solve_D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_CSQR(X1, Y1, tau, b, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="DCSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        #print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
#后面第1次迭代中使用梯度下降计算估计量
def solve_1D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="D1CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_2D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_1D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="D2CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_3D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_2D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="D3CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_4D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_3D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="D4CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_5D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_4D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="D5CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau
def solve_6D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000):
    n, p = X.shape
    M = len(tau)

    # 初始化
    beta_h = np.zeros(p)
    b_tau = np.zeros(M)
    beta_0, b_tau0 = solve_5D_CSQR(X, y, tau, h, b, X1, Y1, nu=1e-4, max_iter=1000)
    beta_h_prev = beta_h.copy()
    b_tau_prev = b_tau.copy()

    # 第一步梯度下降
    grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
    beta_h -= grad_beta_h
    b_tau -= grad_b_tau

    # 迭代过程，加入进度条
    for t in tqdm(range(1, max_iter + 1), desc="D6CSQR 迭代进度"):
        # 计算增量 delta 和梯度差异 eta
        delta = np.concatenate([(beta_h - beta_h_prev), (b_tau - b_tau_prev)])
        grad_beta_h, grad_b_tau = shiftgradient(beta_h, b_tau, X, y, tau, h, b, beta_0, b_tau0, X1, Y1)
        grad_combined = np.concatenate([grad_beta_h, grad_b_tau])

        # 计算 BB 步长
        step_size = BB_step(delta, grad_combined)

        # 更新 beta_h 和 b_tau
        beta_h_prev = beta_h.copy()
        b_tau_prev = b_tau.copy()
        beta_h -= step_size * grad_beta_h
        b_tau -= step_size * grad_b_tau
        # print(step_size, beta_h, b_tau)
        # 判断收敛条件
        if np.linalg.norm(beta_h - beta_h_prev) < nu:
            print(f"在 {t} 次迭代后收敛。")
            break
    else:
        print("达到最大迭代次数，未完全收敛。")
    return beta_h, b_tau








# 示例用法
if __name__ == "__main__":
    #np.random.seed(42)
    # 生成合成数据
    n, p, m = 300, 10, 100
    N = n*m
    nu = 1e-4
    # 设置均值向量和协方差矩阵
    mean = np.zeros(p)  # p维均值向量，均值为0
    def generate_covariance_matrix(size):
        # 初始化协方差矩阵
        covariance_matrix = np.zeros((size, size))

        # 填充协方差矩阵
        for j in range(size):
            for k in range(size):
                covariance_matrix[j, k] = 2*(0.5 ** abs(j - k))

        return covariance_matrix
    cov = generate_covariance_matrix(p)
    # 生成 n 个 p 维的多元正态随机数
    X = np.random.multivariate_normal(mean, cov, size=N)
    beta_true = np.ones(p)
    errors = np.random.standard_t(3, size=N)
    y = X @ beta_true + errors
    # 分位数水平和参数
    tau = np.linspace(0.05, 0.95, 19)  # tau 从 0 到 1 均匀划分为 20 等份
    # 计算 h
    b = (p + np.log(n)) / n
    b = b ** (1 / 2)  # 开二分之一次方
    h = (p + np.log(N)) / N
    h = h ** (1 / 2)  # 开二分之一次方
    print(h,b,"带宽")
    print(X.shape)
    #定义分布式，将总体随机分配到m个机器
    # 随机打乱数据
    shuffled_indices = np.random.permutation(N)
    shuffled_X = X[shuffled_indices]
    shuffled_errors = errors[shuffled_indices]
    shuffled_dataY = shuffled_X @ beta_true + shuffled_errors
    # 将数据分成m份
    split_dataX = np.array_split(shuffled_X, m)
    split_dataErrors = np.array_split(shuffled_errors, m)
    # 为每一份定义一个向量并存储它们
    distubitedX = [part for part in split_dataX]
    distubitedErrors = [part for part in split_dataErrors]
    # 初始化结果列表
    distubitedY = []
    for i in range(m):
        distubitedY0 = distubitedX[i] @ beta_true + distubitedErrors[i]
        distubitedY.append(distubitedY0)



    #所有分布看成整体,求解出all beta
    beta_hall, b_tauall = solve_CSQR(X, y, tau, h, nu)  # 全局#所有分布看成整体
    biasall = beta_difference_2_norm(beta_hall, beta_true)
    #计算每个机器的平均，使用快速算法
    def solve_local_CSQR(i, X, y, tau, h, nu):
        beta_hloc, b_tauloc = solve_CSQR(X, y, tau, h, nu)
        return beta_hloc, b_tauloc
    # 并行计算
    results = Parallel(n_jobs=-1)(
        delayed(solve_local_CSQR)(i, distubitedX[i], distubitedY[i], tau, b, nu) for i in range(m))
    # 分离结果
    results_beta = np.array([res[0] for res in results])
    results_b_tau = np.array([res[1] for res in results])
    sums = np.sum(results_beta, axis=0)
    results_beta = sums / len(results_beta)
    biasresults = beta_difference_2_norm(results_beta, beta_true)

    #一步通信迭代
    beta_1D, b_tau1D = solve_D_CSQR(X, y, tau, h, b, distubitedX[0], distubitedY[0], nu=1e-4, max_iter=1000)
    bias1D = beta_difference_2_norm(beta_1D, beta_true)

    #三步通信迭代
    beta_4D, b_tau4D = solve_3D_CSQR(X, y, tau, h, b, distubitedX[0], distubitedY[0], nu=1e-4, max_iter=1000)
    bias4D = beta_difference_2_norm(beta_4D, beta_true)

    #五步通信迭代
    beta_7D, b_tau7D = solve_6D_CSQR(X, y, tau, h, b, distubitedX[0], distubitedY[0], nu=1e-4, max_iter=1000)
    bias7D = beta_difference_2_norm(beta_7D, beta_true)
    print("全局SCQR:","平均SCQR:","一步通信:","四步通信:","七步通信:",biasall, biasresults, bias1D, bias4D, bias7D)
#-----------------------------------------
    #局部通信迭代
    beta_LD, b_tauLD = solve_CSQR(distubitedX[0],  distubitedY[0], tau, b, nu=1e-4, max_iter=1000)
    bias1D = beta_difference_2_norm(beta_1D, beta_true)