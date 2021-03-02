import numpy as np
import pandas as pd
from OLPSResult import OLPSResult
from Strategy import Strategy


def simplex_projection(v: np.ndarray, b) -> np.ndarray:
    """projection onto simplex of specified radius returns the solution
    to the following constrained minimization problem:
    min   ||w - v||_2
    s.t.  sum(w) <= b, w >= 0.

    That is, performs Euclidean projection of v to the positive simplex of
    radius b.
    """
    if b < 0:
        raise ValueError('Radius of simplex is negative!', b)

    v_copy = v.copy().flatten()
    for i in range(v.size):
        if v_copy[i] < 0:
            v_copy[i] = 0
    v_copy = v_copy.reshape(v.shape)
    v = v_copy * v  # element-wise multiplication
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    fraction = np.divide(sv - b, np.arange(1, u.size + 1).transpose()). \
        reshape(u.shape)
    bool_comparison = list(np.flip(np.greater(u, fraction)).flatten().T)
    # a list of boolean matrix/vector in reverse order
    rho = bool_comparison.index(True)
    theta = max(0, (sv[rho] - b) / rho)
    return_val = (v - theta).flatten()
    for i in range(len(return_val)):
        if return_val[i] < 0:
            return_val[i] = 0
    return return_val

    # OLMAR-1 has only one expert i.e. portfolio manager, so we go expert
    # directly. This helper function generates


def olmar1_expert(df: pd.DataFrame, weight_o: np.array,
                  epsilon, W) -> any:
    time = df.shape[0]
    num = df.shape[1]

    if time < W + 1:
        data_phi = df.to_numpy()[time - 1]
        # extract the tth row
    else:
        data_phi = np.zeros((1, num))
        tmp_x = np.ones((1, num))
        for i in range(0, W):
            data_phi = data_phi + np.divide(1, tmp_x)
            tmp_x = tmp_x * df.to_numpy()[time - i - 1]  # index TBC
        data_phi = data_phi * (1 / W)

    # suffer loss
    ell = max(0, epsilon - np.matmul(data_phi, weight_o))  # a number

    # set parameters
    x_bar = np.mean(data_phi)
    demean = data_phi - x_bar
    denominator = np.matmul(demean, demean.transpose())  # a number
    if denominator != 0.0:
        lambda1 = ell / denominator
    else:
        lambda1 = 0

    # update portfolio
    weight = weight_o + np.matmul(lambda1 * np.ones(data_phi.shape),
                                  (data_phi.transpose() - x_bar))

    # Normalize portfolio
    weight = simplex_projection(weight, 1)
    return weight


class OLMAR1(Strategy):

    def __init__(self):
        pass

    def run(self, df: pd.DataFrame):
        # Extract the parameters
        # epsilon =
        # W =

        # get number of days and number of stocks
        n = df.shape[0]
        m = df.shape[1]

        # initialization
        cum_ret = 1
        cumprod_ret = np.ones((n, 1))
        daily_ret = np.ones(n)

        # portfolio weights, starting with uniform portfolio
        day_weight = np.ones((m, 1)) / m
        day_weight_o = np.zeros((m, 1))  # Last closing price adjusted portfolio
        daily_portfolio = np.zeros((n, m))

        # Trading

        for t in range(0, n):
            # Step 1: Receive stock price relatives
            if t >= 2:
                day_weight = olmar1_expert(df[0:t], day_weight, epsilon=10,
                                           W=5)
            # Normalize the constraint, always useless
            day_weight = np.divide(day_weight, sum(day_weight))
            daily_portfolio[t] = day_weight.transpose()

            # Step 2: calculate daily and total returns on day t
            daily_ret[t] = df.to_numpy()[t].dot(day_weight)
            cum_ret = cum_ret * daily_ret[t]
            cumprod_ret[t] = cum_ret

            # Adjust weight for the transaction cost issue
            # day_weight_o = np.multiply(day_weight, df[t].transpose()) /
            # daily_ret[t, 0]
        ts_daily_return = pd.Series(index=df.index,
                                    data=daily_ret.transpose())
        return OLPSResult(ts_daily_return)

    def name(self):
        return 'OnLine Moving Average Reversion1'


def olmar2_expert(df, data_phi, weight_o, epsilon, alpha) -> (
        any, any):
    time = df.shape[0]
    num = df.shape[1]
    data_phi = alpha + np.divide((1 - alpha) * data_phi,
                                 df.to_numpy()[time - 1])
    # ./ denotes element-wise right division
    # suffer loss
    ell = max(0, epsilon - np.matmul(data_phi, weight_o))

    # set parameters
    x_bar = np.mean(data_phi)
    demean = data_phi - x_bar
    denominator = np.matmul(demean, demean.transpose())
    # denominator is a number
    if denominator != 0.0:
        lambda1 = float(ell / denominator)  # as a number, not a ndarray
    else:
        lambda1 = 0

    # update portfolio
    weight = weight_o + np.matmul(lambda1 * np.ones(data_phi.shape),
                                  (data_phi.transpose() - x_bar))

    # Normalize portfolio
    weight = simplex_projection(weight, 1)
    return weight, data_phi


class OLMAR2(Strategy):

    def __init__(self):
        pass

    def run(self, df: pd.DataFrame):
        # get number of days and number of stocks
        n = df.shape[0]
        m = df.shape[1]

        # initialization
        cum_ret = 1
        cumprod_ret = np.ones((n, 1))
        daily_ret = np.ones(n)
        data_phi = np.ones((1, m))

        # portfolio weights, starting with uniform portfolio
        day_weight = np.ones((m, 1)) / m
        day_weight_o = np.zeros((m, 1))  # Last closing price adjusted portfolio
        daily_portfolio = np.zeros((n, m))

        # trading
        for i in range(0, n):
            # step 1: receive stock price relatives
            if i >= 1:
                day_weight, data_phi = olmar2_expert(df[0:i], data_phi,
                                                     day_weight, epsilon=10,
                                                     alpha=0.5)
            # normalize the constraint
            day_weight = np.divide(day_weight, sum(day_weight))
            daily_portfolio[i] = day_weight.transpose()

            # step 2: calculate daily and total returns on day t
            daily_ret[i] = df.to_numpy()[i].dot(day_weight)
            cum_ret = cum_ret * daily_ret[i]
            cumprod_ret[i] = cum_ret

        ts_daily_return = pd.Series(index=df.index,
                                    data=daily_ret.transpose())
        return OLPSResult(ts_daily_return)

    def name(self):
        return 'OnLine Moving Average Reversion2'
