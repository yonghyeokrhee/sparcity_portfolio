from sklearn import linear_model
import statsmodels.api as sm
from sklearn.model_selection import KFold
import numpy as np


def mean_cov_train_test(df, std, end, asset):
    batch = df[std:end][asset]
    batch = np.log(batch).diff(periods=1).iloc[1:, :] - 0.0175/252
    msk = round(len(batch)*0.7)
    training = batch[:msk]
    test = batch[msk:]
    return training, test


def plug_in(dataset, sigma_given):
    # optimal weight and max sharpe ratio
    mu = dataset.mean()
    sigma = dataset.cov()
    inv_sigma = np.linalg.inv(sigma)

    optimal_w = (sigma_given * np.matmul(inv_sigma, mu)) / np.sqrt(mu.T.dot(inv_sigma).dot(mu))
    max_sharpe = np.sqrt(mu.T.dot(inv_sigma).dot(mu))
    return optimal_w, max_sharpe


def rc(dataset, sigma_given):
    max_sharpe = plug_in(dataset, sigma_given)[1]
    return 0.04*(1+max_sharpe**2)/max_sharpe


def mn_gen(mu, sigma):
    oneset = np.random.multivariate_normal(mu, sigma, 120)
    return oneset


def val_result(val: object, coef: object) -> object:
    std = np.matmul(val, coef).std()
    sharpe = np.matmul(val, coef).mean()/std

    return sharpe, std


def score(std):
    diff = abs(0.04 - std)
    return diff


def lasso_coef(rc_value, lambda_, train):
    y = rc_value*np.ones(len(train))
    clf = linear_model.Lasso(fit_intercept=False, alpha=lambda_)
    result = clf.fit(np.array(train)*100, y*100)
    temp = result.coef_
    return temp


def cross_validation(rc_value: object, lambda_: object, training: object) -> object:
    empty_list = []
    sharpe_list = []
    kf = KFold(n_splits=10)

    for train, val in kf.split(training):
        temp_set = training.iloc[train]
        val_set = training.iloc[val]
        val_tuple = val_result(val_set,lasso_coef(rc_value, lambda_, temp_set))
        empty_list.append(score(val_tuple[1]))
        sharpe_list.append(val_tuple[0])
    return np.array(empty_list).mean(), np.array(sharpe_list).mean()


if __name__ == "__main__":
    pass