import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_auc_score
import mylogging


def probability(cov1, cov2, m1, m2, p1, p2, X):
    Px1 = p1 * multivariate_normal(mean=m1, cov=cov1).pdf(X)
    Px2 = p2 * multivariate_normal(mean=m2, cov=cov2).pdf(X)
    return Px2 / (Px1 + Px2)


def cv_estimate_qda(X, y, method, candidates, n_folds=20):
    best_e, best_accuracy = None, -1
    for e1 in candidates:
        for e2 in candidates:
            accuracies, ns = [], []
            kf = KFold(n_splits=n_folds)
            for train_index, test_index in kf.split(X):
                train_X, test_X = X[train_index], X[test_index]
                train_y, test_y = y[train_index], y[test_index]
                # find the covariance matrices
                C0, l0 = method(train_X[train_y == 0], e1)
                C1, l1 = method(train_X[train_y == 1], e2)
                p1 = train_y.mean()
                p0 = 1 - p1
                try:
                    accuracies.append(
                        roc_auc_score(
                            test_y,
                            np.nan_to_num(probability(C0, C1, l0, l1, p0, p1, test_X)),
                        )
                    )
                except np.linalg.LinAlgError as err:
                    mylogging.log_error(str(err))
                    mylogging.log_error("e1 {}".format(e1))
                    mylogging.log_error("e2 {}".format(e2))
                    accuracies.append(-np.inf)
                ns.append(len(test_y))
            accuracy = (np.array(accuracies) * np.array(ns)).sum() / np.sum(ns)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_e = (e1, e2)
    p1 = y.mean()
    p0 = 1 - p1
    C0, l0 = method(X[y == 0], best_e[0])
    C1, l1 = method(X[y == 1], best_e[1])
    return C0, l0, p0, C1, l1, p1

