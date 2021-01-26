import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import multivariate_normal
from tqdm import tqdm

import qda_methods
import cross_validation

from warnings import filterwarnings
import os
import mylogging


def probability(cov1, cov2, m1, m2, p1, p2, X):
    Px1 = p1 * multivariate_normal(mean=m1, cov=cov1).pdf(X)
    Px2 = p2 * multivariate_normal(mean=m2, cov=cov2).pdf(X)
    return np.nan_to_num(Px2 / (Px1 + Px2))


def test_QDA(file, name, n_repetitions=100, n_candidates=50):
    # load the dataset
    X, y = load_svmlight_file(file)
    # format y
    y = (y + 1) / 2
    # data holders for results
    accuracies = {
        "Empirical": [],
        "LedoitWolf": [],
        "Linear": [],
        "LinearDiagonal": [],
        "Wasserstein": [],
        "KL": [],
        "Fisher-Rao": [],
    }

    aucs = {
        "Empirical": [],
        "LedoitWolf": [],
        "Linear": [],
        "LinearDiagonal": [],
        "Wasserstein": [],
        "KL": [],
        "Fisher-Rao": [],
    }

    # repeated testing
    for i in tqdm(range(n_repetitions)):
        train_X, test_X, train_y, test_y = train_test_split(
            X.toarray(), y, test_size=0.25, random_state=i, shuffle=True
        )

        # results for Emprical Covariance
        p1 = train_y.mean()
        p0 = 1 - p1
        c0 = EmpiricalCovariance(assume_centered=False).fit(train_X[train_y == 0])
        c1 = EmpiricalCovariance(assume_centered=False).fit(train_X[train_y == 1])
        aucs["Empirical"].append(
            roc_auc_score(
                test_y,
                probability(
                    c0.covariance_,
                    c1.covariance_,
                    c0.location_,
                    c1.location_,
                    p0,
                    p1,
                    test_X,
                ),
            )
        )
        accuracies["Empirical"].append(
            accuracy_score(
                test_y,
                (
                    probability(
                        c0.covariance_,
                        c1.covariance_,
                        c0.location_,
                        c1.location_,
                        p0,
                        p1,
                        test_X,
                    )
                    >= 0.5
                ).astype(float),
            )
        )

        # results for Ledoit Wolf
        p1 = train_y.mean()
        p0 = 1 - p1
        c0 = LedoitWolf(assume_centered=False).fit(train_X[train_y == 0])
        c1 = EmpiricalCovariance(assume_centered=False).fit(train_X[train_y == 1])
        aucs["LedoitWolf"].append(
            roc_auc_score(
                test_y,
                probability(
                    c0.covariance_,
                    c1.covariance_,
                    c0.location_,
                    c1.location_,
                    p0,
                    p1,
                    test_X,
                ),
            )
        )
        accuracies["LedoitWolf"].append(
            accuracy_score(
                test_y,
                (
                    probability(
                        c0.covariance_,
                        c1.covariance_,
                        c0.location_,
                        c1.location_,
                        p0,
                        p1,
                        test_X,
                    )
                    >= 0.5
                ).astype(float),
            )
        )

        # results for Linear
        mylogging.log_error("Linear")
        C0, l0, p0, C1, l1, p1 = cross_validation.cv_estimate_qda(
            train_X,
            train_y,
            qda_methods.linear_shrinkage,
            np.logspace(-5, 0, n_candidates),
        )
        aucs["Linear"].append(
            roc_auc_score(test_y, probability(C0, C1, l0, l1, p0, p1, test_X))
        )
        accuracies["Linear"].append(
            accuracy_score(
                test_y,
                (probability(C0, C1, l0, l1, p0, p1, test_X,) >= 0.5).astype(float),
            )
        )

        # results for Linear Diagonal
        mylogging.log_error("LinearDiagonal")
        C0, l0, p0, C1, l1, p1 = cross_validation.cv_estimate_qda(
            train_X,
            train_y,
            qda_methods.linear_shrinkage2,
            np.logspace(-5, 0, n_candidates),
        )
        aucs["LinearDiagonal"].append(
            roc_auc_score(test_y, probability(C0, C1, l0, l1, p0, p1, test_X))
        )
        accuracies["LinearDiagonal"].append(
            accuracy_score(
                test_y,
                (probability(C0, C1, l0, l1, p0, p1, test_X,) >= 0.5).astype(float),
            )
        )

        # results for Wasserstein
        mylogging.log_error("Wasserstein")
        C0, l0, p0, C1, l1, p1 = cross_validation.cv_estimate_qda(
            train_X, train_y, qda_methods.wasserstein, np.logspace(-2, 2, n_candidates)
        )
        aucs["Wasserstein"].append(
            roc_auc_score(test_y, probability(C0, C1, l0, l1, p0, p1, test_X))
        )
        accuracies["Wasserstein"].append(
            accuracy_score(
                test_y,
                (probability(C0, C1, l0, l1, p0, p1, test_X,) >= 0.5).astype(float),
            )
        )

        # results for KL
        mylogging.log_error("KL")
        C0, l0, p0, C1, l1, p1 = cross_validation.cv_estimate_qda(
            train_X, train_y, qda_methods.KL, np.logspace(-1, 3, n_candidates)
        )
        aucs["KL"].append(
            roc_auc_score(test_y, probability(C0, C1, l0, l1, p0, p1, test_X))
        )
        accuracies["KL"].append(
            accuracy_score(
                test_y,
                (probability(C0, C1, l0, l1, p0, p1, test_X,) >= 0.5).astype(float),
            )
        )

        # results for Fisher-Rao
        mylogging.log_error("Fisher-Rao")
        C0, l0, p0, C1, l1, p1 = cross_validation.cv_estimate_qda(
            train_X, train_y, qda_methods.fisher_rao, np.logspace(-2, 3, n_candidates)
        )
        aucs["Fisher-Rao"].append(
            roc_auc_score(test_y, probability(C0, C1, l0, l1, p0, p1, test_X))
        )
        accuracies["Fisher-Rao"].append(
            accuracy_score(
                test_y,
                (probability(C0, C1, l0, l1, p0, p1, test_X,) >= 0.5).astype(float),
            )
        )

    # save the results
    for method in aucs.keys():
        np.savetxt("QDA_results/{}_{}.txt".format(name, method), np.array(aucs[method]))

    for method in accuracies.keys():
        np.savetxt(
            "QDA_results/{}_{}_accuracy.txt".format(name, method),
            np.array(accuracies[method]),
        )

    # display the results
    for method in aucs.keys():
        print("{} : \t{:.6f}".format(method, np.mean(aucs[method])))


if __name__ == "__main__":
    filterwarnings("ignore")
    for filename in os.listdir("datasets"):
        name = filename.split(".")[0]
        print("Running for {}".format(name))
        mylogging.log_error("-" * 20)
        mylogging.log_error(name)
        mylogging.log_error("-" * 20)
        test_QDA(
            "datasets/{}".format(filename), name, n_repetitions=20, n_candidates=50
        )

