from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.datasets import load_breast_cancer
from sklearn.covariance import empirical_covariance
import covariance_DRO
from tqdm import tqdm
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from scipy.stats import multivariate_normal
import matlab.engine
import nl_shrinkage

NUM_EPSILONS = 50


class CustomQDA(BaseEstimator, ClassifierMixin):
    def __init__(self, covariance_estimator, eps):
        self.covariance_estimator = covariance_estimator
        self.eps = eps

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.label_encoder_ = LabelEncoder().fit(y)
        y_encoded = self.label_encoder_.transform(y)

        self.means_ = []
        self.covariances_ = []
        self.priors_ = []

        for group in self.classes_:
            Xg = X[y_encoded == group, :]
            self.means_.append(np.mean(Xg, axis=0))
            self.covariances_.append(self.covariance_estimator(Xg, self.eps))
            self.priors_.append(len(Xg) / len(X))

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        log_likelihoods = np.array([
            multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=True) + np.log(prior)
            for mean, cov, prior in zip(self.means_, self.covariances_, self.priors_)
        ])

        predictions = np.argmax(log_likelihoods, axis=0)
        return self.label_encoder_.inverse_transform(predictions)


def qda_with_custom_covariance(X, y, X_t, y_t, covariance_estimator_func, eps_candidates):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_acc = 0
    best_eps = None

    for eps in eps_candidates:
        custom_qda = CustomQDA(covariance_estimator_func, eps)
        custom_qda.fit(X_train, y_train)
        y_pred = custom_qda.predict(X_val)

        acc = metric_score(y_val, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_eps = eps

    custom_qda = CustomQDA(covariance_estimator_func, best_eps)
    custom_qda.fit(X, y)
    y_test_pred = custom_qda.predict(X_t)
    test_acc = metric_score(y_t, y_test_pred)

    return test_acc


def evaluate_estimators(estimators, eps_ranges, dataset_loader, test_sizes):
    results = []
    dataset_names = []

    for test_size in tqdm(test_sizes):
        dataset_name = dataset_loader.__name__.replace('load_', '')
        dataset_names.append(dataset_name)
        X, y = dataset_loader(return_X_y=True)
        dataset_results = []

        for estimator_name, estimator_func in estimators.items():
            accuracies = []

            for run in range(100):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=run
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eps_candidates = [0,] if eps_ranges[estimator_name][0] == eps_ranges[estimator_name][1] else np.logspace(eps_ranges[estimator_name][0], eps_ranges[estimator_name][1], num=NUM_EPSILONS)
                    best_acc = qda_with_custom_covariance(
                        X_train, y_train, X_test, y_test,
                        estimator_func, eps_candidates
                    )
                accuracies.append(best_acc)
            mean_acc = np.mean(accuracies)
            std_err = sem(accuracies)

            dataset_results.append(f"{mean_acc:.4f}({std_err:.4f})")

        results.append(dataset_results)

    df = pd.DataFrame(results, columns=estimators.keys(), index=test_sizes)
    df.index.name = 'Test Size'
    return df


# Define the covariance estimation functions
def linearMethod(emp_cov, epsilon):
    mu = np.trace(emp_cov) / emp_cov.shape[0]
    cov = (1.0 - epsilon) * emp_cov
    cov.flat[::emp_cov.shape[0] + 1] += epsilon * mu
    return cov


def getCovariance(X):
    return empirical_covariance(X, assume_centered=False)


def load_banknote(**kwargs):
    df = pd.read_csv('data_banknote_authentication.txt', header=None, names=['X1', 'X2', 'X3', 'X4', 'Y'])
    X = df.drop('Y', axis=1).to_numpy()
    Y = df['Y'].to_numpy()
    return X, Y


if __name__ == '__main__':
    # start matlab for nllw
    eng = matlab.engine.start_matlab()
    eng.addwise(nargout=0)

    def numpy_to_matlab(array):
        return matlab.double(array.tolist())
    nl_LW = nl_shrinkage.MatlabShrinkage()

    # Define estimators
    estimators = {
        "Sample": lambda X, eps: getCovariance(X),
        "Linear": lambda X, eps: linearMethod(getCovariance(X), eps),
        "Wasserstein": lambda X, eps: covariance_DRO.estimate_cov(getCovariance(X), eps, 'Wasserstein'),
        "KL": lambda X, eps: covariance_DRO.estimate_cov(getCovariance(X), eps, 'KLdirect'),
        "FR": lambda X, eps: covariance_DRO.estimate_cov(getCovariance(X), eps, 'Fisher-Rao'),
        "NLLW": lambda X, eps: nl_LW.nl_shrinkage(X)
    }
    # Define eps ranges for each estimator
    eps_ranges = {
        "Sample": (0, 0),
        "Linear": (-3, 0),
        "Wasserstein": (-3, 1),
        "KL": (-3, 1),
        "FR": (-3, 1),
        "NLLW": (0, 0)
    }
    # Specify the datasets to evaluate on

    # Evaluate the estimators across the datasets
    metric_score = accuracy_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_banknote, np.linspace(0, 1, 21)[2:-2])
    results_df.to_csv('results/qda/banknote_accuracy.csv')
    metric_score = f1_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_banknote, np.linspace(0, 1, 21)[2:-2])
    results_df.to_csv('results/qda/banknote_f1.csv')

    metric_score = accuracy_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_breast_cancer, np.linspace(0, 1, 21)[2:-5])
    results_df.to_csv('results/qda/breast_cancer_accuracy.csv')
    metric_score = f1_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_breast_cancer, np.linspace(0, 1, 21)[2:-5])
    results_df.to_csv('results/qda/breast_cancer_f1.csv')
