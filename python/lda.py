from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from scipy.stats import sem
from sklearn.datasets import load_breast_cancer
from sklearn.covariance import empirical_covariance
import covariance_DRO
from tqdm import tqdm
import warnings
import matlab.engine
import nl_shrinkage

NUM_EPSILONS = 50


class CustomCovarianceEstimator:
    def __init__(self, covariance_estimator_func, eps):
        """
        Custom Covariance Estimator for integration with sklearn LDA.

        :param covariance_estimator_func: A function that computes the covariance matrix.
                                          It must accept data X and hyperparameter eps.
        :param eps: Hyperparameter for the covariance estimation function.
        """
        self.covariance_estimator_func = covariance_estimator_func
        self.eps = eps
        self.covariance_ = None

    def fit(self, X, y=None):
        """
        Fit the covariance model to the data X.

        :param X: array-like, shape (n_samples, n_features)
                  Training data, where n_samples is the number of samples
                  and n_features is the number of features.
        :param y: Not used, present for API consistency by convention.
        """
        self.covariance_ = self.covariance_estimator_func(X, self.eps)
        return self


def lda_with_custom_covariance(X, y, X_t, y_t, covariance_estimator_func, eps_candidates):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    best_acc = 0
    best_eps = None

    for eps in eps_candidates:
        # Initialize our custom covariance estimator with the function and current eps
        custom_cov_estimator = CustomCovarianceEstimator(covariance_estimator_func, eps)

        # Initialize LDA with the custom covariance estimator
        lda = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=custom_cov_estimator)

        # Fit the LDA model
        lda.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = lda.predict(X_val)

        # Compute accuracy
        acc = metric_score(y_val, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_eps = eps

    # Retrain using the best eps on the entire training dataset (X, y)
    custom_cov_estimator = CustomCovarianceEstimator(covariance_estimator_func, best_eps)
    lda = LinearDiscriminantAnalysis(solver='lsqr', covariance_estimator=custom_cov_estimator)
    lda.fit(X, y)

    # Evaluate on the test set
    y_test_pred = lda.predict(X_t)
    test_acc = metric_score(y_t, y_test_pred)

    return test_acc


def evaluate_estimators(estimators, eps_ranges, dataset_loader, test_sizes):
    """
    Evaluates various covariance matrix estimators across multiple datasets.

    :param estimators: Dictionary with estimator names as keys and function handles as values.
    :param eps_ranges: Dictionary with estimator names as keys and tuples (el, eh) as values, representing the eps range.
    :param datasets: List of scikit-learn dataset loading functions.
    :return: DataFrame with rows representing datasets, columns representing estimators,
             and values representing "mean(standard error)" of accuracies over 10 runs with different train-test splits.
    """
    results = []  # Initialize results table
    dataset_names = []  # To store dataset names for the DataFrame index

    for test_size in tqdm(test_sizes):
        dataset_name = dataset_loader.__name__.replace('load_', '')
        dataset_names.append(dataset_name)
        X, y = dataset_loader(return_X_y=True)  # Load the dataset
        dataset_results = []

        for estimator_name, estimator_func in estimators.items():
            accuracies = []  # To store accuracies for this estimator on the current dataset

            for run in range(100):
                # Use "run" as the random state for reproducibility
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=run
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eps_candidates = [0,] if eps_ranges[estimator_name][0] == eps_ranges[estimator_name][1] else np.logspace(eps_ranges[estimator_name][0], eps_ranges[estimator_name][1], num=NUM_EPSILONS)
                    best_acc = lda_with_custom_covariance(
                        X_train, y_train, X_test, y_test,
                        estimator_func, eps_candidates
                    )
                accuracies.append(best_acc)

            # Compute mean accuracy and standard error
            mean_acc = np.mean(accuracies)
            std_err = sem(accuracies)  # Standard error of the mean

            # Store the result as "mean(std_err)"
            dataset_results.append(f"{mean_acc:.4f}({std_err:.4f})")

        results.append(dataset_results)

    # Convert results to a DataFrame
    df = pd.DataFrame(results, columns=estimators.keys(), index=test_sizes)
    df.index.name = 'Dataset'
    return df


def linearMethod(emp_cov, epsilon):
    mu = np.trace(emp_cov) / emp_cov.shape[0]
    cov = (1.0 - epsilon) * emp_cov
    cov.flat[:: emp_cov.shape[0] + 1] += epsilon * mu
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

    # Example usage
    estimators = {
        "Sample": lambda X, eps: getCovariance(X),
        "Linear": lambda X, eps: linearMethod(getCovariance(X), eps),
        "Wasserstein": lambda X, eps: covariance_DRO.estimate_cov(getCovariance(X), eps, 'Wasserstein'),
        "KL": lambda X, eps: covariance_DRO.estimate_cov(getCovariance(X), eps, 'KLdirect'),
        "FR": lambda X, eps: covariance_DRO.estimate_cov(getCovariance(X), eps, 'Fisher-Rao'),
        "NLLW": lambda X, eps: nl_LW.nl_shrinkage(X)
    }
    eps_ranges = {
        "Sample": (0, 0),
        "Linear": (-3, 0),
        "Wasserstein": (-3, 1),
        "KL": (-3, 1),
        "FR": (-3, 1),
        "NLLW": (0, 0)
    }

    # Evaluate the estimators across the datasets
    metric_score = accuracy_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_banknote, np.linspace(0, 1, 21)[2:-2])
    results_df.to_csv('results/lda/banknote_accuracy.csv')
    metric_score = f1_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_banknote, np.linspace(0, 1, 21)[2:-2])
    results_df.to_csv('results/lda/banknote_f1.csv')

    metric_score = accuracy_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_breast_cancer, np.linspace(0, 1, 21)[2:-5])
    results_df.to_csv('results/lda/breast_cancer_accuracy.csv')
    metric_score = f1_score
    results_df = evaluate_estimators(estimators, eps_ranges, load_breast_cancer, np.linspace(0, 1, 21)[2:-5])
    results_df.to_csv('results/lda/breast_cancer_f1.csv')
