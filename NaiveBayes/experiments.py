import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from gaussian_nb import GaussianNB
from multinomial_nb import MultinomialNB
from metrics import accuracy


def load_gaussian_datasets(test_size=0.3, random_state=42):
    datasets = {}
    for loader, name in [
        (load_iris, "Iris"),
        (load_wine, "Wine"),
        (load_breast_cancer, "Breast Cancer"),
    ]:
        data = loader()
        X = StandardScaler().fit_transform(data.data)
        y = data.target
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        datasets[name] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target_names": [str(t) for t in data.target_names],
        }
    return datasets


def load_multinomial_datasets(test_size=0.3, random_state=42):
    datasets = {}

    digits = load_digits()
    Xd = digits.data
    yd = digits.target
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        Xd,
        yd,
        test_size=test_size,
        random_state=random_state,
        stratify=yd,
    )
    datasets["Digits"] = {
        "X_train": Xd_train,
        "X_test": Xd_test,
        "y_train": yd_train,
        "y_test": yd_test,
        "target_names": [str(i) for i in digits.target_names],
    }

    wine = load_wine()
    Xw = MinMaxScaler().fit_transform(wine.data)
    Xw = np.floor(Xw * 20.0)
    yw = wine.target
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        Xw,
        yw,
        test_size=test_size,
        random_state=random_state,
        stratify=yw,
    )
    datasets["Wine (Binned)"] = {
        "X_train": Xw_train,
        "X_test": Xw_test,
        "y_train": yw_train,
        "y_test": yw_test,
        "target_names": [str(t) for t in wine.target_names],
    }

    return datasets


def run_gaussian_experiments(var_smoothing_values=(1e-11, 1e-10, 1e-9, 1e-8, 1e-7)):
    datasets = load_gaussian_datasets()
    results = {}

    for name, d in datasets.items():
        X_train, X_test = d["X_train"], d["X_test"]
        y_train, y_test = d["y_train"], d["y_test"]

        sweep = {}
        best_value = None
        best_test_acc = -1.0
        best_model = None

        for vs in var_smoothing_values:
            model = GaussianNB(var_smoothing=vs)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            sweep[vs] = {"train_acc": train_acc, "test_acc": test_acc}

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_value = vs
                best_model = model

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        results[name] = {
            "algorithm": "GaussianNB",
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target_names": d["target_names"],
            "sweep": sweep,
            "best_param": best_value,
            "best_model": best_model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "test_acc": accuracy(y_test, y_pred),
        }

    return results


def run_multinomial_experiments(alpha_values=(0.1, 0.5, 1.0, 2.0, 5.0)):
    datasets = load_multinomial_datasets()
    results = {}

    for name, d in datasets.items():
        X_train, X_test = d["X_train"], d["X_test"]
        y_train, y_test = d["y_train"], d["y_test"]

        sweep = {}
        best_value = None
        best_test_acc = -1.0
        best_model = None

        for alpha in alpha_values:
            model = MultinomialNB(alpha=alpha)
            model.fit(X_train, y_train)
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)
            sweep[alpha] = {"train_acc": train_acc, "test_acc": test_acc}

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_value = alpha
                best_model = model

        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)

        results[name] = {
            "algorithm": "MultinomialNB",
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "target_names": d["target_names"],
            "sweep": sweep,
            "best_param": best_value,
            "best_model": best_model,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "test_acc": accuracy(y_test, y_pred),
        }

    return results


def run_all():
    all_results = {}

    gaussian_results = run_gaussian_experiments()
    for name, res in gaussian_results.items():
        all_results[f"{name} - GaussianNB"] = res

    multinomial_results = run_multinomial_experiments()
    for name, res in multinomial_results.items():
        all_results[f"{name} - MultinomialNB"] = res

    return all_results
