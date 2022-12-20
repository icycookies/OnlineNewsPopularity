from model import LogisticRegression
from data import load_data, random_split, split_by_time
from SVM_model import SVM
from PCA_KNN_model import PCA_KNN
from GaussianNB_model import naive_bayes
from FNN_model import FNN
from sklearn import svm
from sklearn.utils.estimator_checks import check_estimator
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import argparse
import random
import numpy as np
import pickle
# from sklearn.linear_model import LogisticRegression


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_args():
    parser = argparse.ArgumentParser()
    # which model to choose
    parser.add_argument("--model", type=str, default="logistic")
    # parameters for preprocessing
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split", type=str, default="random", choices=["random", "time", "time_reverse"])
    parser.add_argument("--seed", type=int, default=2022)
    # parameters for logistic
    parser.add_argument("--lr", type=float, nargs='+', default=[0.5])
    parser.add_argument("--num_iterations", type=int, nargs='+', default=[500])
    # parameters for SVM
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "sigmoid", "rbf"])
    parser.add_argument("--decision_function_shape", type=str, default="ovr", choices=["ovo", "ovr"])
    parser.add_argument("--C", type=float, nargs='+', default=[1.])
    parser.add_argument("--gamma", type=float, nargs='+', default=[1.])
    # parameters for RF
    parser.add_argument("--n_estimators", type=int, nargs='+', default=[30])
    parser.add_argument("--max_depth", type=int, nargs='+', default=[20])
    # parameters for PCA+KNN
    parser.add_argument("--n_components", type=int, nargs='+', default=[10])
    parser.add_argument("--n_neighbors", type=int, nargs='+', default=[5])
    # NO parameters for naive_bayes
    # parameters for FNN
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[3])
    parser.add_argument("--activation", type=str, nargs='+', default=["logistic", "tanh", "relu"])
    parser.add_argument("--batch_size", type=int, nargs='+', default=[256])
    # parameter: max_iter, use num_iterations
    # parameter: learning rate, use lr
    return parser.parse_args()


def evaluate(clf, x, y):
    pred = clf.predict(x)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    return {"accuracy": acc, "auc": auc, "f1": f1, "precision": precision, "recall": recall}

def get_classifier_hparam_search(args, name="logistic"):
    if name == "logistic":
        param_grid = {'learning_rate': args.lr, 'num_iterations': args.num_iterations}
        print(param_grid)
        return GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', verbose=3)
        # return LogisticRegression(max_iter=1000)
    if name == "SVM":
        return SVM(C=args.C, gamma=args.gamma, kernel=args.kernel, decision_function_shape=args.decision_function_shape)
    if name == "RF":
        param_grid = {'n_estimators': args.n_estimators, 'max_depth': args.max_depth}
        print(param_grid)
        return GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1', verbose=3)
    if name == "PCA_KNN":
        param_grid = {'n_components': args.n_components, "n_neighbors": args.n_neighbors}
        print(param_grid)
        return GridSearchCV(PCA_KNN(), param_grid, cv=5, scoring='f1', verbose=3)
    if name == "NB":
        return naive_bayes()
    if name == "FNN":
        hidden_layer_sizes = []
        if 2 in args.hidden_layers:
            hidden_layer_sizes.append((64, 128))
        if 3 in args.hidden_layers:
            hidden_layer_sizes.append((64, 128, 64))
        if 4 in args.hidden_layers:
            hidden_layer_sizes.append((64, 128, 128, 64))
        param_grid = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": args.activation,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "max_iter": args.num_iterations
        }
        print(param_grid)
        return GridSearchCV(FNN(), param_grid, cv=5, scoring='f1', verbose=3)

def get_classifier(name="logistic", params={}):
    if name == "logistic":
        return LogisticRegression(**params)
    if name == "SVM":
        return SVM(C=args.C, gamma=args.gamma, kernel=args.kernel, decision_function_shape=args.decision_function_shape)
    if name == "RF":
        return RandomForestClassifier(**params)
    if name == "PCA_KNN":
        return PCA_KNN(**params)
    if name == "NB":
        return naive_bayes()
    if name == "FNN":
        return FNN(**params)

def main(args):
    data = load_data('./OnlineNewsPopularity.csv')
    if args.split == "random":
        train_data, test_data = random_split(data[0], data[1], args.train_ratio, args.test_ratio)
    elif args.split == "time" or args.split == "time_reverse":
        train_data, test_data = split_by_time(data[0], data[1], args.train_ratio, args.test_ratio, reverse=(args.split == "time_reverse"))
    else:
        raise NotImplementedError
    if args.model == "NB":
        best_params = {}
    elif args.model == "FNN":
        best_params = {"hidden_layer_sizes": (64, 128, 256), "activation": "relu", "batch_size": 256, "lr":1e-3, "max_iter":200}
    else:
        clf = get_classifier_hparam_search(args, name=args.model)
        clf.fit(train_data[0], train_data[1])
        best_params = clf.best_params_
        print(best_params)

    metrics = {"accuracy": [], "auc": [], "f1": [], "precision": [], "recall": []}
    for seed in range(10):
        set_seed(seed)
        if args.split == "random":
            train_data, test_data = random_split(data[0], data[1], args.train_ratio, args.test_ratio)
        elif args.split == "time" or args.split == "time_reverse":
            train_data, test_data = split_by_time(data[0], data[1], args.train_ratio, args.test_ratio, reverse=(args.split == "time_reverse"))
        else:
            raise NotImplementedError

        args.feature_dim = train_data[0].shape[1]
        clf = get_classifier(name=args.model, params=best_params)
        clf.fit(train_data[0], train_data[1])

        train_metrics = evaluate(clf, train_data[0], train_data[1])
        test_metrics = evaluate(clf, test_data[0], test_data[1])

        # print("Train metrics: ", train_metrics)
        print("Test metrics: ", test_metrics)

        metrics["accuracy"].append(test_metrics["accuracy"])
        metrics["auc"].append(test_metrics["auc"])
        metrics["f1"].append(test_metrics["f1"])
        metrics["precision"].append(test_metrics["precision"])
        metrics["recall"].append(test_metrics["recall"])

    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}±{np.std(v):.4f}")

    if isinstance(clf, LogisticRegression):
        np.save("wt.npy", clf.weight)
        np.save("bs.npy", clf.bias)


if __name__ == "__main__":
    args = build_args()
    main(args)