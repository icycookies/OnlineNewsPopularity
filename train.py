from model import LogisticRegression
from data import load_data, random_split, split_by_time
from SVM_model import SVM
from PCA_KNN_model import PCA_KNN
from GaussianNB_model import naive_bayes
from FNN_model import FNN
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import argparse
import random
import numpy as np
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
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--num_iterations", type=int, default=500)
    # parameters for SVM
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "sigmoid", "rbf"])
    parser.add_argument("--decision_function_shape", type=str, default="ovr", choices=["ovo", "ovr"])
    parser.add_argument("--C", type=float, default=1.)
    parser.add_argument("--gamma", type=float, default=1.)
    # parameters for PCA+KNN
    parser.add_argument("--n_components", type=int, default=10)
    parser.add_argument("--n_neighbors", type=int, default=5)
    # NO parameters for naive_bayes
    # parameters for FNN
    parser.add_argument("--hidden_layer_sizes", type=tuple, default=(50,100,100,50))
    parser.add_argument("--activation", type=str, default="relu", choices=["logistic", "identity", "tanh", "relu"])
    parser.add_argument("--solver", type=str, default="adam", choices=["lbfgs", "sgd", "adam"])
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=str, default="constant", choices=["constant", "invscaling", "adaptive"])
    parser.add_argument("--power_t", type=float, default=0.5)
    # parameter: max_iter, use num_iterations
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    return parser.parse_args()


def evaluate(clf, x, y):
    pred = clf.predict(x)
    acc = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred)
    return {"accuracy": acc, "auc": auc, "f1": f1}


def get_classifier(args, name="logistic"):
    if name == "logistic":
        return LogisticRegression(args.feature_dim, learning_rate=args.lr, num_iterations=args.num_iterations)
        # return LogisticRegression(max_iter=1000)
    if name == "SVM":
        return SVM(C=args.C, gamma=args.gamma, kernel=args.kernel, decision_function_shape=args.decision_function_shape)
    if name == "PCA_KNN":
        return PCA_KNN(n_components=args.n_components, n_neighbors=args.n_neighbors)
    if name == "rf":
        return RandomForestClassifier(n_estimators=30)
    if name == "NB":
        return naive_bayes()
    if name == "FNN":
        return FNN(hidden_layer_sizes=args.hidden_layer_sizes, activation=args.activation, solver=args.solver,
            alpha=args.alpha, batch_size=args.batch_size, learning_rate=args.learning_rate,
            power_t=args.power_t, max_iter=args.num_iterations, shuffle=args.shuffle, beta_1=args.beta_1, beta_2=args.beta_2)


def main(args):
    data = load_data('./OnlineNewsPopularity.csv')

    metrics = {"accuracy": [], "auc": [], "f1": []}
    for seed in range(10):
        set_seed(seed)

        if args.split == "random":
            train_data, test_data = random_split(data[0], data[1], args.train_ratio, args.test_ratio)
        elif args.split == "time" or args.split == "time_reverse":
            train_data, test_data = split_by_time(data[0], data[1], args.train_ratio, args.test_ratio, reverse=(args.split == "time_reverse"))
        else:
            raise NotImplementedError

        args.feature_dim = train_data[0].shape[1]

        clf = get_classifier(args, name=args.model)
        clf.fit(train_data[0], train_data[1])

        train_metrics = evaluate(clf, train_data[0], train_data[1])
        test_metrics = evaluate(clf, test_data[0], test_data[1])

        # print("Train metrics: ", train_metrics)
        print("Test metrics: ", test_metrics)

        metrics["accuracy"].append(test_metrics["accuracy"])
        metrics["auc"].append(test_metrics["auc"])
        metrics["f1"].append(test_metrics["f1"])

    for k, v in metrics.items():
        print(f"{k}: {np.mean(v):.4f}±{np.std(v):.4f}")

    if isinstance(clf, LogisticRegression):
        np.save("wt.npy", clf.weight)
        np.save("bs.npy", clf.bias)


if __name__ == "__main__":
    args = build_args()
    main(args)