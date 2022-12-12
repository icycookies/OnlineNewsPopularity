from model import LogisticRegression
from data import load_data, random_split, split_by_time
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import argparse
import random
import numpy as np
# from sklearn.linear_model import LogisticRegression


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="logistic")
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--num_iterations", type=int, default=500)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--split", type=str, default="random", choices=["random", "time", "time_reverse"])
    parser.add_argument("--seed", type=int, default=2022)
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


def main(args):
    set_seed(args.seed)
    data = load_data('./OnlineNewsPopularity.csv')
    if args.split == "random":
        train_data, test_data = random_split(data[0], data[1], args.train_ratio, args.test_ratio)
    elif args.split == "time" or args.split == "time_reverse":
        train_data, test_data = split_by_time(data[0], data[1], args.train_ratio, args.test_ratio, reverse=(args.split == "time_reverse"))
    else:
        raise NotImplementedError

    print("Train data: ", train_data[0].shape, train_data[1].shape)
    args.feature_dim = train_data[0].shape[1]

    clf = get_classifier(args, name=args.model)
    clf.fit(train_data[0], train_data[1])

    train_metrics = evaluate(clf, train_data[0], train_data[1])
    test_metrics = evaluate(clf, test_data[0], test_data[1])

    print("Train metrics: ", train_metrics)
    print("Test metrics: ", test_metrics)


if __name__ == "__main__":
    args = build_args()
    main(args)