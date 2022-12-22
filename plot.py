import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    """
    plt.xlabel("num_features")
    plt.ylabel("acc")
    data1 = pickle.load(open("./logisticablation.pkl", "rb"))
    data2 = pickle.load(open("./logisticrev_ablation.pkl", "rb"))
    data3 = pickle.load(open("./RFablation.pkl", "rb"))
    data4 = pickle.load(open("./RFrev_ablation.pkl", "rb"))
    plt.plot(np.arange(len(data1["f1"])), data1["f1"], label="logistic-high")
    plt.plot(np.arange(len(data3["f1"])), data3["f1"], label="RF-high")
    plt.plot(np.arange(len(data2["f1"])), data2["f1"], label="logistic-low")
    plt.plot(np.arange(len(data4["f1"])), data4["f1"], label="RF-low")
    plt.legend()
    #plt.show()
    plt.savefig("./report/ablation.png")
    """
    models = ["logistic", "PCA_KNN", "RF", "FNN", "NB"]
    #models = ["logistic", "PCA_KNN"]
    for model in models:
        data = pickle.load(open("./auc_result/" + model + ".pkl", "rb"))
        plt.plot(data['fpr'], data['tpr'], label="%s AUC=%.4lf" % (model, data['auc']))
        print(data['fpr'], data['tpr'])
    plt.plot([0, 1], [0, 1], label="Random AUC=0.5000", linestyle='--')
    plt.legend()
    plt.savefig("./report/auc.png")
    #plt.show()