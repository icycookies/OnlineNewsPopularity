import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == "__main__":
    plt.figure(figsize=(6, 4))
    plt.xlabel("num_features")
    plt.ylabel("acc")
    data1 = pickle.load(open("./logisticablation.pkl", "rb"))
    data2 = pickle.load(open("./logisticrev_ablation.pkl", "rb"))
    plt.plot(np.arange(len(data1["f1"])), data1["f1"], label="high")
    plt.plot(np.arange(len(data2["f1"])), data2["f1"], label="low")
    plt.legend()
    #plt.show()
    plt.savefig("./report/ablation.png")