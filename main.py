import argparse
from os.path import exists
import pandas as pd
from src.id3 import DecisionTree
from src.knn import KNN
from src.lr import LogisticRegression

parser = argparse.ArgumentParser(
    description="Algoritma supervised learning (KNN, Logistic Regression, ID3)")

parser.add_argument("-m", "--model", action="store", type=str,
                    help="Nama model (KNN, Logistic Regression, atau ID3)", required=True)
parser.add_argument("-d", "--data", action="store", type=str,
                    help="Direktori file untuk data train", required=True)
parser.add_argument("-v", "--validation", action="store",
                    type=str, help="Direktori file untuk data validasi")
parser.add_argument("-t", "--test", action="store", type=str,
                    help="Direktori file untuk data test untuk diprediksi")

parser.add_argument("--lr", action="store", type=float,
                    help="Angka learning rate pada algoritma Logistic Regression. Diabaikan apabila menggunakan algoritma lain.")
parser.add_argument("--epochs", action="store", type=int,
                    help="Epoch untuk algoritma Logistic Regression. Diabaikan apabila menggunakan algoritma lain.")
parser.add_argument("--k_nearest", action="store", type=int,
                    help="Nilai K pada KNN. Diabaikan apabila menggunakan algoritma lain.")

args = parser.parse_args()
args_dict = vars(args)

train_dir = args_dict["data"]
model = args_dict["model"]

if not exists(train_dir):
    raise FileNotFoundError(train_dir + " tidak ditemukan.")

val_dir = args_dict["validation"]
if val_dir is not None:
    if not exists(val_dir):
        raise FileNotFoundError(val_dir + " tidak ditemukan.")

test_dir = args_dict["test"]
if args_dict["test"] is not None:
    test_dir = args_dict["test"]
    if not exists(test_dir):
        raise FileNotFoundError(test_dir + " tidak ditemukan.")

if model not in ["id_3", "knn", "log_reg"]:
    raise NotImplementedError(
        "Model tidak tersedia. Gunakan id_3, knn, atau log_reg.")

df = pd.read_csv(train_dir)
X = df.iloc[:, :-1].to_numpy()  # drop kolom terakhir
y = df.iloc[:, -1].to_numpy()  # Gunakan kolom terakhir
if model == "id_3":
    id3 = DecisionTree()

    id3.fit(X, y)
    train_acc = sum(id3.predict(X) == y) / len(y)
    print("--- ALGORITMA DECISION TREE ID3 ---")
    print("Akurasi training data:", round(train_acc*100, 2), "%")

    if val_dir is not None:
        val = pd.read_csv(val_dir)
        X_val = val.iloc[:, :-1].to_numpy()
        y_val = val.iloc[:, -1].to_numpy()
        val_acc = sum(id3.predict(X_val) == y_val) / len(y_val)
        print("Akurasi validation data:", round(val_acc*100, 2), "%")

    if test_dir is not None:
        test = pd.read_csv(test_dir)
        X_test = test.to_numpy()
        print("Prediksi label pada test data: ", end="")
        print(id3.predict(X_test))

elif model == "knn":
    nearest = 5
    if args_dict["k_nearest"] is not None:
        nearest = args_dict["k_nearest"]
    knn = KNN(k=nearest)
    knn.fit(X, y)
    train_acc = sum(knn.predict(X) == y) / len(y)

    print("--- ALGORITMA K-Nearest Neighbors ---")
    print("K =", nearest)
    print("Akurasi training data:", round(train_acc*100, 2), "%")

    if val_dir is not None:
        val = pd.read_csv(val_dir)
        X_val = val.iloc[:, :-1].to_numpy()
        y_val = val.iloc[:, -1].to_numpy()
        val_acc = sum(knn.predict(X_val) == y_val) / len(y_val)
        print("Akurasi validation data:", round(val_acc*100, 2), "%")

    if test_dir is not None:
        test = pd.read_csv(test_dir)
        X_test = test.to_numpy()
        print("Prediksi label pada test data: ", end="")
        print(knn.predict(X_test))

elif model == "log_reg":
    learning_rate = 0.1
    num_iter = 100
    if args_dict["lr"] is not None:
        learning_rate = args_dict["lr"]
    if args_dict["epochs"] is not None:
        num_iter = args_dict["epochs"]
    log_reg = LogisticRegression(
        learning_rate=learning_rate, num_iter=num_iter)
    log_reg.fit(X, y)
    train_acc = sum(log_reg.predict(X) == y) / len(y)

    print("--- ALGORITMA Logistic Regression ---")
    print("Learning Rate =", learning_rate)
    print("No. of Iteration =", num_iter)
    print("Akurasi training data:", round(train_acc*100, 2), "%")

    if val_dir is not None:
        val = pd.read_csv(val_dir)
        X_val = val.iloc[:, :-1].to_numpy()
        y_val = val.iloc[:, -1].to_numpy()
        val_acc = sum(log_reg.predict(X_val) == y_val) / len(y_val)
        print("Akurasi validation data:", round(val_acc*100, 2), "%")

    if test_dir is not None:
        test = pd.read_csv(test_dir)
        X_test = test.to_numpy()
        print("Prediksi label pada test data: ", end="")
        print(log_reg.predict(X_test))
