import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def eda(df):
    print(df['TARGET'].value_counts(normalize=True) * 100)  # 0: 68.9%; 1:31.1%
    print(df.isnull().sum().sum())
    print(df.describe())


def imputation(df):
    num_columns = df.select_dtypes(include=[np.number]).columns
    imputer = KNNImputer(n_neighbors=6)
    df[num_columns] = imputer.fit_transform(df[num_columns])

    return df


def outliers(df):
    num_columns = [col for col in df.select_dtypes(include=[np.number]) if col not in ["TARGET", "MES"]]
    for col in num_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lwr_bnd = Q1 - 1.5 * IQR
        ppr_bnd = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lwr_bnd, ppr_bnd)

    return df


def data_preparation(df):
    X = df.drop(["TARGET", "MES"], axis=1)
    y = df["TARGET"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test  # (16000, 34), (16000,), (4000, 34), (4000,) (shape)


def model_fit(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob


def model_evaluate(y_test, y_pred, y_prob, results, model_name):
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "Fscore": f1,
        "AUC": auc
    })

    print(f"\tPrecision:{precision:.3f}")
    print(f"\tRecall:{recall:.3f}")
    print(f"\tFscore:{f1:.3f}")
    print(f"\tAUC:{auc:.3f}")
    # visualitation(y_test, y_prob, auc)
    return results


def visualitation(y_test, y_prob, auc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(12, 12))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "r--")
    plt.title(f"ROC - AUC: {auc:.3f}")
    plt.xlabel("Falsos Positivos")
    plt.ylabel("Verdaderos Positivos")
    plt.show()


def models(name, model, X_train, y_train, X_test, y_test, results):
    print(name)
    y_pred, y_prob = model_fit(model, X_train, y_train, X_test)
    results = model_evaluate(y_test, y_pred, y_prob, results, name)
    return results


def treatment_column_date(df):
    df['MES'] = pd.to_datetime(df['MES'].astype(str) + '01', format='%Y%m%d')
    return df


def save_results_to_csv(results, filename):
    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False)


if __name__ == "__main__":
    dataset_path = "./Data/caso_final_small_20k_con_mes.csv"
    results = []
    df = pd.read_csv(dataset_path, sep=",")
    df = treatment_column_date(df)
    df = imputation(df)
    df = outliers(df)

    X_train, X_test, y_train, y_test = data_preparation(df)

    results = models("LogisticRegression", LogisticRegression(), X_train, y_train, X_test, y_test, results)
    results = models("RandomForestClassifier", RandomForestClassifier(), X_train, y_train, X_test, y_test, results)
    results = models("MLPClassifier", MLPClassifier(max_iter=1000, batch_size=64), X_train, y_train, X_test,
                     y_test, results)

    kmeans = KMeans(n_clusters=2, random_state=42)
    segments = kmeans.fit_predict(X_train)

    X_train_0 = X_train[segments == 0]
    y_train_0 = y_train[segments == 0]
    X_train_1 = X_train[segments == 1]
    y_train_1 = y_train[segments == 1]

    results = models("LogisticRegression cluster 0", LogisticRegression(), X_train_0, y_train_0, X_test, y_test,
                     results)
    results = models("LogisticRegression cluster 1", LogisticRegression(), X_train_1, y_train_1, X_test, y_test,
                     results)
    results = models("RandomForestClassifier cluster 0", RandomForestClassifier(), X_train_0, y_train_0, X_test,
                     y_test, results)
    results = models("RandomForestClassifier cluster 1", RandomForestClassifier(), X_train_1, y_train_1, X_test,
                     y_test, results)
    results = models("MLPClassifier cluster 0", MLPClassifier(max_iter=1000, batch_size=64), X_train_0, y_train_0,
                     X_test, y_test, results)
    results = models("MLPClassifier cluster 1", MLPClassifier(max_iter=1000, batch_size=64), X_train_1, y_train_1,
                     X_test, y_test, results)

    save_results_to_csv(results, "./Data/model_results.csv")

    '''
    The models show a moderate performance:
    - The logistic regression has the best precision and AUC, but the Recall is low, which indicates that the positive class is not being treated well
    - The RandomForest model has a lower precision, but a better performance and a poor predictive capacity for the positive classes
    - The MLP model is the one that has shown the worst performance
    
    Regarding the separation of the clusters, the predictions do not improve significantly
    '''
