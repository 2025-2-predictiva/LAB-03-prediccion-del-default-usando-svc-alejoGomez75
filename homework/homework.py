import json
import gzip
import os
import pickle
import zipfile
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ========================= UTILIDADES DE ENTRADA ========================= #

def leer_csv_desde_zip(ruta_zip: str, nombre_archivo: str) -> pd.DataFrame:
    """
    Abre un .zip y carga en un DataFrame el CSV interno indicado.
    """
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        with zf.open(nombre_archivo) as f:
            return pd.read_csv(f)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajustes básicos de limpieza y normalización de variables.
    """
    datos = df.copy()

    # ID no aporta información predictiva
    datos = datos.drop("ID", axis=1)

    # Homologar nombre de la variable objetivo
    datos = datos.rename(columns={"default payment next month": "default"})

    # Quitar filas incompletas
    datos = datos.dropna()

    # Filtrar categorías que se usan como “otros”
    datos = datos[(datos["EDUCATION"] != 0) & (datos["MARRIAGE"] != 0)]

    # Colapsar niveles altos de EDUCATION
    datos.loc[datos["EDUCATION"] > 4, "EDUCATION"] = 4

    return datos


# ========================= MODELO Y GRIDSEARCH ========================= #

def armar_grid_search() -> GridSearchCV:
    """
    Construye el pipeline completo + configuración de GridSearch.
    """
    columnas_cat = ["SEX", "EDUCATION", "MARRIAGE"]
    columnas_num = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_cat),
            ("std", StandardScaler(), columnas_num),
        ],
        remainder="passthrough",
    )

    modelo_base = Pipeline(steps=[
        ("prep", transformador),
        ("pca", PCA()),                           # aquí se dejan todas las componentes y se ajustan por grid
        ("kbest", SelectKBest(score_func=f_classif)),
        ("svc", SVC(kernel="rbf", random_state=42)),
    ])

    # Grid pequeño y dirigido (ya se sabe más o menos dónde funciona bien)
    param_grid = {
        "pca__n_components": [20, 21],
        "kbest__k": [12],
        "svc__kernel": ["rbf"],
        "svc__gamma": [0.099],
    }

    return GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=10,
        refit=True,
        verbose=1,
        return_train_score=False,
        scoring="balanced_accuracy",
    )


# ========================= MÉTRICAS Y SALIDA ========================= #

def construir_metricas(nombre: str, y_true, y_pred) -> Dict:
    """
    Empaqueta las métricas principales en un diccionario.
    """
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": precision_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def construir_cm(nombre: str, y_true, y_pred) -> Dict:
    """
    Convierte la matriz de confusión en una estructura fácil de guardar.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {"predicted_0": int(tn), "predicted_1": int(fp)},
        "true_1": {"predicted_0": int(fn), "predicted_1": int(tp)},
    }


def guardar_modelo_gzip(modelo) -> None:
    """
    Serializa el modelo entrenado en formato gzip + pickle.
    """
    Path("files/models").mkdir(parents=True, exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as fh:
        pickle.dump(modelo, fh)


def guardar_registros_jsonl(registros: List[Dict]) -> None:
    """
    Escribe una lista de diccionarios en formato JSONL (uno por línea).
    """
    Path("files/output").mkdir(parents=True, exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r) + "\n")


# ========================= PUNTO DE ENTRADA ========================= #

if __name__ == "__main__":
    ruta_test_zip = "files/input/test_data.csv.zip"
    ruta_train_zip = "files/input/train_data.csv.zip"
    nombre_csv_test = "test_default_of_credit_card_clients.csv"
    nombre_csv_train = "train_default_of_credit_card_clients.csv"

    df_test = limpiar_datos(leer_csv_desde_zip(ruta_test_zip, nombre_csv_test))
    df_train = limpiar_datos(leer_csv_desde_zip(ruta_train_zip, nombre_csv_train))

    X_tr, y_tr = df_train.drop("default", axis=1), df_train["default"]
    X_te, y_te = df_test.drop("default", axis=1), df_test["default"]

    buscador = armar_grid_search()
    buscador.fit(X_tr, y_tr)
    guardar_modelo_gzip(buscador)

    y_tr_pred = buscador.predict(X_tr)
    y_te_pred = buscador.predict(X_te)

    train_metrics = construir_metricas("train", y_tr, y_tr_pred)
    test_metrics  = construir_metricas("test",  y_te, y_te_pred)
    train_cm      = construir_cm("train", y_tr, y_tr_pred)
    test_cm       = construir_cm("test",  y_te, y_te_pred)

    guardar_registros_jsonl([train_metrics, test_metrics, train_cm, test_cm])