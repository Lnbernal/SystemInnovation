# tipos.py
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb


class LightGBMCase:
    model = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None

    label_mapping = {"Activo": 0, "Ocasional": 1, "Inactivo": 2}
    reverse_mapping = {0: "Activo", 1: "Ocasional", 2: "Inactivo"}

    @classmethod
    def train(cls):
        data = pd.read_csv("usuarios.csv")
        data["ubicacion"] = data["ubicacion"].map({"Rural": 0, "Urbana": 1})

        X = data[["tiempo_uso", "frecuencia", "interacciones", "ubicacion"]]
        y = data["tipo_usuario"].map(cls.label_mapping)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_data = lgb.Dataset(cls.X_train, label=cls.y_train)

        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1
        }

        cls.model = lgb.train(params, train_data, num_boost_round=50)

    @classmethod
    def predict_label(cls, features):
        if cls.model is None:
            cls.train()

        y_pred = cls.model.predict([features])
        label_index = y_pred[0].argmax()
        label = cls.reverse_mapping[label_index]
        prob = round(y_pred[0][label_index] * 100, 2)

        cls._generate_graphs_from_input(label_index)
        return label, prob

    @classmethod
    def evaluate(cls):
        if cls.model is None:
            raise ValueError("El modelo no está entrenado")

        y_pred = cls.model.predict(cls.X_test)
        y_pred_labels = [p.argmax() for p in y_pred]

        acc = accuracy_score(cls.y_test, y_pred_labels)
        return {"accuracy": round(acc * 100, 2)}

    @classmethod
    def _generate_graphs_from_input(cls, label_index):
        os.makedirs("static", exist_ok=True)

        # Distribución simulada
        counts = [0, 0, 0]
        counts[label_index] = 1

        plt.figure(figsize=(5, 4))
        plt.bar(["Activo", "Ocasional", "Inactivo"], counts, color="skyblue")
        plt.title("Distribución de Clases (Entrada)")
        plt.ylabel("Cantidad")
        plt.tight_layout()
        plt.savefig("static/class_distribution.png")
        plt.close()

        # Matriz de confusión simulada
        cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        cm[label_index][label_index] = 1

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Matriz de Confusión (Entrada)")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        for i in range(3):
            for j in range(3):
                plt.text(j, i, cm[i][j], ha="center", va="center", color="red")
        plt.tight_layout()
        plt.savefig("static/confusion_matrix_lightgbm.png")
        plt.close()
