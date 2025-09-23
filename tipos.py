# tipos.py
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # evita abrir ventanas de GUI
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

        # Convertir ubicacion a numérico
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
        cls._generate_graphs()

    @classmethod
    def predict_label(cls, features):
        if cls.model is None:
            cls.train()

        y_pred = cls.model.predict([features])
        label_index = y_pred[0].argmax()
        label = cls.reverse_mapping[label_index]
        prob = round(y_pred[0][label_index] * 100, 2)
        return label, prob

    @classmethod
    def evaluate(cls):
        if cls.model is None:
            raise ValueError("El modelo no está entrenado")

        y_pred = cls.model.predict(cls.X_test)
        y_pred_labels = [p.argmax() for p in y_pred]

        acc = accuracy_score(cls.y_test, y_pred_labels)
        report = classification_report(
            cls.y_test, y_pred_labels,
            target_names=list(cls.reverse_mapping.values()),
            zero_division=0
        )

        return {"accuracy": round(acc * 100, 2), "report": report}

    @classmethod
    def _generate_graphs(cls):
        os.makedirs("static", exist_ok=True)

        # Matriz de confusión
        y_pred = cls.model.predict(cls.X_test)
        y_pred_labels = [p.argmax() for p in y_pred]
        cm = confusion_matrix(cls.y_test, y_pred_labels)

        plt.figure(figsize=(5, 4))
        plt.imshow(cm, cmap="Blues")
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center", color="red")
        plt.tight_layout()
        plt.savefig("static/confusion_matrix_lightgbm.png")
        plt.close()
