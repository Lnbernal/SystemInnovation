# tipos.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os

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
        # Cargar dataset
        data = pd.read_csv("usuarios.csv")

        # Variables independientes
        X = data[["tiempo_uso", "frecuencia", "interacciones", "ubicacion"]]
        y = data["tipo_usuario"].map(cls.label_mapping)

        # División de datos
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # Configuración de LightGBM
        train_data = lgb.Dataset(cls.X_train, label=cls.y_train)
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "verbosity": -1
        }

        # Entrenar modelo
        cls.model = lgb.train(params, train_data, num_boost_round=50)

        # Guardar gráficas
        cls._generate_graphs(data)

    @classmethod
    def _generate_graphs(cls, data):
        os.makedirs("static", exist_ok=True)

        # Distribución de clases
        plt.figure(figsize=(6, 4))
        sns.countplot(x="tipo_usuario", data=data, palette="Set2")
        plt.title("Distribución de Usuarios")
        plt.savefig("static/class_distribution.png")
        plt.close()

        # Matriz de confusión
        y_pred = cls.model.predict(cls.X_test)
        y_pred_labels = [y.argmax() for y in y_pred]
        cm = confusion_matrix(cls.y_test, y_pred_labels)

        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=cls.reverse_mapping.values(),
                    yticklabels=cls.reverse_mapping.values())
        plt.title("Matriz de Confusión LightGBM")
        plt.ylabel("Real")
        plt.xlabel("Predicho")
        plt.savefig("static/confusion_matrix_lightgbm.png")
        plt.close()

    @classmethod
    def predict_label(cls, features):
        if cls.model is None:
            raise ValueError("El modelo no está entrenado. Llama a train() primero.")

        y_pred = cls.model.predict([features])
        label_index = y_pred[0].argmax()
        label = cls.reverse_mapping[label_index]
        prob = round(y_pred[0][label_index] * 100, 2)

        return label, prob

    @classmethod
    def evaluate(cls):
        if cls.model is None:
            raise ValueError("El modelo no está entrenado. Llama a train() primero.")

        y_pred = cls.model.predict(cls.X_test)
        y_pred_labels = [y.argmax() for y in y_pred]
        acc = accuracy_score(cls.y_test, y_pred_labels)

        return {"accuracy": round(acc * 100, 2)}
