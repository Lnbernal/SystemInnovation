# RegresionLogistica.py
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'datos.csv')
CONF_MATRIX_PATH = os.path.join(BASE_DIR, 'static', 'confusion_matrix.png')

# ==========================
# Cargar y preparar datos
# ==========================
data = pd.read_csv(DATA_PATH)

# Documentación: en este dataset la columna 'Fraude' es la variable objetivo.
# 0 = No (no es fraude), 1 = Sí (es fraude).
y = data['Fraude']
X = data.drop('Fraude', axis=1)

# One-hot encoding para variables categóricas (se usa drop_first=True para evitar multicolinealidad)
X = pd.get_dummies(X, columns=['Tipo_Comercio', 'Pais'], drop_first=True)

# Guardar columnas de referencia para el procesamiento de nuevas entradas
X_columns = X.columns.copy()

# Dividir datos (estratificado para mantener proporciones de clase)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Escalar (fit en train, transform en test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar modelo
Logistic_model = LogisticRegression(max_iter=200)
Logistic_model.fit(X_train_scaled, y_train)

# Predicciones y evaluación (se calculan una vez y se guardan)
y_pred = Logistic_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Función para generar y guardar matriz de confusión como imagen (2x2)
def save_confusion_matrix_image(conf_matrix, path=CONF_MATRIX_PATH):
    plt.figure(figsize=(4,4))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Pred 0', 'Pred 1'])
    plt.yticks(tick_marks, ['Real 0', 'Real 1'])

    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black",
                 fontsize=16)

    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    # Asegurar carpeta static
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

# Guardar imagen al importarse (o la primera vez que se pida)
save_confusion_matrix_image(conf_mat, CONF_MATRIX_PATH)

# Convertir reporte de clasificación a HTML (tabla)
def classification_report_to_html(report_dict):
    # report_dict viene de classification_report(..., output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    # Redondeo para mejor lectura
    df_round = df.round(4)
    # Renombrar índice (opcional)
    df_round.index.name = 'Clase/Métrica'
    html = df_round.to_html(classes="table table-sm table-striped", border=0)
    return html

classification_html = classification_report_to_html(class_report)

# ==========================
# Funciones públicas para Flask
# ==========================
def evaluate():
    """
    Retorna métricas calculadas y garantiza que la imagen de la matriz de confusión esté guardada.
    Devuelve: dict con keys -> accuracy (float), classification_html (str), confusion_path (str)
    """
    # Guardar imagen (por si no existe o se re-calcula)
    save_confusion_matrix_image(conf_mat, CONF_MATRIX_PATH)
    return {
        'accuracy': round(accuracy, 4),  # 4 decimales
        'classification_html': classification_html,
        'confusion_path': os.path.join('static', os.path.basename(CONF_MATRIX_PATH)),
        'conf_matrix': conf_mat
    }

def predict_label(monto, hora, tipo, pais, threshold=0.5):
    """
    Recibe las características (monto: float, hora: int, tipo: str, pais: str)
    Devuelve: (probabilidad_en_0_1, etiqueta_str) donde etiqueta_str es "Sí" (1) o "No" (0)
    threshold: umbral para decidir etiqueta
    """
    # Crear DataFrame con la entrada
    entrada = pd.DataFrame([[monto, hora, tipo, pais]],
                           columns=["Monto", "Hora", "Tipo_Comercio", "Pais"])

    # One-hot encoding usando las mismas reglas
    entrada = pd.get_dummies(entrada, columns=["Tipo_Comercio", "Pais"], drop_first=True)

    # Añadir columnas faltantes con 0 y reordenar según X_columns
    for col in X_columns:
        if col not in entrada.columns:
            entrada[col] = 0
    entrada = entrada[X_columns]

    # Escalar usando el scaler entrenado
    entrada_scaled = scaler.transform(entrada)

    # Probabilidad de clase 1
    prob_1 = Logistic_model.predict_proba(entrada_scaled)[0][1]

    label = "Sí" if prob_1 >= threshold else "No"
    return prob_1, label

def get_accuracy():
    # Devuelve porcentaje con 2 decimales como pediste en app
    return round(accuracy * 100, 2)

# Si quieres exponer también el dataframe de descripción del dataset
def get_dataset_description():
    """
    Retorna un diccionario con información simple del dataset:
    - columnas
    - tamaño
    - conteo de clases
    - significado de la variable objetivo
    """
    return {
        'columns': list(data.columns),
        'shape': data.shape,
        'class_counts': data['Fraude'].value_counts().to_dict(),
        'target_meaning': {'0': 'No (no es fraude)', '1': 'Sí (es fraude)'}
    }