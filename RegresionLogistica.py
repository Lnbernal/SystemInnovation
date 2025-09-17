import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ==========================
# Cargar y preparar datos
# ==========================
data = pd.read_csv('datos.csv')

# Variable objetivo
y = data['Fraude']
X = data.drop('Fraude', axis=1)

# One-hot encoding para variables categóricas
X = pd.get_dummies(X, columns=['Tipo_Comercio', 'Pais'], drop_first=True)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar modelo
Logistic_model = LogisticRegression(max_iter=200)
Logistic_model.fit(X_train, y_train)

# Evaluar
y_pred = Logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Exactitud del modelo:", accuracy * 100, "%")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("Reporte de clasificación:\n", classification_report(y_test, y_pred))


# ==========================
# Funciones para Flask
# ==========================
def predecir_transaccion(monto, hora, tipo, pais):
    # Crear un DataFrame con los datos de entrada
    entrada = pd.DataFrame([[monto, hora, tipo, pais]],
                           columns=["Monto", "Hora", "Tipo_Comercio", "Pais"])

    # One-hot encoding igual que el entrenamiento
    entrada = pd.get_dummies(entrada, columns=["Tipo_Comercio", "Pais"], drop_first=True)

    # Asegurar columnas consistentes con el entrenamiento
    for col in X.columns:
        if col not in entrada.columns:
            entrada[col] = 0

    entrada = entrada[X.columns]
    entrada = scaler.transform(entrada)

    # Probabilidad
    prob = Logistic_model.predict_proba(entrada)[0][1] * 100

    if prob < 40:
        resultado = "No es fraude"
    elif prob < 70:
        resultado = "Posible fraude"
    else:
        resultado = "Fraude total"

    return prob, resultado   # 👈 devuelve los dos valores

def get_accuracy():
    return round(accuracy * 100, 2)
