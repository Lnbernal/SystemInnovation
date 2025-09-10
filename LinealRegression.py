import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# Dataset de entrenamiento 
data = {
    "Training Hours": [5, 8, 6, 4, 9, 3, 10, 7, 6, 8, 5, 2, 11, 1, 12, 3, 9, 2, 13, 1],
    "Balanced Diet": [7, 8, 6, 5, 9, 4, 9, 7, 6, 8, 5, 3, 10, 2, 10, 4, 9, 2, 10, 1],
    "Performance":   [7.5, 8.2, 7.0, 6.0, 9.0, 5.0, 9.5, 7.8, 7.2, 8.5, 6.8, 4.0, 9.8, 3.0, 10, 5.2, 9.1, 3.5, 10, 2.5]
}

df = pd.DataFrame(data)

X = df[["Training Hours", "Balanced Diet"]]
y = df["Performance"]

# Entrenar modelo
model = LinearRegression()
model.fit(X, y)

def Rendimiento(hours, diet):
 
    return model.predict([[hours, diet]])[0]
def grafico_horas(hours, diet):

    X_horas = np.linspace(0, 14, 100)
    
    y_linea = 0.6 * X_horas + 2 
    ruido = np.random.normal(0, 0.5, size=X_horas.shape)
    y_datos = y_linea + ruido  

    fig, ax = plt.subplots()
    ax.scatter(X_horas, y_datos, color="blue", s=50, alpha=0.6, label="Datos simulados")   
    ax.plot(X_horas, y_linea, color="black", linewidth=2, label="Regresión lineal")        
    ax.scatter(hours, Rendimiento(hours, diet), color="red", s=120, label="Tus datos")    

    ax.set_xlabel("Horas de entrenamiento")
    ax.set_ylabel("Rendimiento deportivo")
    ax.set_ylim(0, 10)
    ax.legend()

    path = os.path.join("static", "grafico_horas.png")
    plt.savefig(path)
    plt.close()
    return path

def grafico_dieta(hours, diet):

    X_dieta = np.linspace(1, 10, 100)
    

    y_linea = 0.8 * X_dieta + 2  


    ruido = np.random.normal(0, 0.5, size=X_dieta.shape)
    y_datos = y_linea + ruido  

    fig, ax = plt.subplots()
    ax.scatter(X_dieta, y_datos, color="green", s=50, alpha=0.6, label="Datos simulados")  
    ax.plot(X_dieta, y_linea, color="black", linewidth=2, label="Regresión lineal")        
    ax.scatter(diet, Rendimiento(hours, diet), color="red", s=120, label="Tus datos")      

    ax.set_xlabel("Calidad de alimentación (1-10)")
    ax.set_ylabel("Rendimiento deportivo")
    ax.set_ylim(0, 10)
    ax.legend()

    path = os.path.join("static", "grafico_dieta.png")
    plt.savefig(path)
    plt.close()
    return path