from flask import Flask, request, render_template, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import LinealRegression

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/linearRegression/conceptos')
def linearconceptos():
    return render_template('LRconceptos.html')

@app.route('/linearRegression/ejercicio', methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    graph_url = None

    # Dataset simulado
    X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y_train = np.array([2, 4, 5, 4, 5, 7, 8, 9])

    # Ajuste de regresión
    coef = np.polyfit(X_train, y_train, 1)
    poly1d_fn = np.poly1d(coef)

    # Crear gráfico con estilo mejorado
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#eaf2f8')

    # Degradado de fondo
    for i in range(100):
        ax.axhspan(i, i + 1, facecolor=plt.cm.Blues(i / 100), alpha=0.05)

    # Puntos llamativos
    scatter = ax.scatter(X_train, y_train, c=y_train, cmap="coolwarm", s=150, edgecolors="black", linewidth=1.5, alpha=0.9, label="Datos reales")

    # Línea de regresión
    ax.plot(X_train, poly1d_fn(X_train), color="#e74c3c", linewidth=3, label="Línea de Regresión")

    # Títulos y etiquetas
    ax.set_title("📈 Regresión Lineal - Datos con Estilo", fontsize=14, color="#2c3e50", weight="bold")
    ax.set_xlabel("Horas de Entrenamiento", fontsize=11)
    ax.set_ylabel("Rendimiento Deportivo", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Guardar gráfico
    graph_path = os.path.join("static", "regresion.png")
    plt.tight_layout()
    plt.savefig(graph_path, dpi=120)
    plt.close()

    graph_url = url_for('static', filename='regresion.png')

    # Si el usuario envía datos
    if request.method == "POST":
        hours = float(request.form["hours"])
        diet = float(request.form["diet"])
        calculateResult = LinealRegression.Rendimiento(hours, diet)
        calculateResult = round(calculateResult, 2)

    return render_template('LRindex.html', result=calculateResult, graph_url=graph_url)

@app.route('/index')
def index():
    return render_template('index2.html')

@app.route('/casos')
def casos():
    return render_template('index3.html')

if __name__ == "__main__":
    app.run(debug=True)
