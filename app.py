from flask import Flask, request, render_template, url_for
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import LinealRegression
import RegresionLogistica

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/linearRegression/conceptos')
def linearconceptos():
    return render_template('LRconceptos.html')

@app.route('/linearRegression/ejercicio', methods=["GET", "POST"])
def calculatePerformance():
    calculateResult = None
    graph_url_hours = None
    graph_url_diet = None

    if request.method == "POST": 
        hours = float(request.form["hours"])
        diet = float(request.form["diet"])
        calculateResult = LinealRegression.Rendimiento(hours, diet)
        calculateResult = min(round(calculateResult, 2), 10)  # límite en 10

        # Aquí pasamos los dos valores
        graph_path_hours = LinealRegression.grafico_horas(hours, diet)
        graph_path_diet = LinealRegression.grafico_dieta(hours, diet)

        graph_url_hours = url_for('static', filename=os.path.basename(graph_path_hours))
        graph_url_diet = url_for('static', filename=os.path.basename(graph_path_diet))

    return render_template(
        'LRindex.html',
        result=calculateResult,
        graph_url_hours=graph_url_hours,
        graph_url_diet=graph_url_diet
)

@app.route('/regresionLogistica/conceptos')
def logistica():
    return render_template('RLconceptos.html')

@app.route('/regresionLogistica/ejercicio', methods=["GET", "POST"])
def logistica2():
    prediction = None
    accuracy = None
    graph_url = None

    if request.method == "POST":
        monto = float(request.form["monto"])
        hora = int(request.form["hora"])
        tipo = request.form["tipo"]
        pais = request.form["pais"]

        prob, resultado = RegresionLogistica.predecir_transaccion(monto, hora, tipo, pais)
        prediction = f"{resultado} (Probabilidad: {prob:.2f}%)"
        accuracy = RegresionLogistica.get_accuracy()
        graph_url = url_for('static', filename="confusion_matrix.png")

    return render_template(
        'RLindex.html',
        result=prediction,
        accuracy=accuracy,
        graph_url=graph_url
    )



@app.route('/index')
def index():
    return render_template('index2.html')

@app.route('/casos')
def casos():
    return render_template('index3.html')

if __name__ == "_main_":
    app.run(debug=True)