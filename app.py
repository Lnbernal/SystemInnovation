from flask import Flask, request, render_template, url_for
import os
import LinealRegression  # Se asume que existe
import RegresionLogistica

app = Flask(__name__)

# --------------------
# RUTA PRINCIPAL
# --------------------
@app.route("/")
def home():
    return render_template('index.html')

# --------------------
# REGRESIÓN LINEAL
# --------------------
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

# --------------------
# REGRESIÓN LOGÍSTICA
# --------------------
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

        # Predecir transacción: devuelve (probabilidad, etiqueta)
        prob, resultado = RegresionLogistica.predecir_transaccion(monto, hora, tipo, pais)
        prediction = f"{resultado} ({prob:.2f}%)"
        accuracy = RegresionLogistica.get_accuracy()
        graph_url = url_for('static', filename="confusion_matrix.png")

    return render_template(
        'RLindex.html',
        result=prediction,
        accuracy=accuracy,
        graph_url=graph_url
    )

# --------------------
# BOTÓN INICIO
# --------------------
@app.route('/index')
def index():
    return render_template('index2.html')

# --------------------
# BOTÓN CASOS DE USO
# --------------------
@app.route('/casos')
def casos():
    CASES = [
        {
            "titulo": "Predicción y monitoreo en agricultura con Machine Learning",
            "industria": "Agricultura",
            "problema": "Mejorar la productividad agrícola mediante predicción del rendimiento de cultivos y detección de plagas/enfermedades usando datos meteorológicos y sensores.",
            "algoritmo": "Random Forest, Árboles de decisión, SVM, Gradient Boosting, Redes neuronales convolucionales.",
            "beneficios": "Optimización de recursos, reducción de pérdidas, planificación eficiente de siembras, control de plagas y enfermedades con alta precisión.",
            "referencia": "https://repository.unad.edu.co/handle/10596/67132"
        },
        {
            "titulo": "IA para detección temprana de enfermedades y apoyo diagnóstico",
            "industria": "Salud",
            "problema": "Diagnósticos rápidos y precisos de enfermedades como cáncer, Alzheimer y enfermedades raras.",
            "algoritmo": "Redes neuronales profundas, Transfer Learning, NLP, Computer Vision.",
            "beneficios": "Diagnósticos más rápidos y precisos, reducción de errores médicos, optimización de recursos hospitalarios.",
            "referencia": "https://www.plainconcepts.com/es/inteligencia-artificial-sector-salud-ejemplos"
        },
        {
            "titulo": "Machine Learning en transacciones financieras",
            "industria": "Finanzas y Banca",
            "problema": "Detectar fraudes en transacciones y evaluar riesgo crediticio.",
            "algoritmo": "Modelos de clasificación, Deep Learning, Trading automático.",
            "beneficios": "Mayor seguridad, reducción de fraudes, decisiones de inversión más precisas.",
            "referencia": "https://www.ibm.com/es-es/think/topics/machine-learning-use-cases"
        },
        {
            "titulo": "Machine Learning y transporte",
            "industria": "Transporte",
            "problema": "Optimizar rutas, tiempos de llegada y mejorar la seguridad en transporte autónomo.",
            "algoritmo": "Aprendizaje supervisado, No supervisado, Redes neuronales, Computer Vision.",
            "beneficios": "Reducción de tiempos de viaje, asignación eficiente de recursos, mayor seguridad en transporte.",
            "referencia": "https://www.ibm.com/es-es/think/topics/machine-learning-use-cases"
        }
    ]
    return render_template('index3.html', cases=CASES)

if __name__ == "__main__":
    app.run(debug=True)
