from flask import Flask, request, render_template, url_for
import os
import LinealRegression  # asumes que sigue existiendo
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
# Regresión logística
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

        # predecir_transaccion devuelve (probabilidad, etiqueta)
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

@app.route('/index')
def index():
    return render_template('index2.html')

@app.route('/casos')
def casos():
    CASES = [
        {
            "titulo": "Predicción y monitoreo en agricultura con Machine Learning",
            "industria": "Agricultura",
            "problema": "Mejorar la productividad agrícola mediante predicción del rendimiento de cultivos y detección de plagas/enfermedades usando datos meteorológicos y sensores.",
            "algoritmo": "Random Forest, Árboles de decisión, Máquinas de soporte vectorial (SVM), Gradient Boosting, Redes neuronales convolucionales.",
            "beneficios": "Optimización de recursos, reducción de pérdidas, planificación eficiente de siembras, control de plagas y enfermedades con alta precisión.",
            "referencia": "Chanchí-Golondrino, A. (2022). Aplicación de machine learning en la agricultura: predicción de rendimiento y control de plagas. Universidad Nacional Abierta y a Distancia (UNAD). Disponible en: https://repository.unad.edu.co/handle/10596/67132" 
        },
        {
            "titulo": "IA para detección temprana de enfermedades y apoyo diagnóstico",
            "industria": "Salud",
            "problema": "Dificultad para realizar diagnósticos rápidos y precisos de enfermedades como cáncer, Alzheimer y enfermedades raras, lo que retrasa el tratamiento oportuno.",
            "algoritmo": "Redes neuronales profundas, Árboles de decisión, Bosques aleatorios, Transfer Learning, NLP (Procesamiento de Lenguaje Natural), Computer Vision.",
            "beneficios": "Diagnósticos más rápidos y precisos, predicción temprana, reducción de errores médicos, optimización de recursos hospitalarios.",
            "referencia": "Plain Concepts. (2023). Inteligencia Artificial en el sector salud: ejemplos reales y casos de éxito. Disponible en: https://www.plainconcepts.com/es/inteligencia-artificial-sector-salud-ejemplos"
        },
        {
            "titulo": "Machine Learning en transacciones financieras",
            "industria": "Finanzas y Banca",
            "problema": "Detectar fraudes en transacciones, evaluar el riesgo crediticio y predecir tendencias bursátiles para optimizar las decisiones de inversión.",
            "algoritmo": "Modelos de clasificación, Modelos predictivos, Redes neuronales, Deep Learning, Algoritmos de trading automático.",
            "beneficios": "Mayor seguridad en transacciones, reducción de fraudes, decisiones de inversión más precisas, operaciones bursátiles de alta frecuencia, disminución del riesgo humano.",
            "referencia": "IBM. (s.f.).10 casos de uso cotidianos del machine learning. Disponible en: https://www.ibm.com/es-es/think/topics/machine-learning-use-cases"
        },
        {
            "titulo": "Machine Learning y transporte",
            "industria": "Transporte",
            "problema": "Optimizar rutas, tiempos de llegada, asignación de conductores y mejorar la seguridad en el transporte, incluyendo el desarrollo de vehículos autónomos.",
            "algoritmo": "Aprendizaje supervisado, Aprendizaje no supervisado, Redes neuronales profundas, Computer Vision, Modelos predictivos de tráfico.",
            "beneficios": "Reducción de tiempos de viaje, asignación eficiente de recursos en movilidad compartida, estimación precisa de la hora de llegada, mayor seguridad en transporte autónomo.",
            "referencia": "IBM. (s.f.).10 casos de uso cotidianos del machine learning. Disponible en: https://www.ibm.com/es-es/think/topics/machine-learning-use-cases"
        }


    ]
    return render_template('index3.html', cases=CASES)

if __name__ == "__main__":
    # Asegúrate de ejecutar desde el directorio del proyecto para que encuentre datos.csv y static/
    app.run(debug=True)