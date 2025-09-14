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
    probability = None
    metrics = None
    classification_html = None
    graph_url = None
    dataset_info = None

    # Obtener métricas y paths (evaluación ya se ejecutó en RegresionLogistica al importar)
    eval_res = RegresionLogistica.evaluate()
    metrics = {
        'accuracy': eval_res['accuracy'] * 100  # convertir a porcentaje
    }
    classification_html = eval_res['classification_html']
    graph_url = url_for('static', filename=os.path.basename(eval_res['confusion_path']))
    dataset_info = RegresionLogistica.get_dataset_description()

    if request.method == "POST":
        # Leer valores del formulario
        monto = float(request.form.get("monto"))
        hora = int(request.form.get("hora"))
        tipo = request.form.get("tipo")
        pais = request.form.get("pais")

        prob, label = RegresionLogistica.predict_label(monto, hora, tipo, pais, threshold=0.5)
        probability = round(prob * 100, 2)
        prediction = label  # "Sí" / "No"

    return render_template(
        'RLindex.html',
        result=prediction,
        probability=probability,
        accuracy=round(metrics['accuracy'], 4),
        classification_html=classification_html,
        graph_url=graph_url,
        dataset_info=dataset_info
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
