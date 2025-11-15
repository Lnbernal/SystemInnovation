from flask import Flask, request, render_template, url_for
import os
import LinealRegression
import RegresionLogistica
from tipos import LightGBMCase

# Import del módulo MountainCar (asegúrate de que MountainCarQLearning.py esté en el mismo directorio)
from MountainCarQLearning import train_mountaincar, reward_history, run_policy

# Entrena LightGBM al iniciar (como en tu app original)
LightGBMCase.train()

app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')


# --------------------
# Regresión lineal
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
        calculateResult = min(round(calculateResult, 2), 10)

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


@app.route('/TiposDeAlgoritmos/conceptos')
def Tipos():
    return render_template('indexTipos.html')


@app.route("/TiposDeAlgoritmos/ejercicio", methods=["GET", "POST"])
def tipos_ejercicio():
    result = None
    prob = None
    accuracy = None
    graphs = {
        "distribucion": "/static/class_distribution.png",
        "confusion": "/static/confusion_matrix_lightgbm.png"
    }

    if request.method == "POST":
        try:
            # Capturar valores del formulario
            tiempo_uso = float(request.form["tiempo_uso"])
            frecuencia = int(request.form["frecuencia"])
            interacciones = int(request.form["interacciones"])
            ubicacion = int(request.form["ubicacion"])

            # Crear vector de entrada
            features = [tiempo_uso, frecuencia, interacciones, ubicacion]

            # Predicción
            result, prob = LightGBMCase.predict_label(features)

            # Evaluación del modelo
            eval_results = LightGBMCase.evaluate()
            accuracy = eval_results["accuracy"]

        except Exception as e:
            result = f"Error en predicción: {e}"

    return render_template(
        "indexTiposEj.html",
        result=result,
        prob=prob,
        accuracy=accuracy,
        graphs=graphs
    )


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


# ============================================================
# APRENDIZAJE POR REFUERZO - MOUNTAINCAR
# ============================================================

# Vista con los conceptos
@app.route('/aprendizajeRF/conceptos')
def af_conceptos():
    return render_template('AFconceptos.html')


# Vista principal del caso práctico
@app.route('/aprendizajeRF/ejercicio')
def af_mountaincar():
    return render_template('indexAF.html', graph=None, trajectory=None)


# Acción: entrenar el agente
# ============================================================
# APRENDIZAJE POR REFUERZO - ENTRENAR
# ============================================================
@app.route('/aprendizajeRF/entrenar')
def af_entrenar():
    train_mountaincar()      # Entrenar el modelo

    mensaje = "Entrenamiento completado correctamente."

    return render_template(
        'indexAF.html',
        graph=None,
        trajectory=None,
        mensaje=mensaje
    )


# Acción: mostrar gráficaz
@app.route('/aprendizajeRF/grafica')
def af_grafica():
    import os
    import matplotlib.pyplot as plt

    global reward_history
    load_reward_history()  # ESTA ES LA LÍNEA QUE FALTABA

    if not reward_history:
        mensaje = "Aún no has entrenado el modelo."

    # Crear gráfica
    plt.figure(figsize=(8, 4))
    plt.plot(reward_history)
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa acumulada")
    plt.title("Evolución del aprendizaje")

    os.makedirs('static', exist_ok=True)
    filepath = os.path.join('static', 'reward_plot.png')

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    graph_url = url_for('static', filename='reward_plot.png')

    return render_template(
        'indexAF.html',
        graph=graph_url,
        trajectory=None
    )


# Acción: probar la política aprendida
@app.route('/aprendizajeRF/politica')
def af_politica():
    import os
    import matplotlib.pyplot as plt

    trajectory, action_count = run_policy()

    acciones = ["Izquierda", "Quieto", "Derecha"]
    cantidades = [
        int(action_count.get(0, 0)),
        int(action_count.get(1, 0)),
        int(action_count.get(2, 0))
    ]

    plt.figure(figsize=(8, 4))
    plt.bar(acciones, cantidades)
    plt.title("Acciones realizadas por el agente")
    plt.ylabel("Frecuencia de uso")
    plt.xlabel("Acción")

    os.makedirs("static", exist_ok=True)
    filepath = os.path.join("static", "accion_plot.png")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    graph_acciones = url_for('static', filename='accion_plot.png')

    return render_template(
        "indexAF.html",
        graph=graph_acciones,
        trajectory=trajectory,
        action_count=action_count
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
