from flask import Flask
from flask import render_template
app = Flask(__name__)


@app.route("/")
def home():
    name = "Flask"
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index2.html')

@app.route('/casos')
def casos():
    return render_template('index3.html', cases=CASES)

if __name__ == "_main_":
    app.run(debug=True)