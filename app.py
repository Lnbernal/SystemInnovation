from flask import Flask
from flask import render_template
app = Flask(__name__)


@app.route("/")
def home():
    name = "Flask"
    return render_template('index.html',{name})

@app.route('/index')
def index():
    return render_template('index2.html')


if __name__ == "_main_":
    app.run(debug=True)