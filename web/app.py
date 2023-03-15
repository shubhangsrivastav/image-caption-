from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from image_caption import generate_caption

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        id = request.form.get('id')
        id = id + ".jpg"
        output = (generate_caption(id))
        return render_template('predict.html',id = id , predicted  = output)

if __name__ == 'main':
    app.run(debug=True)