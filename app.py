from flask import Flask, jsonify, request, render_template
from model import generate_txt
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/generate',methods=['POST'])
def generate():
    data = request.json
    query = data.get('query','')

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        response = generate_txt(query)
        return jsonify({'response':response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0',port = 5000)