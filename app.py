from flask import Flask, render_template, redirect, url_for, request, jsonify
import requests

app = Flask(__name__)
CHATBOT_URL = "http://127.0.0.1:8000/get_response"  # Ensure the correct endpoint

@app.route('/')
def index():
    return render_template('ui.html')

@app.route('/run_first_app')
def run_first_app():
    return redirect("http://127.0.0.1:8080")

@app.route('/run_second_app')
def run_second_app():
    return redirect("http://127.0.0.1:5000")

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    response = requests.post(CHATBOT_URL, json={'user_input': user_input})
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
