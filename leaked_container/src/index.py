from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def index():
    api_key = os.getenv('API_KEY')
    return f"API_KEY is: {api_key}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

