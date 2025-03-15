from flask import Flask, render_template, request, jsonify
from flask_bootstrap import Bootstrap
import random
import logging
import os
import requests  # Ensure this import is included

app = Flask(__name__)
app.config['ENV'] = 'development'
app.config['TEMPLATES_AUTO_RELOAD'] = True
Bootstrap(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    return render_template('form.html')

@app.route('/api/random-number', methods=['GET'])
def random_number():
    try:
        random_number = random.randint(1, 100)
        return jsonify(random_number=random_number)
    except Exception as e:
        app.logger.error(f"Error in /api/random-number: {e}")
        return jsonify(error=str(e)), 500

@app.route('/api/news', methods=['POST'])
def get_news():
    try:
        user_input1 = request.form['user_input1']
        user_input2 = request.form['user_input2']
        # Call the news API
        url = f"https://newsapi.org/v2/everything?q=tesla&from=2025-02-08&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        news_list = response.json().get('articles', [])
        return jsonify(user_input1=user_input1, user_input2=user_input2, news_list=news_list)
    except Exception as e:
        app.logger.error(f"Error in /api/news: {e}")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=True)
