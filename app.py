from youtube_summary_tool import analyze_youtube_comments
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import subprocess
import threading
import json
import tempfile
import matplotlib
# Set matplotlib to use a non-interactive backend before any other matplotlib imports
# This must be done before importing youtube_summary_tool
matplotlib.use('Agg')

app = Flask(__name__, static_folder='.')

# Store the latest analysis results
latest_results = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    global latest_results

    data = request.json
    youtube_url = data.get('youtube_url')

    if not youtube_url:
        return jsonify({'error': 'No YouTube URL provided'}), 400

    try:
        # Run the analysis
        results = analyze_youtube_comments(youtube_url)

        # Store the results for later use
        latest_results = results

        return jsonify(results)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    global latest_results

    if not latest_results:
        return jsonify({'error': 'No analysis has been performed yet'}), 400

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Here you could implement a function to answer questions about the comments
        # For now, we'll just return a placeholder response
        answer = f"This is a placeholder answer for your question: '{question}'. In a real implementation, this would use the analysis results to provide a specific answer."

        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Serve static files


@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
