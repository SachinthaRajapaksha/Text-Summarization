from flask import Flask, render_template, request, redirect, url_for
import backend
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['article']
    length = request.form['length']  # Get summary length from the form
    summary = backend.summarize_text(text, length)  # Pass the selected length to the backend function
    keywords = backend.extract_keywords(text)  # Extract keywords
    topics = backend.extract_topics(text)  # Extract topics
    return render_template('index.html', summary=summary, keywords=keywords, topics=topics)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.pdf'):
        text = backend.process_pdf(file)
        length = request.form['length']  # Get summary length from the form (for PDF)
        summary = backend.summarize_text(text, length)  # Pass the selected length to the backend function
        keywords = backend.extract_keywords(text)  # Extract keywords
        topics = backend.extract_topics(text)  # Extract topics
        return render_template('index.html', summary=summary, keywords=keywords, topics=topics)
    return redirect(request.url)


if __name__ == '__main__':
    app.run(debug=True)
