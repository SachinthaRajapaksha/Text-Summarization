from flask import Flask, render_template, request, redirect, url_for, flash
import backend
import os
from werkzeug.utils import secure_filename 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' #
app.secret_key = 'your_secret_key_here'  # Set a secret key for flash messages

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'} # type of the doc

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/') # Home page
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['article'] # taking inputs
    length = request.form['length']
    method = request.form['method']  # New parameter for summarization method
    summary = backend.summarize_text(text, length, method)
    keywords = backend.extract_keywords(text)
    topics = backend.generate_topics(text)
    sentiment_description = backend.sentiment_summary(text)
    
    return render_template('index.html', summary=summary, keywords=keywords, topics=topics, sentiment=sentiment_description)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the uploaded PDF
        text = backend.process_pdf(file_path)
        length = request.form['length']
        method = request.form['method']  # New parameter for summarization method
        summary = backend.summarize_text(text, length, method)
        keywords = backend.extract_keywords(text)
        topics = backend.generate_topics(text)
        sentiment_description = backend.sentiment_summary(text)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return render_template('index.html', summary=summary, keywords=keywords, topics=topics, sentiment=sentiment_description, filename=filename)
    
    flash('Invalid file type. Please upload a PDF.')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
