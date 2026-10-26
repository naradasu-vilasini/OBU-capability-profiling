import os
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from ml_utils import MLProcessor

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
for directory in ['models', 'results', 'uploads']:
    os.makedirs(directory, exist_ok=True)

# Initialize ML processor
ml_processor = MLProcessor()

def allowed_file(filename):
    """Check if uploaded file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Home page route."""
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload for training."""
    if 'dataset' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['dataset']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and process the dataset
            success = ml_processor.load_dataset(filepath)
            if success:
                flash('Dataset uploaded and processed successfully!', 'success')
            else:
                flash('Error processing dataset. Please check the file format.', 'error')
        except Exception as e:
            flash(f'Error processing dataset: {str(e)}', 'error')
            logging.error(f"Error processing dataset: {e}")
    else:
        flash('Invalid file format. Please upload a CSV file.', 'error')
    
    return redirect(url_for('index'))

@app.route('/eda')
def eda():
    """EDA page route."""
    if not ml_processor.is_data_loaded():
        flash('Please upload a dataset first.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate EDA plots
        plot_path = ml_processor.perform_eda()
        return render_template('eda.html', plot_path=plot_path)
    except Exception as e:
        flash(f'Error generating EDA plots: {str(e)}', 'error')
        logging.error(f"Error in EDA: {e}")
        return redirect(url_for('index'))

@app.route('/train_models', methods=['POST'])
def train_models():
    """Handle model training."""
    if not ml_processor.is_data_loaded():
        flash('Please upload a dataset first.', 'error')
        return redirect(url_for('index'))
    
    selected_models = request.form.getlist('models')
    if not selected_models:
        flash('Please select at least one model to train.', 'error')
        return redirect(url_for('index'))
    
    try:
        results = ml_processor.train_selected_models(selected_models)
        flash(f'Successfully trained {len(results)} models!', 'success')
        return redirect(url_for('performance'))
    except Exception as e:
        flash(f'Error training models: {str(e)}', 'error')
        logging.error(f"Error training models: {e}")
        return redirect(url_for('index'))

@app.route('/performance')
def performance():
    """Model performance comparison page."""
    if not ml_processor.is_data_loaded():
        flash('Please upload a dataset first.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get model comparison data
        comparison_data = ml_processor.get_comparison_data()
        plot_paths = ml_processor.get_performance_plots()
        
        return render_template('performance.html', 
                             comparison_data=comparison_data, 
                             plot_paths=plot_paths)
    except Exception as e:
        flash(f'Error loading performance data: {str(e)}', 'error')
        logging.error(f"Error in performance: {e}")
        return redirect(url_for('index'))

@app.route('/prediction')
def prediction():
    """Prediction page route."""
    if not ml_processor.is_data_loaded():
        flash('Please upload a dataset first.', 'error')
        return redirect(url_for('index'))
    
    # Get feature columns for the form
    feature_columns = ml_processor.get_feature_columns()
    return render_template('prediction.html', feature_columns=feature_columns)

@app.route('/predict_single', methods=['POST'])
def predict_single():
    """Handle single prediction."""
    try:
        # Extract form data
        input_data = {}
        feature_columns = ml_processor.get_feature_columns()
        
        for column in feature_columns:
            value = request.form.get(column)
            if value is None or value == '':
                flash(f'Please provide a value for {column}', 'error')
                return redirect(url_for('prediction'))
            input_data[column] = float(value)
        
        # Make prediction
        prediction = ml_processor.predict_single(input_data)
        
        return render_template('prediction.html', 
                             feature_columns=feature_columns,
                             single_prediction=prediction,
                             input_data=input_data)
    except Exception as e:
        flash(f'Error making prediction: {str(e)}', 'error')
        logging.error(f"Error in single prediction: {e}")
        return redirect(url_for('prediction'))

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Handle batch prediction from CSV file."""
    if 'batch_file' not in request.files:
        flash('No file selected for batch prediction', 'error')
        return redirect(url_for('prediction'))
    
    file = request.files['batch_file']
    if file.filename == '':
        flash('No file selected for batch prediction', 'error')
        return redirect(url_for('prediction'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Make batch predictions
            predictions_df = ml_processor.predict_batch(filepath)
            
            return render_template('prediction.html',
                                 feature_columns=ml_processor.get_feature_columns(),
                                 batch_predictions=predictions_df.to_html(classes='table table-striped table-hover'),
                                 batch_file=filename)
        except Exception as e:
            flash(f'Error making batch predictions: {str(e)}', 'error')
            logging.error(f"Error in batch prediction: {e}")
    else:
        flash('Invalid file format. Please upload a CSV file.', 'error')
    
    return redirect(url_for('prediction'))

@app.route('/results/<filename>')
def results(filename):
    """Serve result files (plots, etc.)."""
    return send_from_directory('results', filename)

@app.route('/download_predictions')
def download_predictions():
    """Download the latest batch predictions as CSV."""
    try:
        return send_from_directory('results', 'batch_predictions.csv', as_attachment=True)
    except Exception as e:
        flash('No predictions available for download', 'error')
        return redirect(url_for('prediction'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
