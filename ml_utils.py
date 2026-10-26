import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for web environment

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor

import joblib
from joblib import dump, load
import logging

class MLProcessor:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.label_encoders = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model_results = {}
        
        # Model directories
        self.MODEL_DIR = 'models'
        self.RESULTS_DIR = 'results'
        os.makedirs(self.MODEL_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        
        # Model performance tracking
        self.mae_list = []
        self.mse_list = []
        self.rmse_list = []
        self.r2_list = []
        self.model_names = []

    def load_dataset(self, filepath):
        """Load dataset from CSV file."""
        try:
            self.df = pd.read_csv(filepath)
            logging.info(f"Dataset loaded with shape: {self.df.shape}")
            
            # Preprocess the data
            self.X, self.y, self.label_encoders = self.preprocess_data(self.df, is_train=True)
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.3, random_state=42
            )
            
            logging.info("Data preprocessing and splitting completed")
            return True
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return False

    def is_data_loaded(self):
        """Check if data is loaded and processed."""
        return self.df is not None and self.X is not None and self.y is not None

    def preprocess_data(self, df, is_train=True, label_encoders=None):
        """Preprocess the data - exact copy from original code."""
        df = df.copy()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Drop unnamed cols

        if is_train:
            label_encoders = {}
            for col in df.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        else:
            if label_encoders is None:
                raise ValueError("label_encoders must be provided for test/inference.")
            
            for col in df.select_dtypes(include='object').columns:
                if col in label_encoders:
                    le = label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
                else:
                    raise ValueError(f"Missing encoder for column: {col}")

        df = df.fillna(df.mean(numeric_only=True))

        if is_train:
            X = df.drop(columns=['Eligible'])
            y = df['Eligible']
            return X, y, label_encoders
        else:
            return df

    def perform_eda(self):
        """Perform EDA and save plots - adapted from original code."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.patch.set_facecolor('white')

        # 1. Count Plot of Eligible Scores (Binned)
        bins = pd.cut(self.df['Eligible'], bins=5)
        axes[0, 0].hist([bins.apply(lambda x: x.mid)], bins=20, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Eligible Score Distribution (Binned)")
        axes[0, 0].set_xlabel("Eligible Range")
        axes[0, 0].set_ylabel("Count")

        # 2. Box Plot: RAM vs Eligible
        ram_bins = pd.cut(self.df['RAM'], bins=4)
        ram_data = [self.df[ram_bins == cat]['Eligible'].values for cat in ram_bins.cat.categories]
        axes[0, 1].boxplot(ram_data, labels=[str(cat) for cat in ram_bins.cat.categories])
        axes[0, 1].set_title("Eligible Score by RAM Ranges")
        axes[0, 1].set_xlabel("RAM Bins")
        axes[0, 1].set_ylabel("Eligible Score")

        # 3. Box Plot: Storage vs Eligible
        storage_bins = pd.cut(self.df['storage'], bins=4)
        storage_data = [self.df[storage_bins == cat]['Eligible'].values for cat in storage_bins.cat.categories]
        axes[0, 2].boxplot(storage_data, labels=[str(cat) for cat in storage_bins.cat.categories])
        axes[0, 2].set_title("Eligible Score by Storage Ranges")
        axes[0, 2].set_xlabel("Storage Bins")
        axes[0, 2].set_ylabel("Eligible Score")

        # 4. Scatter Plot: Trustfactor vs Eligible
        axes[1, 0].scatter(self.df['Trustfactor'], self.df['Eligible'], color='green', alpha=0.6)
        axes[1, 0].set_title("Eligible Score vs Trustfactor")
        axes[1, 0].set_xlabel("Trustfactor")
        axes[1, 0].set_ylabel("Eligible Score")

        # 5. Scatter Plot: TransmissionRate vs Eligible
        axes[1, 1].scatter(self.df['TransmissionRate'], self.df['Eligible'], color='purple', alpha=0.6)
        axes[1, 1].set_title("Eligible Score vs Transmission Rate")
        axes[1, 1].set_xlabel("Transmission Rate")
        axes[1, 1].set_ylabel("Eligible Score")

        # 6. Correlation Heatmap
        corr_matrix = self.df.corr()
        im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 2].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 2].set_xticklabels(corr_matrix.columns, rotation=45)
        axes[1, 2].set_yticklabels(corr_matrix.columns)
        axes[1, 2].set_title("Correlation Heatmap")
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 2])

        plt.tight_layout()
        plot_path = os.path.join(self.RESULTS_DIR, 'system_eda_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return 'system_eda_plots.png'

    def evaluate_performance(self, y_true, y_pred, model_name):
        """Evaluate model performance - exact copy from original code."""
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred) / 1000
        mse = mean_squared_error(y_true, y_pred) / 10000
        rmse = np.sqrt(mse) / 1000
        r2 = r2_score(y_true, y_pred)
        
        # Append metrics to lists
        self.model_names.append(model_name)
        self.mae_list.append(mae)
        self.mse_list.append(mse)
        self.rmse_list.append(rmse)
        self.r2_list.append(r2)
        
        # Create scatterplot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values - {model_name}')
        
        # Save plot
        plot_path = os.path.join(self.RESULTS_DIR, f'{model_name}_scatterplot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 score': r2}

    def train_decision_tree(self):
        """Train Decision Tree Regressor - exact copy from original code."""
        model_path = os.path.join(self.MODEL_DIR, 'dtr.joblib')
        
        if os.path.exists(model_path):
            model = load(model_path)
            logging.info("Loaded DecisionTreeRegressor model.")
        else:
            model = DecisionTreeRegressor(random_state=42, max_depth=1)
            model.fit(self.X_train, self.y_train)
            dump(model, model_path)
            logging.info("Trained and saved new DecisionTreeRegressor model.")

        y_pred = model.predict(self.X_test)
        return self.evaluate_performance(self.y_test, y_pred, "Decision Tree Regressor")

    def train_omp_regressor(self):
        """Train Orthogonal Matching Pursuit Regressor - exact copy from original code."""
        model_path = os.path.join(self.MODEL_DIR, 'omp_regressor.joblib')

        if os.path.exists(model_path):
            model = load(model_path)
            logging.info("Loaded Orthogonal Matching Pursuit Regressor model.")
        else:
            model = OrthogonalMatchingPursuit()
            model.fit(self.X_train, self.y_train)
            dump(model, model_path)
            logging.info("Trained and saved new Orthogonal Matching Pursuit Regressor model.")

        y_pred = model.predict(self.X_test)
        return self.evaluate_performance(self.y_test, y_pred, "Orthogonal Matching Pursuit Regressor")

    def train_knn_regressor(self):
        """Train KNN Regressor - exact copy from original code."""
        model_path = os.path.join(self.MODEL_DIR, 'knn_regressor.joblib')

        if os.path.exists(model_path):
            model = load(model_path)
            logging.info("Loaded KNN Regressor model.")
        else:
            model = KNeighborsRegressor(n_neighbors=5)
            model.fit(self.X_train, self.y_train)
            dump(model, model_path)
            logging.info("Trained and saved new KNN Regressor model.")

        y_pred = model.predict(self.X_test)
        return self.evaluate_performance(self.y_test, y_pred, "KNN Regressor")

    def train_random_forest_linear_regressor(self):
        """Train Random Forest + Linear Regressor hybrid - exact copy from original code."""
        model_path = os.path.join(self.MODEL_DIR, 'random_forest_linear_regressor.joblib')

        if os.path.exists(model_path):
            model = load(model_path)
            logging.info("Loaded Random Forest + Linear Regressor hybrid model.")
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            lr = LinearRegression()

            model = VotingRegressor(estimators=[
                ('rf', rf),
                ('lr', lr)
            ])
            
            model.fit(self.X_train, self.y_train)
            dump(model, model_path)
            logging.info("Trained and saved new Random Forest + Linear Regressor hybrid model.")

        y_pred = model.predict(self.X_test)
        return self.evaluate_performance(self.y_test, y_pred, "Random Forest + Linear Regressor")

    def train_selected_models(self, selected_models):
        """Train selected models based on user choice."""
        results = {}
        
        # Clear previous results
        self.mae_list = []
        self.mse_list = []
        self.rmse_list = []
        self.r2_list = []
        self.model_names = []
        
        model_functions = {
            'decision_tree': self.train_decision_tree,
            'omp': self.train_omp_regressor,
            'knn': self.train_knn_regressor,
            'random_forest_linear': self.train_random_forest_linear_regressor
        }
        
        for model_name in selected_models:
            if model_name in model_functions:
                try:
                    result = model_functions[model_name]()
                    results[model_name] = result
                    logging.info(f"Successfully trained {model_name}")
                except Exception as e:
                    logging.error(f"Error training {model_name}: {e}")
        
        self.model_results = results
        return results

    def get_comparison_data(self):
        """Create comparison table of all models' performance metrics."""
        if not self.model_names:
            return None
            
        comparison_df = pd.DataFrame({
            'Model': self.model_names,
            'MAE': self.mae_list,
            'MSE': self.mse_list,
            'RMSE': self.rmse_list,
            'R2 Score': self.r2_list
        })
        
        # Round numeric columns to 4 decimal places
        numeric_columns = ['MAE', 'MSE', 'RMSE', 'R2 Score']
        comparison_df[numeric_columns] = comparison_df[numeric_columns].round(4)
        
        # Save comparison table as CSV
        comparison_df.to_csv(os.path.join(self.RESULTS_DIR, 'regression_model_comparison.csv'), index=False)
        
        return comparison_df.to_dict('records')

    def get_performance_plots(self):
        """Get list of performance plot paths."""
        plot_paths = []
        for model_name in self.model_names:
            plot_filename = f'{model_name}_scatterplot.png'
            plot_path = os.path.join(self.RESULTS_DIR, plot_filename)
            if os.path.exists(plot_path):
                plot_paths.append(plot_filename)
        return plot_paths

    def get_feature_columns(self):
        """Get feature column names."""
        if self.X is not None:
            return list(self.X.columns)
        return []

    def predict_single(self, input_data):
        """Make prediction on single input."""
        # Load the best model (Random Forest + Linear Regressor)
        model_path = os.path.join(self.MODEL_DIR, 'random_forest_linear_regressor.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Please train the model first.")
        
        model = load(model_path)
        
        # Convert input to DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.X.columns]  # Ensure correct column order
        
        prediction = model.predict(input_df)[0]
        
        return {
            'prediction': prediction,
            'input_data': input_data
        }

    def predict_batch(self, filepath):
        """Make predictions on batch file."""
        # Load test data
        test_data = pd.read_csv(filepath)
        
        # Preprocess test data
        processed_test = self.preprocess_data(test_data, is_train=False, label_encoders=self.label_encoders)
        
        # Load the best model
        model_path = os.path.join(self.MODEL_DIR, 'random_forest_linear_regressor.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model not found. Please train the model first.")
        
        model = load(model_path)
        
        # Make predictions
        predictions = model.predict(processed_test)
        
        # Create output DataFrame
        output_data = []
        for i in range(len(processed_test)):
            row_data = {}
            
            # Add original test input features
            for col in processed_test.columns:
                row_data[col] = processed_test.iloc[i][col]
            
            # Add prediction
            row_data['Predicted_Eligible_Score'] = predictions[i]
            
            output_data.append(row_data)
        
        output_df = pd.DataFrame(output_data)
        
        # Save batch predictions
        output_df.to_csv(os.path.join(self.RESULTS_DIR, 'batch_predictions.csv'), index=False)
        
        return output_df
