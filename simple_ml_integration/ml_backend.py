"""
Simple ML Backend - Everything in one file
Perfect for LLM tool integration
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from typing import Dict, Any

# Global state (simple session management)
CURRENT_MODEL = None
CURRENT_DATA = None
CURRENT_METADATA = None


def train_random_forest(
    data_path: str,
    target_column: str,
    task_type: str,
    n_estimators: int = 100
) -> Dict[str, Any]:
    """Train a random forest model"""
    global CURRENT_MODEL, CURRENT_DATA, CURRENT_METADATA
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Save globally
        CURRENT_MODEL = model
        CURRENT_DATA = {'X_train': X_train, 'X_test': X_test, 
                       'y_train': y_train, 'y_test': y_test}
        CURRENT_METADATA = {'task_type': task_type, 'features': X.columns.tolist()}
        
        return {
            'status': 'success',
            'message': f'Trained {task_type} model with {n_estimators} trees',
            'n_samples': len(X_train)
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def evaluate_model() -> Dict[str, Any]:
    """Evaluate the current model"""
    global CURRENT_MODEL, CURRENT_DATA, CURRENT_METADATA
    
    if CURRENT_MODEL is None:
        return {'status': 'error', 'message': 'No model trained yet'}
    
    try:
        X_test = CURRENT_DATA['X_test']
        y_test = CURRENT_DATA['y_test']
        y_pred = CURRENT_MODEL.predict(X_test)
        
        if CURRENT_METADATA['task_type'] == 'classification':
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
            }
        else:
            metrics = {
                'r2_score': float(r2_score(y_test, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred)))
            }
        
        return {'status': 'success', 'metrics': metrics}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def create_visualizations(output_dir: str = './outputs') -> Dict[str, Any]:
    """Create plots"""
    global CURRENT_MODEL, CURRENT_DATA, CURRENT_METADATA
    
    if CURRENT_MODEL is None:
        return {'status': 'error', 'message': 'No model trained yet'}
    
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        X_test = CURRENT_DATA['X_test']
        y_test = CURRENT_DATA['y_test']
        y_pred = CURRENT_MODEL.predict(X_test)
        
        plots = {}
        
        # Main plot
        plt.figure(figsize=(10, 6))
        if CURRENT_METADATA['task_type'] == 'regression':
            plt.scatter(y_test, y_pred, alpha=0.6)
            plt.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Predictions vs Actual')
            plot_path = f'{output_dir}/predictions.png'
        else:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plot_path = f'{output_dir}/confusion_matrix.png'
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['main'] = plot_path
        
        # Feature importance
        plt.figure(figsize=(10, 6))
        importance = CURRENT_MODEL.feature_importances_
        features = CURRENT_DATA['X_train'].columns
        idx = np.argsort(importance)[-15:]
        plt.barh(range(len(idx)), importance[idx])
        plt.yticks(range(len(idx)), [features[i] for i in idx])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plot_path = f'{output_dir}/feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['features'] = plot_path
        
        return {'status': 'success', 'plots': plots}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# That's it! Just 3 functions for LLM tools