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
    """Create detailed plots with metrics and annotations"""
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
        
        if CURRENT_METADATA['task_type'] == 'regression':
            # --- Regression: Predictions vs Actual ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Predictions vs Actual
            ax1 = axes[0]
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax1.plot([y_test.min(), y_test.max()], 
                    [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
            ax1.set_xlabel('Actual Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title('Predictions vs Actual Values', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper left')
            
            # Add metrics annotation
            textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nn = {len(y_test)}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Plot 2: Residuals
            ax2 = axes[1]
            residuals = y_test - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
            ax2.axhline(y=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Predicted Values', fontsize=12)
            ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
            ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
            
            # Add residual stats
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            textstr = f'Mean = {residual_mean:.4f}\nStd = {residual_std:.4f}'
            ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plot_path = f'{output_dir}/predictions.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['predictions'] = plot_path
            
            # --- Regression: Distribution of Errors ---
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
            ax.set_xlabel('Residual Value', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
            ax.legend()
            
            plot_path = f'{output_dir}/residual_distribution.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['residual_dist'] = plot_path
            
        else:
            # --- Classification: Confusion Matrix ---
            from sklearn.metrics import confusion_matrix, classification_report
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Confusion Matrix
            ax1 = axes[0]
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       cbar_kws={'label': 'Count'})
            ax1.set_xlabel('Predicted Label', fontsize=12)
            ax1.set_ylabel('True Label', fontsize=12)
            ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            
            # Add accuracy annotation
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            textstr = f'Accuracy = {acc:.4f}\nF1 Score = {f1:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax1.text(1.35, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            # Plot 2: Class Distribution
            ax2 = axes[1]
            unique, counts_true = np.unique(y_test, return_counts=True)
            unique, counts_pred = np.unique(y_pred, return_counts=True)
            
            x = np.arange(len(unique))
            width = 0.35
            ax2.bar(x - width/2, counts_true, width, label='Actual', alpha=0.8)
            ax2.bar(x + width/2, counts_pred, width, label='Predicted', alpha=0.8)
            ax2.set_xlabel('Class', fontsize=12)
            ax2.set_ylabel('Count', fontsize=12)
            ax2.set_title('Class Distribution: Actual vs Predicted', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([f'Class {i}' for i in unique])
            ax2.legend()
            
            plt.tight_layout()
            plot_path = f'{output_dir}/confusion_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['confusion'] = plot_path
        
        # --- Feature Importance (both tasks) ---
        fig, ax = plt.subplots(figsize=(10, 8))
        importance = CURRENT_MODEL.feature_importances_
        features = CURRENT_DATA['X_train'].columns
        
        # Sort by importance
        idx = np.argsort(importance)[-15:]  # Top 15
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(idx)))
        
        bars = ax.barh(range(len(idx)), importance[idx], color=colors, edgecolor='black')
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([features[i] for i in idx], fontsize=10)
        ax.set_xlabel('Feature Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, val in zip(bars, importance[idx]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
        
        # Add model info
        n_trees = CURRENT_MODEL.n_estimators
        textstr = f'Model: Random Forest\nTrees: {n_trees}\nTask: {CURRENT_METADATA["task_type"]}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        plot_path = f'{output_dir}/feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['features'] = plot_path
        
        return {
            'status': 'success', 
            'plots': plots,
            'message': f'Created {len(plots)} visualizations in {output_dir}'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}