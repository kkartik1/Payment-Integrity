"""
Healthcare Claims Upcoding Prediction System
(Updated for `ClaimsData` schema)

- Reads from ClaimsData table (see provided schema)
- Robust Label -> is_upcoded mapping
- Feature engineering aligned to new columns (dates, amounts, dx/modifiers, provider/procedure aggregates)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, roc_curve
)
# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# Utility helpers
# =========================
def to_snake_case(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("-", "_")
         .replace("__", "_")
         .lower()
    )

def safe_div(n, d):
    return np.where((d is None) | (pd.Series(d).fillna(0) == 0), 0, n / (d + 1e-9))

def parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def map_label_to_binary(s) -> int:
    """Map various encodings of upcoding labels to {0,1}."""
    if pd.isna(s):
        return 0
    val = str(s).strip().lower()
    positives = {"1", "true", "t", "y", "yes", "upcoded", "upcode", "fraud", "suspected", "positive"}
    negatives = {"0", "false", "f", "n", "no", "clean", "not upcoded", "negative", "non-upcoded"}
    if val in positives:
        return 1
    if val in negatives:
        return 0
    # fallback: try numeric
    try:
        return int(float(val) > 0)
    except:
        return 0


# =========================
# Data Loading
# =========================
class DataLoader:
    """Module for loading and initial data exploration"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def load_data(self, query: str = None) -> pd.DataFrame:
        """Load data from SQLite database"""
        if query is None:
            query = "SELECT * FROM ClaimsData"
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql(query, conn)
        conn.close()

        # Normalize column names to snake_case once
        df.columns = [to_snake_case(c) for c in df.columns]

        # Create robust binary target from `label`
        if 'label' in df.columns:
            df['is_upcoded'] = df['label'].apply(map_label_to_binary).astype(int)
        else:
            raise ValueError("Expected column `Label` in ClaimsData to identify upcoding.")

        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate data summary statistics"""
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum(),
            'dtypes': df.dtypes
        }
        if 'is_upcoded' in df.columns:
            summary['target_distribution'] = df['is_upcoded'].value_counts(dropna=False)
            summary['target_percentage'] = df['is_upcoded'].value_counts(normalize=True) * 100
        else:
            summary['target_distribution'] = "N/A"
            summary['target_percentage'] = "N/A"
        return summary


# =========================
# Feature Engineering
# =========================
class FeatureEngineering:
    """Module for feature engineering and preprocessing"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()  # (kept for optional future scaling)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new features from ClaimsData schema"""
        df = df.copy()

        # --- Date features ---
        date_cols = [
            'claim_paid_date',
            'claim_service_start_date', 'claim_service_end_date',
            'line_service_start_date', 'line_service_end_date',
            'admission_date', 'discharge_date',
            'member_date_of_birth'
        ]
        for c in date_cols:
            if c in df.columns:
                df[c] = parse_date(df[c])

        # Member age at claim service start
        if {'member_date_of_birth', 'claim_service_start_date'}.issubset(df.columns):
            df['age_at_service'] = (
                (df['claim_service_start_date'] - df['member_date_of_birth'])
                .dt.days / 365.25
            ).clip(lower=0)
        else:
            df['age_at_service'] = np.nan

        # Service durations
        if {'claim_service_start_date', 'claim_service_end_date'}.issubset(df.columns):
            df['claim_service_duration_days'] = (
                df['claim_service_end_date'] - df['claim_service_start_date']
            ).dt.days.clip(lower=0)
        else:
            df['claim_service_duration_days'] = np.nan

        if {'line_service_start_date', 'line_service_end_date'}.issubset(df.columns):
            df['line_service_duration_days'] = (
                df['line_service_end_date'] - df['line_service_start_date']
            ).dt.days.clip(lower=0)
        else:
            df['line_service_duration_days'] = np.nan

        # Time to payment
        if {'claim_paid_date', 'claim_service_end_date'}.issubset(df.columns):
            df['days_to_payment'] = (
                df['claim_paid_date'] - df['claim_service_end_date']
            ).dt.days
        else:
            df['days_to_payment'] = np.nan

        # Admission -> Discharge
        if {'admission_date', 'discharge_date'}.issubset(df.columns):
            df['admission_to_discharge_days'] = (
                df['discharge_date'] - df['admission_date']
            ).dt.days
        else:
            df['admission_to_discharge_days'] = np.nan

        # --- Diagnosis complexity ---
        dx_cols = [
            'primary_diagnosis_code',
            'diagnosis_code_1','diagnosis_code_2','diagnosis_code_3','diagnosis_code_4',
            'diagnosis_code_5','diagnosis_code_6','diagnosis_code_7','diagnosis_code_8'
        ]
        present_dx_cols = [c for c in dx_cols if c in df.columns]
        df['diagnosis_count'] = df[present_dx_cols].notna().sum(axis=1)

        # --- Financial features (claim-level) ---
        # Claim totals: charge and payable
        if {'claim_total_charge', 'claim_total_payable'}.issubset(df.columns):
            df['claim_charge_to_payable_ratio'] = safe_div(
                df['claim_total_charge'], df['claim_total_payable']
            )
            df['claim_payment_rate'] = safe_div(
                df['claim_total_payable'], df['claim_total_charge']
            )
        else:
            df['claim_charge_to_payable_ratio'] = np.nan
            df['claim_payment_rate'] = np.nan

        # --- Financial features (line-level) ---
        # line: charge, paid, allowed, units
        for col in ['line_charge_amount','line_paid_amount','line_allowed_amount','line_units','allowed_units']:
            if col not in df.columns:
                df[col] = np.nan

        df['line_charge_to_allowed_ratio'] = safe_div(df['line_charge_amount'], df['line_allowed_amount'])
        df['line_paid_to_allowed_ratio'] = safe_div(df['line_paid_amount'], df['line_allowed_amount'])

        df['patient_cost_share'] = (
            df.get('coinsurance_amount', 0).fillna(0) +
            df.get('copay_amount', 0).fillna(0) +
            df.get('deductible_amount', 0).fillna(0)
        )
        df['disallowed_rate'] = safe_div(df.get('disallowed_amount', 0).fillna(0),
                                         df['line_charge_amount'].replace(0, np.nan))
        df['discount_rate'] = safe_div(df.get('discount_amount', 0).fillna(0),
                                       df['line_charge_amount'].replace(0, np.nan))
        df['risk_withhold_rate'] = safe_div(df.get('risk_withhold_amount', 0).fillna(0),
                                            df['line_paid_amount'].replace(0, np.nan))

        # Unit pricing
        df['unit_price'] = safe_div(df['line_charge_amount'], df['line_units'].replace(0, np.nan))
        df['allowed_unit_price'] = safe_div(df['line_allowed_amount'], df['allowed_units'].replace(0, np.nan))

        # High-usage / high-charge flags (global percentile; optionally do per-procedure)
        df['high_units'] = (df['line_units'] > df['line_units'].quantile(0.95)).astype('int', errors='ignore')
        df['high_line_charge'] = (df['line_charge_amount'] > df['line_charge_amount'].quantile(0.95)).astype('int', errors='ignore')

        # --- Modifiers ---
        mod_cols = ['modifier_code1','modifier_code2','modifier_code3','modifier_code4']
        present_mod_cols = [c for c in mod_cols if c in df.columns]
        if present_mod_cols:
            df['modifier_count'] = df[present_mod_cols].notna().sum(axis=1)
        else:
            df['modifier_count'] = 0

        # --- Provider-level aggregates ---
        prov_key = 'provider_id' if 'provider_id' in df.columns else None
        if prov_key:
            agg = df.groupby(prov_key).agg({
                'line_charge_amount': ['mean', 'sum', 'count'],
                'line_allowed_amount': 'mean',
                'line_units': 'mean',
                'is_upcoded': 'mean'
            }).reset_index()
            agg.columns = [
                prov_key,
                'provider_avg_line_charge', 'provider_total_line_charge', 'provider_line_count',
                'provider_avg_line_allowed', 'provider_avg_line_units', 'provider_upcoding_rate'
            ]
            df = df.merge(agg, on=prov_key, how='left')
        else:
            # Fill with NaNs if not available
            df['provider_avg_line_charge'] = np.nan
            df['provider_total_line_charge'] = np.nan
            df['provider_line_count'] = np.nan
            df['provider_avg_line_allowed'] = np.nan
            df['provider_avg_line_units'] = np.nan
            df['provider_upcoding_rate'] = np.nan

        # --- Procedure-level aggregates ---
        proc_key = 'procedure_code_id' if 'procedure_code_id' in df.columns else None
        if proc_key:
            p_agg = df.groupby(proc_key).agg({
                'line_charge_amount': ['mean', 'count'],
                'is_upcoded': 'mean'
            }).reset_index()
            p_agg.columns = [proc_key, 'proc_avg_charge', 'proc_frequency', 'proc_upcoding_rate']
            df = df.merge(p_agg, on=proc_key, how='left')
        else:
            df['proc_avg_charge'] = np.nan
            df['proc_frequency'] = np.nan
            df['proc_upcoding_rate'] = np.nan

        # --- Textual lengths (procedure / authorization) ---
        if 'procedure_description' in df.columns:
            df['procedure_desc_length'] = df['procedure_description'].fillna('').str.len()
            df['procedure_desc_word_count'] = df['procedure_description'].fillna('').str.split().str.len()
        else:
            df['procedure_desc_length'] = 0
            df['procedure_desc_word_count'] = 0

        if 'authorization_description' in df.columns:
            df['authorization_desc_length'] = df['authorization_description'].fillna('').str.len()
            df['authorization_desc_word_count'] = df['authorization_description'].fillna('').str.split().str.len()
        else:
            df['authorization_desc_length'] = 0
            df['authorization_desc_word_count'] = 0

        return df

    def prepare_features(self, df: pd.DataFrame, target_col: str = 'is_upcoded') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""
        df = df.copy()

        # Separate target
        if target_col not in df.columns:
            raise ValueError(f"Target column `{target_col}` was not found.")
        y = df[target_col]
        X = df.drop(columns=[target_col], errors='ignore')

        # Drop columns that are pure identifiers or long text fields
        drop_cols = [
            # Identifiers
            'claim_id', 'line_sequence_no', 'claim_patient_account_no',
            'member_id', 'provider_id', 'provider_npi', 'group_id', 'group_name',
            # Free text and near-raw text leakage
            'service_provider_name', 'service_provider_city',
            'procedure_description', 'authorization_description',
            # Label artifacts
            'label', 'label_type',
            # Raw date columns (we already distilled durations)
            'claim_paid_date', 'claim_service_start_date', 'claim_service_end_date',
            'line_service_start_date', 'line_service_end_date',
            'admission_date', 'discharge_date', 'member_date_of_birth'
        ]
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors='ignore')
        
        sel_cols = [
            'age_at_service','claim_total_charge','diagnosis_code_2',
            'diagnosis_code_1','diagnosis_code_3','diagnosis_code_4','diagnosis_code_7','provider_line_count',
            'provider_total_line_charge','diagnosis_code_6','diagnosis_code_5','primary_diagnosis_code',
            'diagnosis_code_8','provider_avg_line_charge','provider_avg_line_units','service_provider_zip',
            'procedure_code_id','proc_avg_charge','claim_total_payable','bill_type',
            'provider_avg_line_allowed','diagnosis_count','provider_taxonomy_code',
            'service_provider_state','claim_service_duration_days','member_gender','admission_to_discharge_days',
            'unit_price','disallowed_amount','line_charge_amount','proc_frequency'
        ]
        
        X = X[sel_cols]
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str).fillna("NA"))
            else:
                X[col] = self.label_encoders[col].transform(X[col].astype(str).fillna("NA"))

        # Convert boolean to int if present
        bool_cols = X.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            X[col] = X[col].astype(int)

        # Handle missing numeric values
        for col in X.select_dtypes(include=[np.number]).columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        return X, y


# =========================
# Model Training
# =========================
class ModelTrainer:
    """Module for training individual ML models"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def get_models(self) -> Dict:
        """Initialize all ML models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=self.random_state, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        return models

    def train_model(self, model, X_train, y_train, X_test, y_test, model_name: str) -> Dict:
        """Train and evaluate a single model"""
        print(f"\nTraining {model_name}...")
        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Metrics
        results = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
            results['cv_f1_mean'] = cv_scores.mean()
            results['cv_f1_std'] = cv_scores.std()
        except Exception as e:
            results['cv_f1_mean'] = np.nan
            results['cv_f1_std'] = np.nan

        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}" if results['roc_auc'] else "ROC AUC: N/A")
        return results

    def train_all_models(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train all models and store results"""
        models = self.get_models()
        for name, model in models.items():
            self.results[name] = self.train_model(model, X_train, y_train, X_test, y_test, name)
            self.models[name] = self.results[name]['model']
        return self.results


# =========================
# Ensembles
# =========================
class EnsembleModeler:
    """Module for ensemble methods"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.ensemble_models = {}

    def create_voting_ensemble(self, base_models: Dict) -> VotingClassifier:
        estimators = [(name, model) for name, model in base_models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        return voting_clf

    def create_stacking_ensemble(self) -> StackingClassifier:
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=200, random_state=self.random_state)),
            ('xgb', XGBClassifier(n_estimators=200, random_state=self.random_state, eval_metric='logloss'))
        ]
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
        return stacking_clf

    def train_ensemble(self, ensemble_model, X_train, y_train, X_test, y_test, name: str) -> Dict:
        print(f"\nTraining {name}...")
        ensemble_model.fit(X_train, y_train)
        y_pred = ensemble_model.predict(X_test)
        y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
        results = {
            'model': ensemble_model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        return results


# =========================
# Export & Visualization
# =========================
class ResultsExporter:
    """Module for exporting results to CSV and visualization"""

    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_predictions_to_csv(self, results: Dict, filename: str = 'model_predictions.csv') -> str:
        filepath = os.path.join(self.output_dir, filename)
        # Get test set from first model
        first_model_results = next(iter(results.values()))
        y_test = first_model_results['y_test']
        export_df = pd.DataFrame({'actual': y_test.values})
        for model_name, result in results.items():
            export_df[f'{model_name}_pred'] = result['y_pred']
            if result['y_pred_proba'] is not None:
                export_df[f'{model_name}_proba'] = result['y_pred_proba']
        export_df.to_csv(filepath, index=False)
        print(f"\nPredictions exported to: {filepath}")
        return filepath

    def export_metrics_summary(self, results: Dict, filename: str = 'model_metrics.csv') -> str:
        filepath = os.path.join(self.output_dir, filename)
        metrics_list = []
        for model_name, result in results.items():
            metrics_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1'],
                'ROC AUC': result.get('roc_auc', np.nan),
                'CV F1 Mean': result.get('cv_f1_mean', np.nan),
                'CV F1 Std': result.get('cv_f1_std', np.nan)
            })
        metrics_df = pd.DataFrame(metrics_list).sort_values('F1 Score', ascending=False)
        metrics_df.to_csv(filepath, index=False)
        print(f"Metrics exported to: {filepath}")
        return filepath

    def plot_confusion_matrices(self, results: Dict, top_n: int = 4):
        sorted_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:top_n]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        for idx, (name, result) in enumerate(sorted_models):
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d',
                        ax=axes[idx], cmap='Blues', cbar=False)
            axes[idx].set_title(f'{name}\nF1: {result["f1"]:.4f}')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {filepath}")
        plt.show()

    def plot_roc_curves(self, results: Dict):
        fig, ax = plt.subplots(figsize=(12, 8))
        for model_name, result in results.items():
            if result.get('y_pred_proba') is not None:
                fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
                auc_score = roc_auc_score(result['y_test'], result['y_pred_proba'])
                ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc_score:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves - Model Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to: {filepath}")
        plt.show()

    def plot_metrics_comparison(self, results: Dict):
        metrics_list = []
        for model_name, result in results.items():
            metrics_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1'],
                'ROC AUC': result.get('roc_auc', np.nan)
            })
        df_metrics = pd.DataFrame(metrics_list).set_index('Model')
        df_metrics = df_metrics.sort_values('F1 Score', ascending=False)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        df_metrics.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Model Metrics Comparison', fontsize=13, fontweight='bold')
        axes[0].set_ylabel('Score')
        axes[0].legend(loc='best')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')
        sns.heatmap(df_metrics, annot=True, fmt='.3f', cmap='RdYlGn',
                    ax=axes[1], cbar_kws={'label': 'Score'})
        axes[1].set_title('Metrics Heatmap', fontsize=13, fontweight='bold')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to: {filepath}")
        plt.show()


# =========================
# Evaluation
# =========================
class ModelEvaluator:
    """Module for comprehensive model evaluation"""

    def compare_models(self, results: Dict) -> pd.DataFrame:
        comparison = []
        for name, result in results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1 Score': result['f1'],
                'ROC AUC': result.get('roc_auc', None),
                'CV F1 Mean': result.get('cv_f1_mean', None),
                'CV F1 Std': result.get('cv_f1_std', None)
            })
        comparison_df = pd.DataFrame(comparison).sort_values('F1 Score', ascending=False)
        return comparison_df

    def print_summary(self, comparison_df: pd.DataFrame):
        print("\n" + "="*100)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*100)
        print("\nDetailed Metrics:")
        print(comparison_df.to_string(index=False))
        print("\n" + "-"*100)
        print("SUMMARY STATISTICS")
        print("-"*100)
        numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns
        summary_stats = comparison_df[numeric_cols].describe()
        print("\n" + summary_stats.to_string())
        print("\n" + "-"*100)
        print("BEST PERFORMERS")
        print("-"*100)
        for col in numeric_cols:
            best_idx = comparison_df[col].idxmax()
            best_model = comparison_df.loc[best_idx, 'Model']
            best_score = comparison_df.loc[best_idx, col]
            print(f"Best {col}: {best_model} ({best_score:.4f})")


# =========================
# Main pipeline
# =========================
def main(db_path: str):
    print("="*100)
    print("HEALTHCARE CLAIMS UPCODING PREDICTION SYSTEM")
    print("(Schema: ClaimsData)")
    print("="*100)

    # 1. Load Data
    print("\n[1/8] Loading Data...")
    loader = DataLoader(db_path)
    df = loader.load_data()
    summary = loader.data_summary(df)
    if isinstance(summary.get('target_distribution'), pd.Series):
        print(f"\nTarget Distribution:\n{summary['target_distribution']}")
        print(f"\nTarget Percentage:\n{summary['target_percentage']}")
    else:
        print("\nTarget stats not available.")

    # 2. Feature Engineering
    print("\n[2/8] Engineering Features...")
    fe = FeatureEngineering()
    df_engineered = fe.engineer_features(df)
    print(f"Features after engineering: {df_engineered.shape[1]}")

    # 3. Prepare Features
    print("\n[3/8] Preparing Features...")
    X, y = fe.prepare_features(df_engineered, target_col='is_upcoded')
    print(f"Final feature set: {X.shape}")
    print(f"Features: {list(X.columns)}... (showing first 10)")

    # 4. Train-Test Split
    print("\n[4/8] Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 5. Train Individual Models
    print("\n[5/8] Training Individual Models...")
    print("="*100)
    trainer = ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)

    # 6. Train Ensemble Models
    print("\n[6/8] Training Ensemble Models...")
    print("="*100)
    ensemble = EnsembleModeler()
    top_models = dict(sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3])
    top_base_models = {name: results[name]['model'] for name in top_models.keys()}

    if len(top_base_models) >= 2:
        voting_clf = ensemble.create_voting_ensemble(top_base_models)
        results['Voting Ensemble'] = ensemble.train_ensemble(
            voting_clf, X_train, y_train, X_test, y_test, 'Voting Ensemble'
        )

    stacking_clf = ensemble.create_stacking_ensemble()
    results['Stacking Ensemble'] = ensemble.train_ensemble(
        stacking_clf, X_train, y_train, X_test, y_test, 'Stacking Ensemble'
    )

    # 7. Export Results
    print("\n[7/8] Exporting Results...")
    print("="*100)
    exporter = ResultsExporter(output_dir='outputs')
    exporter.export_predictions_to_csv(results)
    exporter.export_metrics_summary(results)

    # 8. Evaluate and Visualize
    print("\n[8/8] Generating Visualizations and Summary...")
    print("="*100)
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(results)
    evaluator.print_summary(comparison_df)

    # Visualizations
    exporter.plot_confusion_matrices(results)
    exporter.plot_roc_curves(results)
    exporter.plot_metrics_comparison(results)

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE - All outputs saved to './outputs' directory")
    print("="*100)
    return results, comparison_df, X.columns.tolist()


if __name__ == "__main__":
    db_path = "claims_database.db"
    results, comparison, feature_names = main(db_path)
