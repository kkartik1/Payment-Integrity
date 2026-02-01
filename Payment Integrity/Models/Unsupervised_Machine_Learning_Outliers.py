"""
Healthcare Payment Integrity - Unsupervised Outlier Detection
Adapter for SQLite DB: claims_database.db, Table: ClaimsData (schema provided)

Key mappings to internal features:
- provider_id <- Provider_ID
- claim_id <- Claim_ID
- total_charges_line <- Line_Charge_Amount
- allowed_amount <- Line_Allowed_Amount
- units <- Line_Units
- expected_payment <- Line_Paid_Amount
- service_date_line <- Line_Service_Start_Date
- service_date_header <- Claim_Service_Start_Date
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# (Optional) suppress oneDNN optimizations warnings (TensorFlow)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# -------------------------
# Data loading & adaptation
# -------------------------
class DataLoader:
    """Module for loading and preprocessing data from SQLite (ClaimsData)"""

    def __init__(self, db_path: str, table: str = "ClaimsData"):
        self.db_path = db_path
        self.table = table

    def _connect(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite DB not found at: {self.db_path}")
        return sqlite3.connect(self.db_path)

    def load_data(self, query: str = None) -> pd.DataFrame:
        """
        Load data from SQLite database.
        Default query selects basic fields and filters out clearly invalid lines.
        """
        if query is None:
            # Select * so we can flexibly remap; filter light to keep as much as possible
            query = f"""
            SELECT *
            FROM {self.table}
            WHERE Line_Allowed_Amount IS NOT NULL
              AND Line_Units IS NOT NULL
            """
        conn = self._connect()
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()

        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Standardize / remap to internal feature names the rest of the pipeline expects
        df = self._standardize_columns(df)
        df = self._parse_dates(df)
        df = self._derive_internal_fields(df)

        # Basic sanity filtering (optional, can be relaxed)
        df = df[df['units'].fillna(0) >= 0]
        df = df[df['allowed_amount'].fillna(0) >= 0]
        df = df[df['total_charges_line'].fillna(0) >= 0]

        return df

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename ClaimsData columns -> internal snake_case fields."""
        rename_map = {
            # IDs
            "Provider_ID": "provider_id",
            "Claim_ID": "claim_id",
            "Line_Sequence_No": "line_sequence_no",

            # Money & units (line level)
            "Line_Charge_Amount": "line_charge_amount",
            "Line_Paid_Amount": "line_paid_amount",
            "Line_Allowed_Amount": "line_allowed_amount",
            "Line_Units": "line_units",

            # Dates
            "Line_Service_Start_Date": "line_service_start_date",
            "Line_Service_End_Date": "line_service_end_date",
            "Claim_Service_Start_Date": "claim_service_start_date",
            "Claim_Service_End_Date": "claim_service_end_date",
            "Claim_Paid_Date": "claim_paid_date",

            # Optional header totals (not used directly in aggregation but retained)
            "Claim_Total_Charge": "claim_total_charge",
            "Claim_Total_Payable": "claim_total_payable",
        }

        # Only rename those present
        present_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=present_map)
        return df

    @staticmethod
    def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
        """Parse date strings to datetime; coerce errors to NaT."""
        for c in [
            "line_service_start_date",
            "line_service_end_date",
            "claim_service_start_date",
            "claim_service_end_date",
            "claim_paid_date",
        ]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    @staticmethod
    def _derive_internal_fields(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the exact internal fields expected by downstream modules.
        - service_date_line: primary time key (line-level start date)
        - service_date_header: claim header service start date (fallback)
        - total_charges_line, allowed_amount, units, expected_payment
        """
        df = df.copy()

        # service dates
        if "line_service_start_date" in df.columns:
            df["service_date_line"] = df["line_service_start_date"]
        elif "line_service_end_date" in df.columns:
            df["service_date_line"] = df["line_service_end_date"]
        else:
            df["service_date_line"] = pd.NaT

        if "claim_service_start_date" in df.columns:
            df["service_date_header"] = df["claim_service_start_date"]
        elif "claim_service_end_date" in df.columns:
            df["service_date_header"] = df["claim_service_end_date"]
        else:
            df["service_date_header"] = pd.NaT

        # amounts & units
        df["total_charges_line"] = df.get("line_charge_amount", pd.Series(np.nan, index=df.index))
        df["allowed_amount"] = df.get("line_allowed_amount", pd.Series(np.nan, index=df.index))
        df["units"] = df.get("line_units", pd.Series(np.nan, index=df.index))

        # Use Line_Paid_Amount as proxy for expected payment (available in schema)
        df["expected_payment"] = df.get("line_paid_amount", pd.Series(np.nan, index=df.index))

        # Ensure provider_id/claim_id exist
        if "provider_id" not in df.columns:
            df["provider_id"] = "UNKNOWN"
        if "claim_id" not in df.columns:
            # Create a synthetic claim ID if not present (should not happen with provided schema)
            df["claim_id"] = df.index.astype(str)

        return df


# -----------------------------
# Time-series feature engineering
# -----------------------------
class TimeSeriesFeatureEngine:
    """Module for creating time-series features for providers"""

    def __init__(self):
        self.provider_baselines = {}

    def create_provider_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create per-provider time series features based on service dates."""
        df = df.copy()

        # Ensure date columns are datetime
        if 'service_date_header' in df.columns:
            df['service_date_header'] = pd.to_datetime(df['service_date_header'])
        if 'service_date_line' in df.columns:
            df['service_date_line'] = pd.to_datetime(df['service_date_line'])

        # Use line date primarily; fallback to header date
        df['service_date'] = df['service_date_line'].fillna(df['service_date_header'])

        # Create time windows
        df['year_month'] = df['service_date'].dt.to_period('M')
        df['year_week'] = df['service_date'].dt.to_period('W')
        df['year_quarter'] = df['service_date'].dt.to_period('Q')

        print("Done create_provider_time_series ...")
        return df

    def aggregate_provider_metrics(self, df: pd.DataFrame, time_period: str = 'M') -> pd.DataFrame:
        """
        Aggregate provider metrics over time periods
        time_period: 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
        """
        period_col = {
            'M': 'year_month',
            'W': 'year_week',
            'Q': 'year_quarter'
        }[time_period]

        # Aggregate by provider and time period
        agg_metrics = df.groupby(['provider_id', period_col]).agg({
            'claim_id': 'count',          # Total claim-lines (proxy for volume)
            'total_charges_line': ['sum', 'mean', 'std', 'max'],
            'allowed_amount': ['sum', 'mean'],
            'units': ['sum', 'mean', 'max'],
            'expected_payment': ['sum', 'mean'],
        }).reset_index()

        # Flatten column names
        agg_metrics.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                               for col in agg_metrics.columns.values]

        # Rename for clarity
        rename_dict = {
            'claim_id_count': 'total_claims',
            'total_charges_line_sum': 'total_charges',
            'total_charges_line_mean': 'avg_charge_per_claim',
            'total_charges_line_std': 'charge_volatility',
            'total_charges_line_max': 'max_charge',
            'allowed_amount_sum': 'total_allowed',
            'allowed_amount_mean': 'avg_allowed',
            'units_sum': 'total_units',
            'units_mean': 'avg_units',
            'units_max': 'max_units',
            'expected_payment_sum': 'total_payment',
            'expected_payment_mean': 'avg_payment',
        }
        agg_metrics = agg_metrics.rename(columns=rename_dict)
        # --- Derived interpretability features for reason generation ---
        # Safe ratios at the period level
        agg_metrics['charge_to_allowed_ratio'] = agg_metrics.apply(
            lambda r: _ratio_safe(r.get('total_charges'), r.get('total_allowed')), axis=1
        )
        agg_metrics['paid_to_allowed_ratio'] = agg_metrics.apply(
            lambda r: _ratio_safe(r.get('total_payment'), r.get('total_allowed')), axis=1
        )
        agg_metrics['avg_charge_to_allowed_ratio'] = agg_metrics.apply(
            lambda r: _ratio_safe(r.get('avg_charge_per_claim'), r.get('avg_allowed')), axis=1
        )
        # Keep providers with a minimum activity to stabilize stats
        agg_metrics = agg_metrics[agg_metrics.total_claims > 5]
        print("Done aggregate_provider_metrics.")
        return agg_metrics

    def calculate_deviations(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate deviations from provider's historical baseline."""
        agg_df = agg_df.copy()

        # Sort within provider by the period column (2nd col)
        time_col = agg_df.columns[1]
        for provider in agg_df['provider_id'].unique():
            provider_mask = agg_df['provider_id'] == provider
            provider_data = agg_df[provider_mask].sort_values(time_col)

            for col in ['total_claims', 'total_charges', 'total_units', 'avg_charge_per_claim']:
                if col in provider_data.columns:
                    rolling_mean = provider_data[col].rolling(window=3, min_periods=1).mean()
                    rolling_std = provider_data[col].rolling(window=3, min_periods=1).std()

                    # Z-score deviation
                    agg_df.loc[provider_mask, f'{col}_zscore'] = (
                        (provider_data[col] - rolling_mean) / (rolling_std + 1e-10)
                    )

                    # Percentage change
                    agg_df.loc[provider_mask, f'{col}_pct_change'] = (
                        provider_data[col].pct_change() * 100
                    )

        agg_df = agg_df.fillna(0)
        return agg_df


# -----------------------------
# Anomaly detectors
# -----------------------------
class IsolationForestDetector:
    """Isolation Forest for detecting volume/charge spikes"""

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = RobustScaler()

    def fit_predict(self, X: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in ['id', 'period'])]

        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_features)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        predictions = self.model.fit_predict(X_scaled)
        anomaly_labels = (predictions == -1).astype(int)
        anomaly_scores = self.model.score_samples(X_scaled)  # lower = more anomalous

        return anomaly_labels, anomaly_scores


class LOFDetector:
    """Local Outlier Factor for density-based anomaly detection"""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.scaler = RobustScaler()

    def fit_predict(self, X: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in ['id', 'period'])]

        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_features)

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        predictions = self.model.fit_predict(X_scaled)
        anomaly_labels = (predictions == -1).astype(int)
        negative_outlier_factors = self.model.negative_outlier_factor_  # more negative = more anomalous

        return anomaly_labels, negative_outlier_factors


class AutoencoderDetector:
    """Autoencoder for reconstruction-based anomaly detection"""

    def __init__(self, encoding_dim: int = 16, contamination: float = 0.1, random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None

        np.random.seed(random_state)
        tf.random.set_seed(random_state)

    def build_autoencoder(self, input_dim: int) -> Model:
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return autoencoder

    def fit(self, X: pd.DataFrame, feature_cols: List[str] = None,
            epochs: int = 50, batch_size: int = 256, validation_split: float = 0.2):
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in ['id', 'period'])]

        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X_features)

        self.model = self.build_autoencoder(X_scaled.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )

        # Determine threshold from reconstruction error quantile
        reconstructions = self.model.predict(X_scaled, verbose=0)
        train_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(train_errors, 100 * (1 - self.contamination))
        return history

    def predict(self, X: pd.DataFrame, feature_cols: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if not any(x in c.lower() for x in ['id', 'period'])]

        X_features = X[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X_features)

        reconstructions = self.model.predict(X_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
        anomaly_labels = (reconstruction_errors > self.threshold).astype(int)
        return anomaly_labels, reconstruction_errors


# -----------------------------
# Analysis & exporting
# -----------------------------
class AnomalyAnalyzer:
    """Combine predictions and scores"""

    def __init__(self):
        self.results = {}

    def combine_predictions(self, predictions_dict: Dict[str, np.ndarray], method: str = 'voting') -> np.ndarray:
        predictions = np.array(list(predictions_dict.values()))
        if method == 'voting':
            combined = (predictions.sum(axis=0) >= len(predictions) / 2).astype(int)
        elif method == 'union':
            combined = (predictions.sum(axis=0) > 0).astype(int)
        elif method == 'intersection':
            combined = (predictions.sum(axis=0) == len(predictions)).astype(int)
        else:
            raise ValueError(f"Unknown combination method: {method}")
        return combined

    def calculate_anomaly_scores(self, scores_dict: Dict[str, np.ndarray]) -> np.ndarray:
        normalized_scores = []
        for name, scores in scores_dict.items():
            # Convert to anomaly scores (higher = more anomalous)
            if name in ['isolation_forest', 'lof']:
                scores = -scores
            smin, smax = scores.min(), scores.max()
            scores_norm = (scores - smin) / (smax - smin + 1e-10)
            normalized_scores.append(scores_norm)
        combined_score = np.mean(normalized_scores, axis=0)
        return combined_score
# -----------------------------
# Reason generator (explainability-lite)
# -----------------------------
def _ratio_safe(n, d):
    try:
        n = float(n)
        d = float(d)
        if d == 0:
            return np.nan
        return n / d
    except Exception:
        return np.nan

def generate_anomaly_reason_row(row) -> str:
    """
    Build concise, human-readable reasons for why this period was flagged anomalous.
    Uses robust, interpretable thresholds on deviation features and ratio outliers.
    Returns a short semicolon-separated string; empty string if no clear reason.
    """
    reasons = []

    # --- Ratio-based rules (current-period signals) ---
    # High charge vs allowed (potential overbilling or non-covered)
    ratio_charge_allowed = _ratio_safe(row.get('total_charges', np.nan), row.get('total_allowed', np.nan))
    if pd.notna(ratio_charge_allowed) and ratio_charge_allowed >= 2.0 and row.get('total_charges', 0) > 0:
        reasons.append(f"High charge-to-allowed ratio (~{ratio_charge_allowed:.2f}x)")

    # High paid vs allowed (pricing/contract anomaly or data issue)
    ratio_paid_allowed = _ratio_safe(row.get('total_payment', np.nan), row.get('total_allowed', np.nan))
    if pd.notna(ratio_paid_allowed) and ratio_paid_allowed >= 1.25 and row.get('total_payment', 0) > 0:
        reasons.append(f"Paid exceeds allowed (~{ratio_paid_allowed:.2f}x)")

    # High average charge per claim (level shift)
    if row.get('avg_charge_per_claim', 0) > 0 and row.get('avg_allowed', 0) > 0:
        ratio_avg_charge_allowed = _ratio_safe(row['avg_charge_per_claim'], row['avg_allowed'])
        if pd.notna(ratio_avg_charge_allowed) and ratio_avg_charge_allowed >= 1.75:
            reasons.append(f"Elevated avg charge per claim vs allowed (~{ratio_avg_charge_allowed:.2f}x)")

    # --- Deviation rules (provider-relative history signals) ---
    # Strong z-score spikes
    if abs(row.get('total_charges_zscore', 0)) >= 3:
        reasons.append(f"Charge spike (z={row['total_charges_zscore']:.1f})")
    if abs(row.get('total_units_zscore', 0)) >= 3:
        reasons.append(f"Units spike (z={row['total_units_zscore']:.1f})")
    if abs(row.get('avg_charge_per_claim_zscore', 0)) >= 3:
        reasons.append(f"Avg charge per claim spike (z={row['avg_charge_per_claim_zscore']:.1f})")
    if abs(row.get('total_claims_zscore', 0)) >= 3:
        reasons.append(f"Volume spike (z={row['total_claims_zscore']:.1f})")

    # Large sudden percentage changes
    if abs(row.get('total_charges_pct_change', 0)) >= 100:
        reasons.append(f"Charges changed {row['total_charges_pct_change']:.0f}% vs prior")
    if abs(row.get('total_units_pct_change', 0)) >= 100:
        reasons.append(f"Units changed {row['total_units_pct_change']:.0f}% vs prior")
    if abs(row.get('avg_charge_per_claim_pct_change', 0)) >= 75:
        reasons.append(f"Avg charge/claim changed {row['avg_charge_per_claim_pct_change']:.0f}% vs prior")
    if abs(row.get('total_claims_pct_change', 0)) >= 100:
        reasons.append(f"Volume changed {row['total_claims_pct_change']:.0f}% vs prior")

    # Keep it concise—return at most 3 top reasons
    if not reasons:
        return ""
    return "; ".join(reasons[:3])

class ResultsExporter:
    """Export and visualize anomaly detection results"""

    def __init__(self, output_dir: str = 'outputs/anomaly_detection'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_anomalies(self, df: pd.DataFrame,
                         predictions_dict: Dict[str, np.ndarray],
                         scores_dict: Dict[str, np.ndarray],
                         filename: str = 'anomaly_results.csv') -> str:
        filepath = os.path.join(self.output_dir, filename)
        export_df = df.copy()

        # Add predictions
        for model_name, predictions in predictions_dict.items():
            export_df[f'{model_name}_anomaly'] = predictions

        # Add scores
        for model_name, scores in scores_dict.items():
            export_df[f'{model_name}_score'] = scores

        # Add combined prediction
        analyzer = AnomalyAnalyzer()
        export_df['combined_anomaly'] = analyzer.combine_predictions(predictions_dict, method='voting')
        export_df['combined_score'] = analyzer.calculate_anomaly_scores(scores_dict)

        # NEW: Add concise reasons only where combined anomaly == 1
        export_df['anomaly_reason'] = ""
        anomaly_mask = export_df['combined_anomaly'] == 1
        if anomaly_mask.any():
            export_df.loc[anomaly_mask, 'anomaly_reason'] = export_df[anomaly_mask].apply(
                generate_anomaly_reason_row, axis=1
            )
            
        export_df.to_csv(filepath, index=False)
        print(f"\nAnomaly results exported to: {filepath}")
        return filepath

    def export_summary_stats(self, df: pd.DataFrame,
                             predictions_dict: Dict[str, np.ndarray],
                             filename: str = 'anomaly_summary.csv') -> str:
        filepath = os.path.join(self.output_dir, filename)
        summary_list = []
        for model_name, predictions in predictions_dict.items():
            n_anomalies = predictions.sum()
            anomaly_rate = (n_anomalies / len(predictions)) * 100

            anomaly_mask = predictions == 1
            if anomaly_mask.sum() > 0 and 'total_charges' in df.columns:
                avg_anomaly_charge = df.loc[anomaly_mask, 'total_charges'].mean()
                avg_normal_charge = df.loc[~anomaly_mask, 'total_charges'].mean()
            else:
                avg_anomaly_charge = np.nan
                avg_normal_charge = np.nan

            summary_list.append({
                'Model': model_name,
                'Total_Records': len(predictions),
                'Anomalies_Detected': int(n_anomalies),
                'Anomaly_Rate_%': anomaly_rate,
                'Avg_Anomaly_Charge': avg_anomaly_charge,
                'Avg_Normal_Charge': avg_normal_charge,
                'Charge_Ratio': (avg_anomaly_charge / avg_normal_charge) if (avg_normal_charge and avg_normal_charge > 0) else np.nan
            })

        summary_df = pd.DataFrame(summary_list)
        summary_df.to_csv(filepath, index=False)
        print(f"Summary statistics exported to: {filepath}")
        return filepath

    def plot_anomaly_distribution(self, df: pd.DataFrame,
                                  predictions_dict: Dict[str, np.ndarray]):
        """Plot distribution of anomalies across models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        # Plot 1: Counts
        model_names = list(predictions_dict.keys())
        anomaly_counts = [pred.sum() for pred in predictions_dict.values()]
        axes[0].bar(model_names, anomaly_counts, color='coral', alpha=0.7)
        axes[0].set_title('Anomalies Detected by Each Model', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Anomalies')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(alpha=0.3, axis='y')

        # Plot 2: Overlap (best-effort if venn available)
        try:
            from matplotlib_venn import venn2, venn3
            if len(predictions_dict) == 2:
                sets = [set(np.where(pred == 1)[0]) for pred in predictions_dict.values()]
                venn2(sets, set_labels=model_names, ax=axes[1])
            elif len(predictions_dict) == 3:
                sets = [set(np.where(pred == 1)[0]) for pred in predictions_dict.values()]
                venn3(sets, set_labels=model_names, ax=axes[1])
            axes[1].set_title('Anomaly Overlap Between Models', fontsize=12, fontweight='bold')
        except Exception:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, "Install matplotlib-venn for overlap plot", ha='center')

        # Plot 3: Time series if available
        if 'year_month' in df.columns or 'year_week' in df.columns:
            time_col = 'year_month' if 'year_month' in df.columns else 'year_week'
            for model_name, predictions in predictions_dict.items():
                temp_df = df.copy()
                temp_df['anomaly'] = predictions
                anomaly_ts = temp_df.groupby(time_col)['anomaly'].sum()
                axes[2].plot(range(len(anomaly_ts)), anomaly_ts.values, marker='o', label=model_name, linewidth=2)
            axes[2].set_title('Anomalies Over Time', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('Time Period')
            axes[2].set_ylabel('Number of Anomalies')
            axes[2].legend()
            axes[2].grid(alpha=0.3)

        # Plot 4: Charge distribution
        if 'total_charges' in df.columns:
            analyzer = AnomalyAnalyzer()
            combined = analyzer.combine_predictions(predictions_dict, method='voting')
            normal_charges = df.loc[combined == 0, 'total_charges']
            anomaly_charges = df.loc[combined == 1, 'total_charges']
            axes[3].hist(normal_charges, bins=50, alpha=0.6, label='Normal', color='blue')
            axes[3].hist(anomaly_charges, bins=50, alpha=0.6, label='Anomaly', color='red')
            axes[3].set_title('Charge Distribution: Normal vs Anomaly', fontsize=12, fontweight='bold')
            axes[3].set_xlabel('Total Charges')
            axes[3].set_ylabel('Frequency')
            axes[3].legend()
            axes[3].set_yscale('log')

        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'anomaly_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Anomaly distribution plot saved to: {filepath}")
        plt.show()

    def plot_anomaly_scores(self, scores_dict: Dict[str, np.ndarray]):
        """Plot anomaly score distributions"""
        fig, axes = plt.subplots(1, len(scores_dict), figsize=(15, 4))
        if len(scores_dict) == 1:
            axes = [axes]
        for idx, (model_name, scores) in enumerate(scores_dict.items()):
            axes[idx].hist(scores, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{model_name}\nScore Distribution', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Anomaly Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3, axis='y')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'anomaly_scores.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Anomaly scores plot saved to: {filepath}")
        plt.show()


# -----------------------------
# Orchestration
# -----------------------------
def main(db_path: str, time_period: str = 'M', contamination: float = 0.1):
    """
    Main execution pipeline for unsupervised anomaly detection
    Parameters:
    - db_path: Path to SQLite database (claims_database.db)
    - time_period: 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
    - contamination: Expected proportion of outliers (0.05-0.15 typical)
    """
    print("="*100)
    print("HEALTHCARE PAYMENT INTEGRITY - UNSUPERVISED ANOMALY DETECTION")
    print("="*100)

    # 1. Load Data
    print("\n[1/6] Loading Data...")
    loader = DataLoader(db_path, table="ClaimsData")
    df = loader.load_data()

    # 2. Create Time Series Features
    print("\n[2/6] Creating Time Series Features...")
    ts_engine = TimeSeriesFeatureEngine()
    df_ts = ts_engine.create_provider_time_series(df)

    # Aggregate by provider and time period
    agg_df = ts_engine.aggregate_provider_metrics(df_ts, time_period=time_period)
    print(f"Aggregated data: {agg_df.shape}")

    # Calculate deviations from baseline
    agg_df = ts_engine.calculate_deviations(agg_df)
    print(f"Features with deviations: {agg_df.shape[1]}")

    # 3. Isolation Forest Detection
    print("\n[3/6] Running Isolation Forest...")
    if_detector = IsolationForestDetector(contamination=contamination)
    if_labels, if_scores = if_detector.fit_predict(agg_df)
    print(f"Anomalies detected: {if_labels.sum()} ({if_labels.mean()*100:.2f}%)")

    # 4. LOF Detection
    print("\n[4/6] Running Local Outlier Factor...")
    lof_detector = LOFDetector(n_neighbors=20, contamination=contamination)
    lof_labels, lof_scores = lof_detector.fit_predict(agg_df)
    print(f"Anomalies detected: {lof_labels.sum()} ({lof_labels.mean()*100:.2f}%)")

    # 5. Autoencoder Detection
    print("\n[5/6] Training Autoencoder...")
    ae_detector = AutoencoderDetector(encoding_dim=16, contamination=contamination)
    ae_detector.fit(agg_df, epochs=50, batch_size=128)
    ae_labels, ae_errors = ae_detector.predict(agg_df)
    print(f"Anomalies detected: {ae_labels.sum()} ({ae_labels.mean()*100:.2f}%)")

    # 6. Export and Visualize Results
    print("\n[6/6] Exporting Results and Creating Visualizations...")
    print("="*100)
    predictions_dict = {
        'isolation_forest': if_labels,
        'lof': lof_labels,
        'autoencoder': ae_labels
    }
    scores_dict = {
        'isolation_forest': if_scores,
        'lof': lof_scores,
        'autoencoder': ae_errors
    }

    exporter = ResultsExporter()
    exporter.export_anomalies(agg_df, predictions_dict, scores_dict)
    exporter.export_summary_stats(agg_df, predictions_dict)

    exporter.plot_anomaly_distribution(agg_df, predictions_dict)
    exporter.plot_anomaly_scores(scores_dict)

    # Combined analysis
    analyzer = AnomalyAnalyzer()
    combined_labels = analyzer.combine_predictions(predictions_dict, method='voting')
    combined_scores = analyzer.calculate_anomaly_scores(scores_dict)

    print("\n" + "="*100)
    print("ANOMALY DETECTION SUMMARY")
    print("="*100)
    print(f"\nIsolation Forest: {if_labels.sum()} anomalies ({if_labels.mean()*100:.2f}%)")
    print(f"LOF: {lof_labels.sum()} anomalies ({lof_labels.mean()*100:.2f}%)")
    print(f"Autoencoder: {ae_labels.sum()} anomalies ({ae_labels.mean()*100:.2f}%)")
    print(f"Combined (Voting): {combined_labels.sum()} anomalies ({combined_labels.mean()*100:.2f}%)")

    agreement = (if_labels == lof_labels).sum() / len(if_labels) * 100
    print(f"\nAgreement between IF and LOF: {agreement:.1f}%")
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE - All outputs saved to './outputs/anomaly_detection' directory")
    print("="*100)

    return {
        'aggregated_data': agg_df,
        'predictions': predictions_dict,
        'scores': scores_dict,
        'combined_labels': combined_labels,
        'combined_scores': combined_scores,
        'models': {
            'isolation_forest': if_detector,
            'lof': lof_detector,
            'autoencoder': ae_detector
        }
    }


if __name__ == "__main__":
    # By default, look for DB in current working directory
    db_path = "claims_database.db"
    # Run analysis with monthly aggregation
    results = main(db_path, time_period='M', contamination=0.10)
    
    # Access results
    # results['aggregated_data'] - DataFrame with time-series features
    # results['predictions'] - Dictionary of anomaly predictions from each model
    # results['scores'] - Dictionary of anomaly scores from each model
    # results['combined_labels'] - Combined anomaly predictions (voting)
    # results['models'] - Trained model objects for future predictions