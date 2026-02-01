
# Healthcare Payment Integrity - Graph-Based Network Analysis
# Provider-Patient-Procedure graph analytics with embeddings and centrality measures
# Detects unusual referral patterns and DME supply chains
#
# Updated to use SQLite DB: `claims_database.db` and table: `ClaimsData` with the provided schema.
# Key field mappings (SQL aliases used so the rest of the pipeline remains unchanged):
#     Claim_ID -> claim_id (TEXT)
#     Member_ID -> patient_id (TEXT)
#     Provider_ID -> provider_id (TEXT)
#     Provider_NPI -> provider_npi (TEXT)
#     Provider_Taxonomy_Code -> provider_specialty (TEXT, used as proxy for specialty)
#     Procedure_Code_ID -> hcpcs_code (TEXT)
#     Procedure_Description -> procedure_description (TEXT)
#     Line_Charge_Amount -> total_charges_line (REAL)
#     Line_Allowed_Amount -> allowed_amount (REAL)
#     Line_Units -> units (INTEGER)
#     Line_Service_Start_Date -> service_date_line (TEXT -> parsed to datetime)

import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional, Set
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Graph Libraries
import networkx as nx
from networkx.algorithms import community
import scipy.sparse as sp

# Graph Embeddings
from node2vec import Node2Vec
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# PyTorch for GraphSAGE (optional - simplified implementation included)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GraphSAGE will use simplified implementation.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


class DataLoader:
    """Module for loading and preprocessing graph data from SQLite ClaimsData"""
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _connect(self):
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"SQLite DB not found at {self.db_path}")
        return sqlite3.connect(self.db_path)

    def load_data(self, query: str = None) -> pd.DataFrame:
        """Load generic data from SQLite database"""
        if query is None:
            # Default: use ClaimsData and new field names
            query = (
                "SELECT * FROM ClaimsData WHERE IFNULL(Line_Allowed_Amount,0) > 1000"
            )
        conn = self._connect()
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    def load_for_graph_analysis(self) -> pd.DataFrame:
        """Load data specifically for graph construction, mapping fields with SQL aliases."""
        query = """
            SELECT
                Claim_ID                AS claim_id,
                Member_ID               AS patient_id,
                Provider_ID             AS provider_id,
                Provider_NPI            AS provider_npi,
                Provider_Taxonomy_Code  AS provider_specialty,
                Procedure_Code_ID       AS hcpcs_code,
                Procedure_Description   AS procedure_description,
                Line_Charge_Amount      AS total_charges_line,
                Line_Allowed_Amount     AS allowed_amount,
                Line_Units              AS units,
                Line_Service_Start_Date AS service_date_line
            FROM ClaimsData
            WHERE IFNULL(Line_Allowed_Amount,0) > 3000
        """
        conn = self._connect()
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        # Clean and prepare
        df['service_date_line'] = pd.to_datetime(df['service_date_line'], errors='coerce')
        # Coerce numerics and handle missing
        for col in ['total_charges_line', 'allowed_amount', 'units']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        # Drop rows missing critical identifiers
        df = df.dropna(subset=['claim_id', 'patient_id', 'provider_id', 'hcpcs_code'])
        return df


class HealthcareGraph:
    """Module for constructing healthcare network graphs"""
    def __init__(self):
        self.graph = None
        self.node_types = {}
        self.edge_types = {}

    def build_tripartite_graph(self, df: pd.DataFrame) -> nx.Graph:
        """
        Build Provider-Patient-Procedure tripartite graph
        """
        G = nx.Graph()
        print("Building tripartite graph...")

        # Add nodes with attributes
        providers = df[['provider_id', 'provider_specialty']].drop_duplicates()
        for _, row in providers.iterrows():
            G.add_node(f"P_{row['provider_id']}",
                       node_type='provider',
                       specialty=row.get('provider_specialty', None))
            self.node_types[f"P_{row['provider_id']}"] = 'provider'

        patients = df['patient_id'].dropna().unique()
        for patient in patients:
            G.add_node(f"PT_{patient}", node_type='patient')
            self.node_types[f"PT_{patient}"] = 'patient'

        procedures = df[['hcpcs_code', 'procedure_description']].drop_duplicates()
        for _, row in procedures.iterrows():
            G.add_node(f"PR_{row['hcpcs_code']}",
                       node_type='procedure',
                       description=row.get('procedure_description', None))
            self.node_types[f"PR_{row['hcpcs_code']}"] = 'procedure'

        # Add edges with weights (interaction frequency and total charges)
        print("Adding edges...")
        # Provider-Patient edges
        provider_patient = df.groupby(['provider_id', 'patient_id']).agg({
            'claim_id': 'count',
            'total_charges_line': 'sum'
        }).reset_index()
        for _, row in provider_patient.iterrows():
            G.add_edge(f"P_{row['provider_id']}", f"PT_{row['patient_id']}",
                       edge_type='treats',
                       weight=row['claim_id'],
                       total_charges=row['total_charges_line'])

        # Patient-Procedure edges
        patient_procedure = df.groupby(['patient_id', 'hcpcs_code']).agg({
            'claim_id': 'count',
            'total_charges_line': 'sum'
        }).reset_index()
        for _, row in patient_procedure.iterrows():
            G.add_edge(f"PT_{row['patient_id']}", f"PR_{row['hcpcs_code']}",
                       edge_type='receives',
                       weight=row['claim_id'],
                       total_charges=row['total_charges_line'])

        # Provider-Procedure edges (derived from claims)
        provider_procedure = df.groupby(['provider_id', 'hcpcs_code']).agg({
            'claim_id': 'count',
            'total_charges_line': 'sum',
            'units': 'sum'
        }).reset_index()
        for _, row in provider_procedure.iterrows():
            G.add_edge(f"P_{row['provider_id']}", f"PR_{row['hcpcs_code']}",
                       edge_type='performs',
                       weight=row['claim_id'],
                       total_charges=row['total_charges_line'],
                       total_units=row['units'])

        self.graph = G
        print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Providers: {sum(1 for n in G.nodes() if n.startswith('P_'))}")
        print(f"Patients: {sum(1 for n in G.nodes() if n.startswith('PT_'))}")
        print(f"Procedures: {sum(1 for n in G.nodes() if n.startswith('PR_'))}")
        return G

    def build_provider_referral_network(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build directed provider referral network
        Inferred from patients seeing multiple providers
        """
        G = nx.DiGraph()
        print("Building provider referral network...")
        # Group by patient and get their providers in chronological order
        df_sorted = df.sort_values('service_date_line')
        patient_providers = df_sorted.groupby('patient_id')['provider_id'].apply(list)
        referral_counts = defaultdict(int)
        for providers in patient_providers:
            # Create edges between consecutive providers (implied referrals)
            for i in range(len(providers) - 1):
                referral_counts[(providers[i], providers[i+1])] += 1
        # Add nodes and edges
        for (source, target), count in referral_counts.items():
            if not G.has_node(source):
                G.add_node(source, node_type='provider')
            if not G.has_node(target):
                G.add_node(target, node_type='provider')
            G.add_edge(source, target, weight=count, edge_type='referral')
        print(f"Referral network: {G.number_of_nodes()} providers, {G.number_of_edges()} referral patterns")
        return G


class CentralityAnalyzer:
    """Module for computing network centrality measures"""
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.centrality_metrics = {}

    def compute_all_centralities(self) -> pd.DataFrame:
        """Compute multiple centrality measures"""
        print("Computing centrality measures...")
        # Degree centrality
        print(" - Degree centrality...")
        degree_cent = nx.degree_centrality(self.graph)
        # Betweenness centrality
        print(" - Betweenness centrality...")
        betweenness_cent = nx.betweenness_centrality(self.graph, weight='weight')
        # Closeness centrality
        print(" - Closeness centrality...")
        closeness_cent = nx.closeness_centrality(self.graph)
        # Eigenvector centrality (may fail for disconnected graphs)
        print(" - Eigenvector centrality...")
        try:
            eigenvector_cent = nx.eigenvector_centrality(self.graph, max_iter=1000, weight='weight')
        except Exception:
            eigenvector_cent = {node: 0 for node in self.graph.nodes()}
            print(" Warning: Eigenvector centrality failed, using zeros")
        # PageRank
        print(" - PageRank...")
        pagerank_cent = nx.pagerank(self.graph, weight='weight')
        # Clustering coefficient
        print(" - Clustering coefficient...")
        clustering_coef = nx.clustering(self.graph)
        # Compile into dataframe
        centrality_df = pd.DataFrame({
            'node': list(self.graph.nodes()),
            'degree_centrality': [degree_cent[n] for n in self.graph.nodes()],
            'betweenness_centrality': [betweenness_cent[n] for n in self.graph.nodes()],
            'closeness_centrality': [closeness_cent[n] for n in self.graph.nodes()],
            'eigenvector_centrality': [eigenvector_cent[n] for n in self.graph.nodes()],
            'pagerank': [pagerank_cent[n] for n in self.graph.nodes()],
            'clustering_coefficient': [clustering_coef[n] for n in self.graph.nodes()],
            'degree': [self.graph.degree(n) for n in self.graph.nodes()]
        })
        # Add node type
        centrality_df['node_type'] = centrality_df['node'].apply(
            lambda x: 'provider' if x.startswith('P_') and not x.startswith('PT_')
            else 'patient' if x.startswith('PT_')
            else 'procedure'
        )
        self.centrality_metrics = centrality_df
        print(f"Centrality metrics computed for {len(centrality_df)} nodes")
        return centrality_df

    def identify_hubs(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """Identify hub nodes (highly central nodes)"""
        if self.centrality_metrics is None or len(self.centrality_metrics) == 0:
            self.compute_all_centralities()
        hubs = {}
        # Top nodes by different centrality measures
        for metric in ['degree_centrality', 'betweenness_centrality', 'pagerank']:
            hubs[metric] = self.centrality_metrics.nlargest(top_n, metric)[
                ['node', 'node_type', metric, 'degree']
            ]
        return hubs

    def detect_anomalous_centrality(self, threshold_percentile: float = 95) -> pd.DataFrame:
        """Detect nodes with unusually high centrality (potential fraud indicators)"""
        if self.centrality_metrics is None or len(self.centrality_metrics) == 0:
            self.compute_all_centralities()
        anomalies = []
        for node_type in ['provider', 'patient', 'procedure']:
            type_df = self.centrality_metrics[self.centrality_metrics['node_type'] == node_type]
            if len(type_df) == 0:
                continue
            for metric in ['degree_centrality', 'betweenness_centrality', 'pagerank']:
                threshold = type_df[metric].quantile(threshold_percentile / 100)
                anomalous = type_df[type_df[metric] > threshold].copy()
                anomalous['anomaly_type'] = f'high_{metric}'
                anomalies.append(anomalous)
        if anomalies:
            anomaly_df = pd.concat(anomalies, ignore_index=True)
            return anomaly_df
        else:
            return pd.DataFrame()


class Node2VecEmbedding:
    """Module for Node2Vec embeddings"""
    def __init__(self, dimensions: int = 64, walk_length: int = 30,
                 num_walks: int = 200, workers: int = 4):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.model = None
        self.embeddings = None

    def fit(self, graph: nx.Graph, p: float = 1, q: float = 1) -> np.ndarray:
        """
        Train Node2Vec embeddings
        p: return parameter (likelihood of returning to previous node)
        q: in-out parameter (BFS vs DFS)
        """
        print("Training Node2Vec embeddings...")
        print(f" Dimensions: {self.dimensions}")
        print(f" Walk length: {self.walk_length}")
        print(f" Num walks: {self.num_walks}")
        print(f" p={p}, q={q}")
        # Initialize Node2Vec
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers,
            p=p,
            q=q,
            weight_key='weight'
        )
        # Train model
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)
        # Get embeddings
        embeddings = {}
        for node in graph.nodes():
            embeddings[node] = self.model.wv[node]
        self.embeddings = embeddings
        print(f"Embeddings generated for {len(embeddings)} nodes")
        return embeddings

    def get_embedding_dataframe(self) -> pd.DataFrame:
        """Convert embeddings to dataframe"""
        if self.embeddings is None:
            raise ValueError("Must train embeddings first")
        embed_df = pd.DataFrame.from_dict(self.embeddings, orient='index')
        embed_df.index.name = 'node'
        embed_df.reset_index(inplace=True)
        # Add node type
        embed_df['node_type'] = embed_df['node'].apply(
            lambda x: 'provider' if x.startswith('P_') and not x.startswith('PT_')
            else 'patient' if x.startswith('PT_')
            else 'procedure'
        )
        return embed_df

    def cluster_embeddings(self, n_clusters: int = 5) -> Tuple[np.ndarray, float]:
        """Cluster nodes based on embeddings"""
        if self.embeddings is None:
            raise ValueError("Must train embeddings first")
        # Get embeddings matrix
        nodes = list(self.embeddings.keys())
        X = np.array([self.embeddings[node] for node in nodes])
        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        # Silhouette score
        silhouette = silhouette_score(X, clusters)
        print(f"Clustering complete: {n_clusters} clusters, silhouette score: {silhouette:.3f}")
        return clusters, silhouette

    def find_similar_nodes(self, node: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar nodes based on embedding similarity"""
        if self.model is None:
            raise ValueError("Must train embeddings first")
        similar = self.model.wv.most_similar(node, topn=top_k)
        return similar


class CommunityDetector:
    """Module for community detection in networks"""
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.communities = None

    def detect_communities(self, method: str = 'louvain') -> Dict[Any, int]:
        """
        Detect communities using various algorithms
        method: 'louvain', 'greedy_modularity', 'label_propagation'
        """
        print(f"Detecting communities using {method}...")
        if method == 'louvain':
            # Louvain method (requires python-louvain)
            try:
                import community as community_louvain
                self.communities = community_louvain.best_partition(self.graph, weight='weight')
            except ImportError:
                print("python-louvain not available, using greedy modularity instead")
                method = 'greedy_modularity'
        if method == 'greedy_modularity':
            communities_generator = community.greedy_modularity_communities(self.graph, weight='weight')
            communities_list = list(communities_generator)
            self.communities = {}
            for idx, comm in enumerate(communities_list):
                for node in comm:
                    self.communities[node] = idx
        elif method == 'label_propagation':
            communities_generator = community.label_propagation_communities(self.graph)
            communities_list = list(communities_generator)
            self.communities = {}
            for idx, comm in enumerate(communities_list):
                for node in comm:
                    self.communities[node] = idx
        n_communities = len(set(self.communities.values()))
        print(f"Detected {n_communities} communities")
        return self.communities

    def analyze_communities(self) -> pd.DataFrame:
        """Analyze community composition and characteristics"""
        # Ensure communities exist
        if self.communities is None or len(self.communities) == 0:
            # Attempt detection with a safe default
            detected = self.detect_communities(method='greedy_modularity')
            if detected is None or len(detected) == 0:
                # Return an empty but well-formed DataFrame
                return pd.DataFrame(
                    columns=['community_id', 'size', 'providers', 'patients', 'procedures', 'edges', 'density']
                )

        community_stats = []
        # If we are here, self.communities should have entries
        comm_ids = set(self.communities.values()) if len(self.communities) else set()
        if not comm_ids:
            return pd.DataFrame(
                columns=['community_id', 'size', 'providers', 'patients', 'procedures', 'edges', 'density']
            )

        for comm_id in comm_ids:
            nodes_in_comm = [n for n, c in self.communities.items() if c == comm_id]
            # Count by node type
            providers = sum(1 for n in nodes_in_comm if n.startswith('P_') and not n.startswith('PT_'))
            patients = sum(1 for n in nodes_in_comm if n.startswith('PT_'))
            procedures = sum(1 for n in nodes_in_comm if n.startswith('PR_'))
            # Subgraph for metrics
            subgraph = self.graph.subgraph(nodes_in_comm)
            community_stats.append({
                'community_id': comm_id,
                'size': len(nodes_in_comm),
                'providers': providers,
                'patients': patients,
                'procedures': procedures,
                'edges': subgraph.number_of_edges(),
                'density': nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0,
            })

        if not community_stats:
            # Still return the same schema
            return pd.DataFrame(
                columns=['community_id', 'size', 'providers', 'patients', 'procedures', 'edges', 'density']
            )

        return pd.DataFrame(community_stats).sort_values('size', ascending=False)



class ReferralPatternAnalyzer:
    """Module for analyzing referral patterns and supply chains"""
    def __init__(self, referral_graph: nx.DiGraph):
        self.graph = referral_graph

    def detect_referral_loops(self) -> List[List[str]]:
        """Detect circular referral patterns (potential fraud)"""
        print("Detecting circular referral patterns...")
        # Find all cycles
        cycles = list(nx.simple_cycles(self.graph))
        # Filter cycles by length (2-5 providers)
        suspicious_cycles = [c for c in cycles if 2 <= len(c) <= 5]
        print(f"Found {len(suspicious_cycles)} potential referral loops")
        return suspicious_cycles

    def identify_referral_hubs(self, top_n: int = 20) -> pd.DataFrame:
        """Identify providers who are hubs in referral network"""
        in_degree = dict(self.graph.in_degree(weight='weight'))
        out_degree = dict(self.graph.out_degree(weight='weight'))
        hub_data = []
        for node in self.graph.nodes():
            hub_data.append({
                'provider_id': node,
                'referrals_received': in_degree.get(node, 0),
                'referrals_sent': out_degree.get(node, 0),
                'total_referrals': in_degree.get(node, 0) + out_degree.get(node, 0),
                'referral_ratio': out_degree.get(node, 0) / (in_degree.get(node, 0) + 1)
            })
        hub_df = pd.DataFrame(hub_data).sort_values('total_referrals', ascending=False).head(top_n)
        return hub_df

    def analyze_supply_chains(self, min_chain_length: int = 3) -> List[List[str]]:
        """
        Identify DME supply chains (long referral paths)
        Potentially indicates kickback schemes
        """
        print(f"Analyzing supply chains (min length: {min_chain_length})...")
        long_paths = []
        # Find longest paths between all pairs
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source != target:
                    try:
                        # Get all simple paths
                        paths = nx.all_simple_paths(self.graph, source, target,
                                                    cutoff=min_chain_length + 2)
                        for path in paths:
                            if len(path) >= min_chain_length:
                                long_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        # Sort by length
        long_paths.sort(key=len, reverse=True)
        print(f"Found {len(long_paths)} supply chains of length >= {min_chain_length}")
        return long_paths[:100]  # Return top 100


class ResultsExporter:
    """Module for exporting graph analysis results"""
    def __init__(self, output_dir: str = 'outputs/graph_analysis'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_centrality_metrics(self, centrality_df: pd.DataFrame,
                                  filename: str = 'centrality_metrics.csv') -> str:
        """Export centrality metrics to CSV"""
        filepath = os.path.join(self.output_dir, filename)
        centrality_df.to_csv(filepath, index=False)
        print(f"Centrality metrics exported to: {filepath}")
        return filepath

    def export_embeddings(self, embed_df: pd.DataFrame,
                          filename: str = 'node_embeddings.csv') -> str:
        """Export node embeddings to CSV"""
        filepath = os.path.join(self.output_dir, filename)
        embed_df.to_csv(filepath, index=False)
        print(f"Node embeddings exported to: {filepath}")
        return filepath

    def export_anomalies(self, anomaly_df: pd.DataFrame,
                          filename: str = 'graph_anomalies.csv') -> str:
        """Export detected anomalies"""
        filepath = os.path.join(self.output_dir, filename)
        anomaly_df.to_csv(filepath, index=False)
        print(f"Graph anomalies exported to: {filepath}")
        return filepath

    def export_referral_analysis(self,
                                 hubs_df: pd.DataFrame,
                                 loops: List[List[str]],
                                 chains: List[List[str]]) -> str:
        """Export referral pattern analysis"""
        # Hubs
        hubs_path = os.path.join(self.output_dir, 'referral_hubs.csv')
        hubs_df.to_csv(hubs_path, index=False)
        # Loops
        if loops:
            loops_df = pd.DataFrame({
                'loop_id': range(len(loops)),
                'providers': [' -> '.join(loop) for loop in loops],
                'length': [len(loop) for loop in loops]
            })
            loops_path = os.path.join(self.output_dir, 'referral_loops.csv')
            loops_df.to_csv(loops_path, index=False)
        # Supply chains
        if chains:
            chains_df = pd.DataFrame({
                'chain_id': range(len(chains)),
                'providers': [' -> '.join(chain) for chain in chains],
                'length': [len(chain) for chain in chains]
            })
            chains_path = os.path.join(self.output_dir, 'supply_chains.csv')
            chains_df.to_csv(chains_path, index=False)
        print(f"Referral analysis exported to: {self.output_dir}")
        return self.output_dir

    def visualize_graph_sample(self, graph: nx.Graph, node_types: Dict,
                               max_nodes: int = 100):
        """Visualize sample of the graph"""
        print("Generating graph visualization...")
        # Sample nodes if too large
        if graph.number_of_nodes() > max_nodes:
            # Get high-degree nodes
            degrees = dict(graph.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            subgraph = graph.subgraph(top_nodes)
        else:
            subgraph = graph
        # Color by node type
        color_map = {
            'provider': '#FF6B6B',
            'patient': '#4ECDC4',
            'procedure': '#95E1D3'
        }
        node_colors = []
        for node in subgraph.nodes():
            if node.startswith('P_') and not node.startswith('PT_'):
                node_colors.append(color_map['provider'])
            elif node.startswith('PT_'):
                node_colors.append(color_map['patient'])
            else:
                node_colors.append(color_map['procedure'])
        # Layout
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50, seed=42)
        # Plot
        fig, ax = plt.subplots(figsize=(16, 12))
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors,
                               node_size=100, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, ax=ax)
        # Legend
        legend_elements = [
            Patch(facecolor=color_map['provider'], label='Provider'),
            Patch(facecolor=color_map['patient'], label='Patient'),
            Patch(facecolor=color_map['procedure'], label='Procedure')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        ax.set_title('Healthcare Network Graph (Sample)', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'graph_visualization.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to: {filepath}")
        plt.show()

    def visualize_centrality_distribution(self, centrality_df: pd.DataFrame):
        """Visualize centrality metrics distribution"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        metrics = ['degree_centrality', 'betweenness_centrality', 'closeness_centrality',
                   'eigenvector_centrality', 'pagerank', 'clustering_coefficient']
        for idx, metric in enumerate(metrics):
            for node_type in ['provider', 'patient', 'procedure']:
                data = centrality_df[centrality_df['node_type'] == node_type][metric]
                if len(data) > 0:
                    axes[idx].hist(data, bins=30, alpha=0.6, label=node_type)
            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'centrality_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Centrality distribution plot saved to: {filepath}")
        plt.show()

    def visualize_embeddings_2d(self, embed_df: pd.DataFrame):
        """Visualize embeddings in 2D using t-SNE"""
        print("Generating embedding visualization (t-SNE)...")
        # Get embedding columns (numerical)
        embed_cols = [col for col in embed_df.columns if isinstance(col, int) or str(col).isdigit()]
        X = embed_df[embed_cols].values
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X)
        # Plot
        fig, ax = plt.subplots(figsize=(12, 9))
        for node_type, color in [('provider', '#FF6B6B'),
                                 ('patient', '#4ECDC4'),
                                 ('procedure', '#95E1D3')]:
            mask = embed_df['node_type'] == node_type
            if mask.sum() > 0:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                           c=color, label=node_type, alpha=0.6, s=50)
        ax.set_title('Node Embeddings (t-SNE 2D Projection)', fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'embeddings_tsne.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Embedding visualization saved to: {filepath}")
        plt.show()


class FraudSignalDetector:
    """Module for detecting fraud signals in graph"""
    def __init__(self, graph: nx.Graph, centrality_df: pd.DataFrame):
        self.graph = graph
        self.centrality_df = centrality_df

    def detect_unusual_connectivity(self, threshold_percentile: float = 95) -> pd.DataFrame:
        """
        Detect nodes with unusual connectivity patterns
        High degree but low clustering = hub-and-spoke fraud pattern
        """
        print("Detecting unusual connectivity patterns...")
        suspicious = []
        # Focus on providers
        provider_df = self.centrality_df[self.centrality_df['node_type'] == 'provider'].copy()
        if len(provider_df) == 0:
            return pd.DataFrame()
        # High degree but low clustering (hub-and-spoke pattern)
        high_degree_threshold = provider_df['degree'].quantile(threshold_percentile / 100)
        low_clustering_threshold = provider_df['clustering_coefficient'].quantile(0.2)
        hub_spoke = provider_df[
            (provider_df['degree'] > high_degree_threshold) &
            (provider_df['clustering_coefficient'] < low_clustering_threshold)
        ].copy()
        hub_spoke['fraud_signal'] = 'hub_spoke_pattern'
        suspicious.append(hub_spoke)
        # High betweenness (broker position)
        high_betweenness_threshold = provider_df['betweenness_centrality'].quantile(threshold_percentile / 100)
        brokers = provider_df[
            provider_df['betweenness_centrality'] > high_betweenness_threshold
        ].copy()
        brokers['fraud_signal'] = 'broker_position'
        suspicious.append(brokers)
        if suspicious:
            suspicious_df = pd.concat(suspicious, ignore_index=True)
            print(f"Found {len(suspicious_df)} nodes with unusual connectivity")
            return suspicious_df
        else:
            return pd.DataFrame()

    def detect_isolated_cliques(self, min_size: int = 4) -> List[Set[str]]:
        """
        Detect tightly connected groups (cliques) that are isolated from rest of network
        Potential collusion rings
        """
        print(f"Detecting isolated cliques (min size: {min_size})...")
        cliques = list(nx.find_cliques(self.graph))
        large_cliques = [set(c) for c in cliques if len(c) >= min_size]
        # Check isolation: low connectivity to rest of network
        isolated_cliques = []
        for clique in large_cliques:
            clique_nodes = list(clique)
            # Count edges going outside clique
            external_edges = 0
            for node in clique_nodes:
                neighbors = set(self.graph.neighbors(node))
                external_neighbors = neighbors - clique
                external_edges += len(external_neighbors)
            # Isolation ratio: internal edges / (internal + external)
            internal_edges = len(list(nx.subgraph(self.graph, clique_nodes).edges()))
            isolation_ratio = internal_edges / (internal_edges + external_edges + 1)
            if isolation_ratio > 0.7:  # Highly isolated
                isolated_cliques.append(clique)
        print(f"Found {len(isolated_cliques)} isolated cliques")
        return isolated_cliques

    def detect_star_topology(self, min_spokes: int = 10) -> pd.DataFrame:
        """
        Detect star topologies (one provider connected to many patients with no cross-connections)
        Potential patient steering or phantom billing
        """
        print(f"Detecting star topologies (min spokes: {min_spokes})...")
        stars = []
        for node in self.graph.nodes():
            if not (node.startswith('P_') and not node.startswith('PT_')):
                continue  # Only check providers
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < min_spokes:
                continue
            # Check if neighbors are mostly patients
            patient_neighbors = [n for n in neighbors if n.startswith('PT_')]
            if len(patient_neighbors) < min_spokes:
                continue
            # Check cross-connectivity among patients
            neighbor_subgraph = self.graph.subgraph(patient_neighbors)
            cross_connections = neighbor_subgraph.number_of_edges()
            # Low cross-connectivity = star pattern
            if cross_connections < len(patient_neighbors) * 0.1:
                stars.append({
                    'provider': node,
                    'num_patients': len(patient_neighbors),
                    'cross_connections': cross_connections,
                    'star_score': len(patient_neighbors) / (cross_connections + 1)
                })
        if stars:
            stars_df = pd.DataFrame(stars).sort_values('star_score', ascending=False)
            print(f"Found {len(stars_df)} potential star topologies")
            return stars_df
        else:
            return pd.DataFrame()

    def detect_unusual_procedure_combinations(self, min_support: int = 3) -> pd.DataFrame:
        """
        Detect unusual procedure combinations performed together
        Based on co-occurrence patterns in provider-procedure subgraph
        """
        print("Detecting unusual procedure combinations...")
        unusual_combos = []
        # Get provider-procedure edges
        provider_procedures = defaultdict(set)
        for node in self.graph.nodes():
            if node.startswith('P_') and not node.startswith('PT_'):
                neighbors = [n for n in self.graph.neighbors(node) if n.startswith('PR_')]
                provider_procedures[node] = set(neighbors)
        # Count procedure pair occurrences
        from itertools import combinations
        pair_counts = Counter()
        for provider, procedures in provider_procedures.items():
            if len(procedures) > 1:
                for pair in combinations(sorted(procedures), 2):
                    pair_counts[pair] += 1
        # Rare but existing combinations
        rare_pairs = [(pair, count) for pair, count in pair_counts.items()
                      if min_support <= count <= max(1, int(len(provider_procedures) * 0.05))]
        for pair, count in rare_pairs:
            # Find providers doing this combination
            providers_with_pair = [p for p, procs in provider_procedures.items()
                                   if pair[0] in procs and pair[1] in procs]
            unusual_combos.append({
                'procedure_1': pair[0],
                'procedure_2': pair[1],
                'co_occurrence_count': count,
                'num_providers': len(providers_with_pair),
                'providers': ', '.join(providers_with_pair[:5])  # First 5
            })
        if unusual_combos:
            combos_df = pd.DataFrame(unusual_combos).sort_values('co_occurrence_count')
            print(f"Found {len(combos_df)} unusual procedure combinations")
            return combos_df
        else:
            return pd.DataFrame()


def main(db_path: str, max_nodes_for_embedding: int = 5000):
    """
    Main execution pipeline for graph-based analytics
    Parameters:
    - db_path: Path to SQLite database
    - max_nodes_for_embedding: Maximum nodes for Node2Vec (computational constraint)
    """
    print("="*100)
    print("HEALTHCARE PAYMENT INTEGRITY - GRAPH-BASED NETWORK ANALYSIS")
    print("="*100)

    # 1. Load Data
    print("[1/9] Loading Data...")
    loader = DataLoader(db_path)
    df = loader.load_for_graph_analysis()

    # 2. Build Graphs
    print("[2/9] Building Network Graphs...")
    print("="*100)
    graph_builder = HealthcareGraph()
    # Tripartite graph
    tripartite_graph = graph_builder.build_tripartite_graph(df)
    # Referral network
    referral_graph = graph_builder.build_provider_referral_network(df)

    # 3. Centrality Analysis
    print("[3/9] Computing Centrality Metrics...")
    print("="*100)
    centrality_analyzer = CentralityAnalyzer(tripartite_graph)
    centrality_df = centrality_analyzer.compute_all_centralities()
    # Identify hubs
    hubs = centrality_analyzer.identify_hubs(top_n=20)
    print("Top 10 nodes by PageRank:")
    print(hubs['pagerank'].head(10).to_string(index=False))
    # Detect anomalous centrality
    centrality_anomalies = centrality_analyzer.detect_anomalous_centrality(threshold_percentile=95)
    if len(centrality_anomalies) > 0:
        print(f"Detected {len(centrality_anomalies)} nodes with anomalous centrality")

    # 4. Community Detection
    print("[4/9] Detecting Communities...")
    print("="*100)
    community_detector = CommunityDetector(tripartite_graph)
    communities = community_detector.detect_communities(method='greedy_modularity')
    community_stats = community_detector.analyze_communities()

    print("\nTop 5 Largest Communities:")
    if community_stats is not None and len(community_stats) > 0:
        print(community_stats.head(5).to_string(index=False))
    else:
        print("No communities detected or graph was empty after filtering.")
    # 5. Node Embeddings (Node2Vec)
    print("[5/9] Training Node2Vec Embeddings...")
    print("="*100)
    # Sample graph if too large
    if tripartite_graph.number_of_nodes() > max_nodes_for_embedding:
        print(f"Graph has {tripartite_graph.number_of_nodes()} nodes, sampling {max_nodes_for_embedding}...")
        # Keep high-degree nodes
        degrees = dict(tripartite_graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes_for_embedding]
        embedding_graph = tripartite_graph.subgraph(top_nodes)
    else:
        embedding_graph = tripartite_graph
    node2vec = Node2VecEmbedding(dimensions=64, walk_length=30, num_walks=200)
    embeddings = node2vec.fit(embedding_graph, p=1, q=1)
    embed_df = node2vec.get_embedding_dataframe()
    # Cluster embeddings
    clusters, silhouette = node2vec.cluster_embeddings(n_clusters=8)
    embed_df['cluster'] = clusters

    # 6. Referral Pattern Analysis
    print("[6/9] Analyzing Referral Patterns...")
    print("="*100)
    referral_analyzer = ReferralPatternAnalyzer(referral_graph)
    # Detect circular referrals
    referral_loops = referral_analyzer.detect_referral_loops()
    if referral_loops:
        print(f"Top 5 Referral Loops:")
        for i, loop in enumerate(referral_loops[:5], 1):
            print(f" {i}. {' -> '.join(loop)} -> {loop[0]}")
    # Identify referral hubs
    referral_hubs = referral_analyzer.identify_referral_hubs(top_n=20)
    print(f"Top 10 Referral Hubs:")
    print(referral_hubs.head(10).to_string(index=False))
    # Analyze supply chains
    supply_chains = referral_analyzer.analyze_supply_chains(min_chain_length=3)
    if supply_chains:
        print(f"Longest Supply Chains (Top 5):")
        for i, chain in enumerate(supply_chains[:5], 1):
            print(f" {i}. Length {len(chain)}: {' -> '.join(chain)}")

    # 7. Fraud Signal Detection
    print("[7/9] Detecting Fraud Signals...")
    print("="*100)
    fraud_detector = FraudSignalDetector(tripartite_graph, centrality_df)
    # Unusual connectivity
    unusual_connectivity = fraud_detector.detect_unusual_connectivity(threshold_percentile=95)
    if len(unusual_connectivity) > 0:
        print(f"Found {len(unusual_connectivity)} nodes with unusual connectivity patterns")
    # Isolated cliques
    isolated_cliques = fraud_detector.detect_isolated_cliques(min_size=4)
    if isolated_cliques:
        print(f"Found {len(isolated_cliques)} isolated cliques (potential collusion)")
    # Star topologies
    star_topologies = fraud_detector.detect_star_topology(min_spokes=10)
    if len(star_topologies) > 0:
        print(f"Found {len(star_topologies)} potential star topology patterns")
        print("Top 5 Star Patterns:")
        print(star_topologies.head(5).to_string(index=False))
    # Unusual procedure combinations
    unusual_procedures = fraud_detector.detect_unusual_procedure_combinations(min_support=3)
    if len(unusual_procedures) > 0:
        print(f"Found {len(unusual_procedures)} unusual procedure combinations")

    # 8. Export Results
    print("[8/9] Exporting Results...")
    print("="*100)
    exporter = ResultsExporter()
    exporter.export_centrality_metrics(centrality_df)
    exporter.export_embeddings(embed_df)
    if len(centrality_anomalies) > 0:
        exporter.export_anomalies(centrality_anomalies)
    exporter.export_referral_analysis(referral_hubs, referral_loops, supply_chains)
    # Export fraud signals
    if len(unusual_connectivity) > 0:
        unusual_connectivity.to_csv(
            os.path.join(exporter.output_dir, 'unusual_connectivity.csv'),
            index=False
        )
    if len(star_topologies) > 0:
        star_topologies.to_csv(
            os.path.join(exporter.output_dir, 'star_topologies.csv'),
            index=False
        )
    if len(unusual_procedures) > 0:
        unusual_procedures.to_csv(
            os.path.join(exporter.output_dir, 'unusual_procedures.csv'),
            index=False
        )

    # 9. Visualizations
    print("[9/9] Generating Visualizations...")
    print("="*100)
    exporter.visualize_graph_sample(tripartite_graph, graph_builder.node_types, max_nodes=100)
    exporter.visualize_centrality_distribution(centrality_df)
    exporter.visualize_embeddings_2d(embed_df)

    print("" + "="*100)
    print("GRAPH ANALYSIS COMPLETE")
    print("="*100)
    print(f"Summary:")
    print(f" - Graph Nodes: {tripartite_graph.number_of_nodes()}")
    print(f" - Graph Edges: {tripartite_graph.number_of_edges()}")
    print(f" - Communities Detected: {len(set(communities.values()))}")
    print(f" - Referral Loops: {len(referral_loops)}")
    print(f" - Supply Chains (length >= 3): {len(supply_chains)}")
    print(f" - Star Topologies: {len(star_topologies)}")
    print(f" - Centrality Anomalies: {len(centrality_anomalies)}")
    print(f"All outputs saved to './outputs/graph_analysis' directory")
    print("="*100)

    return {
        'tripartite_graph': tripartite_graph,
        'referral_graph': referral_graph,
        'centrality_df': centrality_df,
        'embeddings': embed_df,
        'communities': communities,
        'community_stats': community_stats,
        'referral_loops': referral_loops,
        'supply_chains': supply_chains,
        'star_topologies': star_topologies,
        'fraud_signals': {
            'unusual_connectivity': unusual_connectivity,
            'isolated_cliques': isolated_cliques,
            'unusual_procedures': unusual_procedures
        }
    }


if __name__ == "__main__":
    # Use DB file name exactly as requested
    db_path = "claims_database.db"
    # Run graph analysis
    results = main(db_path, max_nodes_for_embedding=5000)
    # Access results
    # results['tripartite_graph'] - Main network graph
    # results['centrality_df']    - Centrality metrics for all nodes
    # results['embeddings']       - Node2Vec embeddings
    # results['referral_loops']   - Circular referral patterns
    # results['supply_chains']    - Long referral chains (DME indicators)
    # results['star_topologies']  - Hub-and-spoke fraud patterns
    # results['fraud_signals']    - Dictionary of various fraud indicators
