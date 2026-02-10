import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GIOTTO-TDA IMPORTS
# ============================================================================
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
    print("Warning: Giotto-TDA not available. Some features will be disabled.")

# ============================================================================
# SPECTRAL FEATURE EXTRACTOR
# ============================================================================
class SpectralFeatureExtractor:
    """Extract spectral features from graph Laplacians"""
    
    def __init__(self, n_eigenvalues=5):
        self.n_eigenvalues = n_eigenvalues
    
    def compute_laplacian_0(self, adjacency):
        """Compute weighted 0-Laplacian (graph Laplacian)"""
        # Degree matrix
        degrees = np.sum(adjacency, axis=1)
        D = np.diag(degrees)
        # Laplacian L = D - A
        L = D - adjacency
        return L
    
    def compute_laplacian_1(self, adjacency):
        """Compute weighted 1-Laplacian (edge Laplacian)"""
        G = nx.from_numpy_array(adjacency)
        if G.number_of_edges() == 0:
            return np.zeros((1, 1))
        
        # Build incidence matrix
        edges = list(G.edges())
        nodes = list(G.nodes())
        n_edges = len(edges)
        n_nodes = len(nodes)
        
        if n_edges == 0:
            return np.zeros((1, 1))
        
        # Incidence matrix B (oriented)
        B = np.zeros((n_nodes, n_edges))
        for idx, (u, v) in enumerate(edges):
            B[u, idx] = 1
            B[v, idx] = -1
        
        # Edge weight matrix
        W = np.diag([adjacency[u, v] for u, v in edges])
        
        # 1-Laplacian: L1 = B^T * B (or weighted version)
        L1 = B.T @ B
        
        return L1
    
    def get_eigenvalues(self, L, k=None):
        """Compute smallest k eigenvalues of Laplacian"""
        if k is None:
            k = min(self.n_eigenvalues, L.shape[0] - 1)
        
        k = max(1, min(k, L.shape[0] - 2))
        
        try:
            # For small matrices, use full eigendecomposition
            if L.shape[0] <= 20:
                eigenvalues = np.linalg.eigvalsh(L)
                eigenvalues = np.sort(eigenvalues)[:k]
            else:
                # For larger matrices, use sparse solver
                L_sparse = csr_matrix(L)
                eigenvalues, _ = eigsh(L_sparse, k=k, which='SM', tol=1e-3)
                eigenvalues = np.sort(eigenvalues)
            
            # Pad if necessary
            if len(eigenvalues) < self.n_eigenvalues:
                eigenvalues = np.pad(eigenvalues, 
                                   (0, self.n_eigenvalues - len(eigenvalues)), 
                                   'constant', constant_values=0)
            
            return eigenvalues[:self.n_eigenvalues]
        
        except:
            return np.zeros(self.n_eigenvalues)
    
    def extract_features(self, adjacency):
        """Extract all spectral features"""
        features = {}
        
        # Remove self-loops for proper Laplacian computation
        adj_clean = adjacency.copy()
        np.fill_diagonal(adj_clean, 0)
        
        # 0-Laplacian eigenvalues
        L0 = self.compute_laplacian_0(adj_clean)
        eig_0 = self.get_eigenvalues(L0)
        features['laplacian_0_eigenvalues'] = eig_0
        
        # 1-Laplacian eigenvalues
        L1 = self.compute_laplacian_1(adj_clean)
        if L1.shape[0] > 1:
            eig_1 = self.get_eigenvalues(L1)
            features['laplacian_1_eigenvalues'] = eig_1
        else:
            features['laplacian_1_eigenvalues'] = np.zeros(self.n_eigenvalues)
        
        # Algebraic connectivity (Fiedler value - 2nd smallest eigenvalue of L0)
        features['algebraic_connectivity'] = eig_0[1] if len(eig_0) > 1 else 0
        
        # Spectral gap (difference between 2nd and 1st non-zero eigenvalues)
        features['spectral_gap_0'] = eig_0[2] - eig_0[1] if len(eig_0) > 2 else 0
        features['spectral_gap_1'] = (features['laplacian_1_eigenvalues'][1] - 
                                      features['laplacian_1_eigenvalues'][0])
        
        # Spectral radius approximation
        features['spectral_radius'] = np.max(eig_0) if len(eig_0) > 0 else 0
        
        return features

# ============================================================================
# CARLSSON COORDINATES EXTRACTOR
# ============================================================================
class CarlssonCoordinatesExtractor:
    """Extract Carlsson coordinates (density-based topological features)"""
    
    def __init__(self, k_neighbors=3):
        self.k_neighbors = k_neighbors
    
    def compute_density(self, adjacency, k):
        """Compute k-nearest neighbor density for each node"""
        n = adjacency.shape[0]
        densities = np.zeros(n)
        
        for i in range(n):
            # Get distances from node i
            distances = adjacency[i, :].copy()
            distances[i] = np.inf  # Exclude self
            
            # Get k nearest neighbors (smallest non-zero distances)
            nonzero_dist = distances[distances > 0]
            if len(nonzero_dist) >= k:
                k_nearest = np.partition(nonzero_dist, k-1)[:k]
                densities[i] = np.mean(k_nearest)
            elif len(nonzero_dist) > 0:
                densities[i] = np.mean(nonzero_dist)
        
        return densities
    
    def extract_features(self, adjacency):
        """Extract Carlsson coordinate features"""
        features = {}
        
        # Compute densities
        densities = self.compute_density(adjacency, self.k_neighbors)
        
        # Statistical moments of density
        features['density_mean'] = np.mean(densities)
        features['density_std'] = np.std(densities)
        features['density_max'] = np.max(densities)
        features['density_min'] = np.min(densities) if np.min(densities) > 0 else 0
        
        # Eccentricity-based features
        G = nx.from_numpy_array(adjacency)
        if G.number_of_nodes() > 0 and nx.is_connected(G):
            eccentricities = nx.eccentricity(G)
            ecc_values = list(eccentricities.values())
            features['eccentricity_mean'] = np.mean(ecc_values)
            features['eccentricity_std'] = np.std(ecc_values)
        else:
            features['eccentricity_mean'] = 0
            features['eccentricity_std'] = 0
        
        return features

# ============================================================================
# GRAPH STATISTICS EXTRACTOR
# ============================================================================
class GraphStatisticsExtractor:
    """Extract classical graph theory features"""
    
    def extract_features(self, adjacency):
        """Extract comprehensive graph statistics"""
        features = {}
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency)
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Basic counts
        features['num_nodes'] = n_nodes
        features['num_edges'] = n_edges
        features['density'] = nx.density(G) if n_nodes > 1 else 0
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        if degrees:
            features['mean_degree'] = np.mean(degrees)
            features['std_degree'] = np.std(degrees)
            features['max_degree'] = np.max(degrees)
            features['min_degree'] = np.min(degrees)
        else:
            features['mean_degree'] = 0
            features['std_degree'] = 0
            features['max_degree'] = 0
            features['min_degree'] = 0
        
        # Clustering and transitivity
        features['avg_clustering'] = nx.average_clustering(G)
        features['transitivity'] = nx.transitivity(G)
        
        # Connected components
        features['num_components'] = nx.number_connected_components(G)
        
        # Path-based features (only for connected graphs)
        if nx.is_connected(G) and n_nodes > 1:
            features['diameter'] = nx.diameter(G)
            features['avg_shortest_path'] = nx.average_shortest_path_length(G)
            features['radius'] = nx.radius(G)
        else:
            # For disconnected graphs, use largest component
            if n_nodes > 1:
                largest_cc = max(nx.connected_components(G), key=len)
                G_largest = G.subgraph(largest_cc)
                if G_largest.number_of_nodes() > 1:
                    features['diameter'] = nx.diameter(G_largest)
                    features['avg_shortest_path'] = nx.average_shortest_path_length(G_largest)
                    features['radius'] = nx.radius(G_largest)
                else:
                    features['diameter'] = 0
                    features['avg_shortest_path'] = 0
                    features['radius'] = 0
            else:
                features['diameter'] = 0
                features['avg_shortest_path'] = 0
                features['radius'] = 0
        
        # Cycle detection
        try:
            cycles = nx.cycle_basis(G)
            features['num_cycles'] = len(cycles)
            if cycles:
                cycle_lengths = [len(c) for c in cycles]
                features['mean_cycle_length'] = np.mean(cycle_lengths)
                features['max_cycle_length'] = np.max(cycle_lengths)
            else:
                features['mean_cycle_length'] = 0
                features['max_cycle_length'] = 0
        except:
            features['num_cycles'] = 0
            features['mean_cycle_length'] = 0
            features['max_cycle_length'] = 0
        
        # Assortativity
        try:
            features['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            features['degree_assortativity'] = 0
        
        return features

# ============================================================================
# PERSISTENT HOMOLOGY FEATURE EXTRACTOR
# ============================================================================
class PersistentHomologyExtractor:
    """Extract topological features using giotto-tda"""
    
    def __init__(self, homology_dimensions=(0, 1), n_bins=30, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        
        if GTDA_AVAILABLE:
            self._initialize_gtda()
        else:
            self.persistence = None
            self.entropy = None
            self.amplitudes = {}
    
    def _initialize_gtda(self):
        """Initialize giotto-tda components"""
        self.persistence = VietorisRipsPersistence(
            metric='precomputed',
            homology_dimensions=self.homology_dimensions,
            collapse_edges=True,
            n_jobs=self.n_jobs
        )
        
        self.entropy = PersistenceEntropy(n_jobs=self.n_jobs)
        
        # Betti curve extractor
        self.betti_curve = BettiCurve(n_bins=self.n_bins, n_jobs=self.n_jobs)
        
        # Extended set of amplitude metrics
        self.amplitudes = {
            'bottleneck': Amplitude(metric='bottleneck', n_jobs=self.n_jobs),
            'wasserstein': Amplitude(metric='wasserstein', metric_params={'p': 1}, n_jobs=self.n_jobs),
            'landscape': Amplitude(
                metric='landscape',
                metric_params={'n_layers': 1, 'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
            'betti': Amplitude(
                metric='betti',
                metric_params={'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
            'heat': Amplitude(
                metric='heat',
                metric_params={'sigma': 0.1, 'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
        }
    
    def adjacency_to_distance(self, adjacency_matrix):
        """Convert adjacency matrix to distance matrix"""
        distance = adjacency_matrix.copy()
        np.fill_diagonal(distance, 0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = 1.0 / (distance + 1e-10)
            distance = np.clip(distance, 0, 1e6)
        
        distance = np.maximum(distance, distance.T)
        np.fill_diagonal(distance, 0)
        
        return distance
    
    def extract_features(self, adjacency_matrices):
        """Extract persistent homology features"""
        if not GTDA_AVAILABLE:
            return self._get_empty_features(adjacency_matrices)
        
        features = {}
        
        distance_matrices = np.array([
            self.adjacency_to_distance(adj) for adj in adjacency_matrices
        ])
        
        persistence_diagrams = self.persistence.fit_transform(distance_matrices)
        features['persistence_diagrams'] = persistence_diagrams
        
        # Persistence entropy
        entropy_features = self.entropy.fit_transform(persistence_diagrams)
        entropy_features[~np.isfinite(entropy_features)] = 0
        features['persistent_entropy'] = entropy_features
        
        # Betti curves
        try:
            betti_features = self.betti_curve.fit_transform(persistence_diagrams)
            betti_features[~np.isfinite(betti_features)] = 0
            features['betti_curves'] = betti_features
        except:
            n_samples = len(adjacency_matrices)
            n_dims = len(self.homology_dimensions)
            features['betti_curves'] = np.zeros((n_samples, self.n_bins * n_dims))
        
        # Amplitudes
        amplitude_features = {}
        for metric_name, amplitude_calc in self.amplitudes.items():
            try:
                amp = amplitude_calc.fit_transform(persistence_diagrams)
                amp[~np.isfinite(amp)] = 0
                amplitude_features[metric_name] = amp
            except:
                n_samples = len(adjacency_matrices)
                n_dims = len(self.homology_dimensions)
                amplitude_features[metric_name] = np.zeros((n_samples, n_dims))
        features['amplitudes'] = amplitude_features
        
        return features
    
    def _get_empty_features(self, adjacency_matrices):
        """Return empty features when giotto-tda is not available"""
        n_samples = len(adjacency_matrices)
        n_dims = len(self.homology_dimensions)
        
        return {
            'persistence_diagrams': [np.array([])] * n_samples,
            'persistent_entropy': np.zeros((n_samples, n_dims)),
            'betti_curves': np.zeros((n_samples, self.n_bins * n_dims)),
            'amplitudes': {
                'bottleneck': np.zeros((n_samples, n_dims)),
                'wasserstein': np.zeros((n_samples, n_dims)),
                'landscape': np.zeros((n_samples, n_dims)),
                'betti': np.zeros((n_samples, n_dims)),
                'heat': np.zeros((n_samples, n_dims)),
            }
        }

# ============================================================================
# MAIN FEATURE GENERATOR
# ============================================================================
class DiscriminativeFeatureGenerator:
    """
    Comprehensive feature generator combining:
    - Persistent homology (TDA)
    - Spectral features (Laplacians)
    - Carlsson coordinates
    - Graph statistics
    """
    
    def __init__(self, homology_dimensions=(0, 1), n_bins=30, 
                 n_eigenvalues=5, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.n_eigenvalues = n_eigenvalues
        
        # Initialize all extractors
        self.scaler = StandardScaler()
        self.ph_extractor = PersistentHomologyExtractor(
            homology_dimensions=homology_dimensions,
            n_bins=n_bins,
            n_jobs=n_jobs
        )
        self.spectral_extractor = SpectralFeatureExtractor(n_eigenvalues=n_eigenvalues)
        self.carlsson_extractor = CarlssonCoordinatesExtractor(k_neighbors=3)
        self.graph_stats_extractor = GraphStatisticsExtractor()
        
        self.feature_dim = None
        self.feature_names = []
    
    def generate_features(self, adjacency_matrices, labels):
        """Generate all features"""
        print("=" * 60)
        print("GENERATING COMPREHENSIVE TOPOLOGICAL FEATURES")
        print("=" * 60)
        
        X = []
        y = []
        
        # Determine feature dimension on first graph
        if self.feature_dim is None:
            print("\nDetermining feature dimension...")
            test_feature = self._graph_to_features(adjacency_matrices[0])
            self.feature_dim = len(test_feature)
            self.feature_names = self._get_feature_names()
            print(f"Total feature dimension: {self.feature_dim}")
            self._print_feature_breakdown()
        
        # Extract features for all graphs
        for i, adjacency in enumerate(adjacency_matrices):
            if i % 500 == 0:
                print(f"Processing graph {i+1}/{len(adjacency_matrices)}...")
            
            feature_vector = self._graph_to_features(adjacency)
            
            # Ensure consistent length
            if len(feature_vector) != self.feature_dim:
                if len(feature_vector) > self.feature_dim:
                    feature_vector = feature_vector[:self.feature_dim]
                else:
                    feature_vector = np.pad(
                        feature_vector, 
                        (0, self.feature_dim - len(feature_vector)), 
                        'constant'
                    )
            
            X.append(feature_vector)
            y.append(labels[i])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{'='*60}")
        print(f"Feature extraction completed!")
        print(f"Total samples: {len(adjacency_matrices)}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"{'='*60}\n")
        
        return X, y, {'feature_names': self.feature_names}
    
    def _graph_to_features(self, adjacency):
        """Extract all features from a single graph"""
        features = []
        
        # 1. Persistent homology features
        ph_features = self.ph_extractor.extract_features([adjacency])
        
        # Persistent entropy
        if 'persistent_entropy' in ph_features:
            entropy_vals = ph_features['persistent_entropy'][0]
            if hasattr(entropy_vals, '__iter__'):
                features.extend(entropy_vals)
            else:
                features.append(entropy_vals)
        
        # Betti curves - need to flatten
        if 'betti_curves' in ph_features:
            betti_vals = ph_features['betti_curves'][0]
            if isinstance(betti_vals, np.ndarray):
                features.extend(betti_vals.flatten())
            elif hasattr(betti_vals, '__iter__'):
                features.extend(betti_vals)
            else:
                features.append(betti_vals)
        
        # Amplitudes
        if 'amplitudes' in ph_features:
            for metric in ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']:
                if metric in ph_features['amplitudes']:
                    amp_vals = ph_features['amplitudes'][metric][0]
                    if hasattr(amp_vals, '__iter__'):
                        features.extend(amp_vals)
                    else:
                        features.append(amp_vals)
        
        # 2. Spectral features
        spectral_features = self.spectral_extractor.extract_features(adjacency)
        features.extend(spectral_features['laplacian_0_eigenvalues'])
        features.extend(spectral_features['laplacian_1_eigenvalues'])
        features.append(spectral_features['algebraic_connectivity'])
        features.append(spectral_features['spectral_gap_0'])
        features.append(spectral_features['spectral_gap_1'])
        features.append(spectral_features['spectral_radius'])
        
        # 3. Carlsson coordinates
        carlsson_features = self.carlsson_extractor.extract_features(adjacency)
        for key in ['density_mean', 'density_std', 'density_max', 'density_min',
                    'eccentricity_mean', 'eccentricity_std']:
            features.append(carlsson_features[key])
        
        # 4. Graph statistics
        graph_features = self.graph_stats_extractor.extract_features(adjacency)
        for key in ['num_nodes', 'num_edges', 'density', 'mean_degree', 'std_degree',
                    'max_degree', 'min_degree', 'avg_clustering', 'transitivity',
                    'num_components', 'diameter', 'avg_shortest_path', 'radius',
                    'num_cycles', 'mean_cycle_length', 'max_cycle_length',
                    'degree_assortativity']:
            features.append(graph_features[key])
        
        # Clean features: replace inf/nan with 0
        features = np.array(features, dtype=np.float64)
        features[~np.isfinite(features)] = 0
        
        return features
    
    def _get_feature_names(self):
        """Get descriptive feature names"""
        names = []
        
        # Persistent homology - Entropy
        for dim in self.homology_dimensions:
            names.append(f'persistent_entropy_H{dim}')
        
        # Betti curves
        for dim in self.homology_dimensions:
            for bin_idx in range(self.n_bins):
                names.append(f'betti_curve_H{dim}_bin{bin_idx}')
        
        # Amplitudes
        for metric in ['bottleneck', 'wasserstein', 'landscape', 'betti', 'heat']:
            for dim in self.homology_dimensions:
                names.append(f'amplitude_{metric}_H{dim}')
        
        # Spectral features
        for i in range(self.n_eigenvalues):
            names.append(f'L0_eigenvalue_{i}')
        for i in range(self.n_eigenvalues):
            names.append(f'L1_eigenvalue_{i}')
        names.extend(['algebraic_connectivity', 'spectral_gap_0', 
                     'spectral_gap_1', 'spectral_radius'])
        
        # Carlsson coordinates
        names.extend(['density_mean', 'density_std', 'density_max', 'density_min',
                     'eccentricity_mean', 'eccentricity_std'])
        
        # Graph statistics
        names.extend(['num_nodes', 'num_edges', 'graph_density', 'mean_degree', 
                     'std_degree', 'max_degree', 'min_degree', 'avg_clustering',
                     'transitivity', 'num_components', 'diameter', 'avg_shortest_path',
                     'radius', 'num_cycles', 'mean_cycle_length', 'max_cycle_length',
                     'degree_assortativity'])
        
        return names
    
    def _print_feature_breakdown(self):
        """Print detailed feature breakdown"""
        n_dims = len(self.homology_dimensions)
        
        counts = {
            'Persistent Entropy': n_dims,
            'Betti Curves': self.n_bins * n_dims,
            'Amplitude Metrics': 5 * n_dims,  # bottleneck, wasserstein, landscape, betti, heat
            'L0 Eigenvalues': self.n_eigenvalues,
            'L1 Eigenvalues': self.n_eigenvalues,
            'Spectral Stats': 4,  # connectivity, gaps, radius
            'Carlsson Coordinates': 6,
            'Graph Statistics': 17
        }
        
        print("\nFEATURE BREAKDOWN:")
        print("-" * 60)
        for category, count in counts.items():
            print(f"  {category:30s}: {count:3d} features")
        print("-" * 60)
        print(f"  {'TOTAL':30s}: {sum(counts.values()):3d} features")
        print("-" * 60)
    
    def fit_transform(self, adjacency_matrices, labels):
        """Generate features and fit scaler"""
        X, y, feature_info = self.generate_features(adjacency_matrices, labels)
        if len(X) == 0:
            return None, None, None
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, feature_info
    
    def transform(self, adjacency_matrices):
        """Transform new data"""
        X = []
        for adjacency in adjacency_matrices:
            feature_vector = self._graph_to_features(adjacency)
            if len(feature_vector) != self.feature_dim:
                if len(feature_vector) > self.feature_dim:
                    feature_vector = feature_vector[:self.feature_dim]
                else:
                    feature_vector = np.pad(
                        feature_vector, 
                        (0, self.feature_dim - len(feature_vector)), 
                        'constant'
                    )
            X.append(feature_vector)
        
        X = np.array(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

# ============================================================================
# FEATURE COUNT CALCULATOR
# ============================================================================
def calculate_feature_counts(homology_dimensions=(0, 1), n_bins=30, n_eigenvalues=5):
    """Calculate total number of features"""
    n_dims = len(homology_dimensions)
    
    counts = {
        'Persistent_Entropy': n_dims,
        'Betti_Curves': n_bins * n_dims,
        'Amplitude_Metrics': 5 * n_dims,  # bottleneck, wasserstein, landscape, betti, heat
        'L0_Eigenvalues': n_eigenvalues,
        'L1_Eigenvalues': n_eigenvalues,
        'Spectral_Statistics': 4,
        'Carlsson_Coordinates': 6,
        'Graph_Statistics': 17
    }
    
    total = sum(counts.values())
    
    print("\nCOMPREHENSIVE FEATURE BREAKDOWN:")
    print("=" * 60)
    for category, count in counts.items():
        print(f"{category:30s}: {count:3d} features")
    print("=" * 60)
    print(f"{'TOTAL':30s}: {total:3d} features")
    print("=" * 60)
    
    return total, counts

if __name__ == "__main__":
    # Test feature counting
    print("\nTesting feature configuration...")
    total, breakdown = calculate_feature_counts(
        homology_dimensions=(0, 1),
        n_bins=30,
        n_eigenvalues=5
    )
    print(f"\nConfiguration produces {total} features")