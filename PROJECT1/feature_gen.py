import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GIOTTO-TDA IMPORTS
# ============================================================================
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceEntropy, Amplitude, BettiCurve
    GTDATDA_AVAILABLE = True
except ImportError:
    GTDATDA_AVAILABLE = False
    print("Warning: Giotto-TDA not available. Some features will be disabled.")

# ============================================================================
# PERSISTENT HOMOLOGY FEATURE EXTRACTOR
# ============================================================================
class PersistentHomologyExtractor:
    """Extract topological features using giotto-tda"""
    
    def __init__(self, homology_dimensions=(0, 1), n_bins=50, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        
        if GTDATDA_AVAILABLE:
            self._initialize_gtda()
        else:
            self.persistence = None
            self.entropy = None
            self.amplitudes = {}
            self.betti = None
    
    def _initialize_gtda(self):
        """Initialize giotto-tda components"""
        # Persistence diagrams
        self.persistence = VietorisRipsPersistence(
            metric='precomputed',
            homology_dimensions=self.homology_dimensions,
            collapse_edges=True,
            n_jobs=self.n_jobs
        )
        
        # Persistence entropy
        self.entropy = PersistenceEntropy(n_jobs=self.n_jobs)
        
        # Amplitude metrics
        self.amplitudes = {
            'wasserstein': Amplitude(
                metric='wasserstein',
                metric_params={'p': 1},
                n_jobs=self.n_jobs
            ),
            'bottleneck': Amplitude(
                metric='bottleneck',
                n_jobs=self.n_jobs
            ),
            'landscape': Amplitude(
                metric='landscape',
                metric_params={'n_layers': 1, 'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
            'heat': Amplitude(
                metric='heat',
                metric_params={'sigma': 0.1, 'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            ),
            'betti': Amplitude(
                metric='betti',
                metric_params={'n_bins': self.n_bins},
                n_jobs=self.n_jobs
            )
        }
        
        # Betti curves (this includes H0 and H1 curves)
        self.betti = BettiCurve(n_bins=self.n_bins, n_jobs=self.n_jobs)
    
    def adjacency_to_distance(self, adjacency_matrix):
        """Convert adjacency matrix to distance matrix"""
        distance = adjacency_matrix.copy()
        np.fill_diagonal(distance, 0)
        
        # Convert weights to distances (inverse relationship)
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = 1.0 / (distance + 1e-10)
            distance[distance > 1e6] = 1e6  # Cap large distances
        
        # Ensure symmetry
        distance = np.maximum(distance, distance.T)
        np.fill_diagonal(distance, 0)
        
        return distance
    
    def extract_features(self, adjacency_matrices):
        """
        Extract all persistent homology features
        
        Returns:
        --------
        dict: Dictionary with features
        """
        if not GTDATDA_AVAILABLE:
            return self._get_empty_features(adjacency_matrices)
        
        features = {}
        
        # 1. Convert to distance matrices
        distance_matrices = np.array([
            self.adjacency_to_distance(adj) for adj in adjacency_matrices
        ])
        
        # 2. Persistence diagrams
        persistence_diagrams = self.persistence.fit_transform(distance_matrices)
        features['persistence_diagrams'] = persistence_diagrams
        
        # 3. Persistence entropy
        entropy_features = self.entropy.fit_transform(persistence_diagrams)
        features['persistent_entropy'] = entropy_features
        
        # 4. Amplitudes (all metrics)
        amplitude_features = {}
        for metric_name, amplitude_calc in self.amplitudes.items():
            try:
                amp = amplitude_calc.fit_transform(persistence_diagrams)
                amplitude_features[metric_name] = amp
            except:
                n_samples = len(adjacency_matrices)
                n_dims = len(self.homology_dimensions)
                amplitude_features[metric_name] = np.zeros((n_samples, n_dims))
        features['amplitudes'] = amplitude_features
        
        # 5. Betti curves
        betti_curves = self.betti.fit_transform(persistence_diagrams)
        features['betti_curves'] = betti_curves
        
        return features
    
    def _get_empty_features(self, adjacency_matrices):
        """Return empty features when giotto-tda is not available"""
        n_samples = len(adjacency_matrices)
        n_dims = len(self.homology_dimensions)
        
        return {
            'persistence_diagrams': [np.array([])] * n_samples,
            'persistent_entropy': np.zeros((n_samples, n_dims)),
            'amplitudes': {
                'wasserstein': np.zeros((n_samples, n_dims)),
                'bottleneck': np.zeros((n_samples, n_dims)),
                'landscape': np.zeros((n_samples, n_dims)),
                'heat': np.zeros((n_samples, n_dims)),
                'betti': np.zeros((n_samples, n_dims))
            },
            'betti_curves': np.zeros((n_samples, n_dims * self.n_bins))
        }


class DiscriminativeFeatureGenerator:
    """
    Feature generator with only TDA features: 
    persistent entropy, amplitudes, and Betti curves
    """
    
    def __init__(self, homology_dimensions=(0, 1), n_bins=50, n_jobs=1):
        self.homology_dimensions = homology_dimensions
        self.n_bins = n_bins
        
        # Initialize components
        self.scaler = StandardScaler()
        self.ph_extractor = PersistentHomologyExtractor(
            homology_dimensions=homology_dimensions,
            n_bins=n_bins,
            n_jobs=n_jobs
        )
        
        # Feature tracking
        self.feature_dim = None
        self.feature_names = []
    
    def generate_features(self, adjacency_matrices, labels):
        """Generate discriminative features"""
        print("Generating discriminative features...")
        X = []
        y = []
        feature_info = {
            'feature_shapes': [],
            'feature_names': []
        }
        
        # First pass: determine feature dimension
        if self.feature_dim is None:
            print("Determining feature dimension...")
            test_feature = self._graph_to_discriminative_features(adjacency_matrices[0])
            self.feature_dim = len(test_feature)
            feature_info['feature_names'] = self._get_feature_names()
            print(f"Feature dimension: {self.feature_dim}")
            print(f"Giotto-TDA available: {GTDATDA_AVAILABLE}")
        
        for i, adjacency in enumerate(adjacency_matrices):
            if i % 500 == 0:
                print(f"Processing graph {i+1}/{len(adjacency_matrices)}")
            
            feature_vector = self._graph_to_discriminative_features(adjacency)
            
            # Ensure consistent feature length
            if len(feature_vector) != self.feature_dim:
                if len(feature_vector) > self.feature_dim:
                    feature_vector = feature_vector[:self.feature_dim]
                else:
                    feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)), 
                                          mode='constant')
            
            X.append(feature_vector)
            y.append(labels[i])
            feature_info['feature_shapes'].append(feature_vector.shape)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Features shape: {X.shape}")
        print(f"Total features: {X.shape[1]}")
        print(f"Feature extraction completed for {len(adjacency_matrices)} molecules")
        
        return X, y, feature_info
    
    def _graph_to_discriminative_features(self, adjacency):
        """Convert graph to discriminative features"""
        # Extract persistent homology features for this graph
        ph_features_single = self.ph_extractor.extract_features([adjacency])
        
        features = []
        
        # 1. Persistent entropy from giotto-tda
        if 'persistent_entropy' in ph_features_single:
            features.extend(ph_features_single['persistent_entropy'][0])
        
        # 2. Amplitudes from giotto-tda (all metrics)
        if 'amplitudes' in ph_features_single:
            for metric in ['wasserstein', 'bottleneck', 'landscape', 'heat', 'betti']:
                if metric in ph_features_single['amplitudes']:
                    features.extend(ph_features_single['amplitudes'][metric][0])
        
        # 3. Betti curves from giotto-tda (flattened)
        if 'betti_curves' in ph_features_single:
            betti_flat = ph_features_single['betti_curves'][0].flatten()
            features.extend(betti_flat)
        
        return np.array(features)
    
    def _get_feature_names(self):
        """Get descriptive feature names"""
        names = []
        
        # Persistent entropy
        for dim in self.homology_dimensions:
            names.append(f'persistent_entropy_H{dim}')
        
        # Amplitudes
        for metric in ['wasserstein', 'bottleneck', 'landscape', 'heat', 'betti']:
            for dim in self.homology_dimensions:
                names.append(f'amplitude_{metric}_H{dim}')
        
        # Betti curves
        for dim in self.homology_dimensions:
            for bin_idx in range(self.n_bins):
                names.append(f'betti_H{dim}_bin_{bin_idx}')
        
        return names
    
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
            feature_vector = self._graph_to_discriminative_features(adjacency)
            if len(feature_vector) != self.feature_dim:
                if len(feature_vector) > self.feature_dim:
                    feature_vector = feature_vector[:self.feature_dim]
                else:
                    feature_vector = np.pad(feature_vector, (0, self.feature_dim - len(feature_vector)), 
                                          mode='constant')
            X.append(feature_vector)
        
        X = np.array(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X
    

class SpectralFeatureGenerator:
    pass

# ============================================================================
# FEATURE COUNT CALCULATOR
# ============================================================================
def calculate_feature_counts(homology_dimensions=(0, 1), n_bins=50):
    """Calculate total number of features"""
    n_dims = len(homology_dimensions)
    
    counts = {
        'Persistent_entropy': n_dims,
        'Amplitudes': 5 * n_dims,  # 5 metrics
        'Betti_curves': n_bins * n_dims
    }
    
    total = sum(counts.values())
    
    print("FEATURE BREAKDOWN (TDA FEATURES ONLY):")
    print("=" * 40)
    for category, count in counts.items():
        print(f"{category:25s}: {count:3d} features")
    print("=" * 40)
    print(f"{'TOTAL':25s}: {total:3d} features")
    
    return total, counts