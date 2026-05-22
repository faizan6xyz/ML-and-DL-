# =============================================================================
# SCIKIT-LEARN COMPLETE ALGORITHM REFERENCE
# All major methods with every parameter explained
# =============================================================================

# pip install scikit-learn xgboost lightgbm


# =============================================================================
# 1. CLASSIFICATION
# =============================================================================

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',                    # Regularization type: 'l1', 'l2', 'elasticnet', None
    dual=False,                      # Dual formulation; only for l2 + liblinear solver
    tol=1e-4,                        # Tolerance for stopping criterion
    C=1.0,                           # Inverse regularization strength; smaller = stronger regularization
    fit_intercept=True,              # Whether to include a bias/intercept term
    intercept_scaling=1,             # Intercept scaling; only for liblinear solver
    class_weight=None,               # 'balanced' auto-weights by frequency; or dict {class: weight}
    random_state=None,               # Seed for reproducibility
    solver='lbfgs',                  # Optimizer: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    max_iter=100,                    # Max iterations for the solver to converge
    multi_class='auto',              # 'ovr' one-vs-rest, 'multinomial' softmax, 'auto' picks
    verbose=0,                       # Verbosity level for solvers
    warm_start=False,                # Reuse previous fit as starting point
    n_jobs=None,                     # Number of CPU cores (-1 = all)
    l1_ratio=None                    # ElasticNet mixing: 0 = L2, 1 = L1; only for elasticnet
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.svm import SVC

model = SVC(
    C=1.0,                           # Regularization; smaller = stronger regularization
    kernel='rbf',                    # Kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree=3,                        # Degree of polynomial kernel (ignored by others)
    gamma='scale',                   # Kernel coefficient: 'scale'=1/(n_features*X.var()), 'auto'=1/n_features
    coef0=0.0,                       # Independent term in poly/sigmoid kernel
    shrinking=True,                  # Use shrinking heuristic to speed training
    probability=False,               # Enable predict_proba(); adds 5-fold CV overhead
    tol=1e-3,                        # Stopping tolerance
    cache_size=200,                  # Kernel cache in MB; increase for large datasets
    class_weight=None,               # 'balanced' or dict {class: weight}
    verbose=False,                   # Verbosity
    max_iter=-1,                     # Max iterations (-1 = unlimited)
    decision_function_shape='ovr',   # 'ovr' one-vs-rest, 'ovo' one-vs-one for multiclass
    break_ties=False,                # Break ties via decision function (slower)
    random_state=None                # Seed; used only when probability=True
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,                   # Number of nearest neighbors to use
    weights='uniform',               # 'uniform' equal weight, 'distance' inverse distance weighting
    algorithm='auto',                # NN algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,                    # Leaf size for BallTree / KDTree (affects speed/memory)
    p=2,                             # Minkowski metric power: 1 = Manhattan, 2 = Euclidean
    metric='minkowski',              # Distance metric to use
    metric_params=None,              # Extra keyword arguments for the metric function
    n_jobs=None                      # Parallel jobs for neighbor search (-1 = all cores)
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',                # Split quality: 'gini', 'entropy', 'log_loss'
    splitter='best',                 # Split strategy: 'best' optimal, 'random' random best
    max_depth=None,                  # Max tree depth; None = unlimited (risk of overfitting)
    min_samples_split=2,             # Min samples needed to split an internal node
    min_samples_leaf=1,              # Min samples required at a leaf node
    min_weight_fraction_leaf=0.0,    # Min weighted fraction of samples at a leaf
    max_features=None,               # Features considered per split: int, float, 'sqrt', 'log2', None
    random_state=None,               # Seed for reproducibility
    max_leaf_nodes=None,             # Max number of leaf nodes (None = unlimited)
    min_impurity_decrease=0.0,       # Min impurity decrease required to make a split
    class_weight=None,               # 'balanced' or dict {class: weight}
    ccp_alpha=0.0                    # Complexity parameter for minimal cost-complexity pruning
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,                # Number of trees in the forest
    criterion='gini',                # Split quality: 'gini', 'entropy', 'log_loss'
    max_depth=None,                  # Max tree depth (None = fully grown)
    min_samples_split=2,             # Min samples to split an internal node
    min_samples_leaf=1,              # Min samples at leaf node
    min_weight_fraction_leaf=0.0,    # Min weighted fraction at leaf
    max_features='sqrt',             # Features per split: 'sqrt' (default), 'log2', int, float
    max_leaf_nodes=None,             # Max leaf nodes per tree
    min_impurity_decrease=0.0,       # Min impurity decrease to allow a split
    bootstrap=True,                  # Use bootstrap samples; False = full dataset per tree
    oob_score=False,                 # Use out-of-bag samples for free generalization estimate
    n_jobs=None,                     # Parallel jobs (-1 = all cores)
    random_state=None,               # Seed for reproducibility
    verbose=0,                       # Verbosity
    warm_start=False,                # Reuse previous fit; add more estimators incrementally
    class_weight=None,               # 'balanced', 'balanced_subsample', or dict
    ccp_alpha=0.0,                   # Cost-complexity pruning parameter
    max_samples=None                 # Samples per bootstrap draw (int or float fraction)
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    loss='log_loss',                 # Loss function: 'log_loss', 'exponential' (AdaBoost-like)
    learning_rate=0.1,               # Shrinkage applied to each tree's contribution
    n_estimators=100,                # Number of boosting stages (trees)
    subsample=1.0,                   # Fraction of samples per tree; <1 = stochastic GB
    criterion='friedman_mse',        # Split quality measure
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,                     # Tree depth; 3–5 typical for boosting
    min_impurity_decrease=0.0,
    init=None,                       # Initial estimator for base predictions
    random_state=None,
    max_features=None,               # Features per split (same as DecisionTree)
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,         # Fraction held out for early stopping
    n_iter_no_change=None,           # Stop if score flat for this many iterations
    tol=1e-4,                        # Min improvement for early stopping
    ccp_alpha=0.0
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.naive_bayes import GaussianNB

model = GaussianNB(
    priors=None,                     # Prior probabilities; None = learned from data
    var_smoothing=1e-9               # Portion of largest variance added for numerical stability
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(
    alpha=1.0,                       # Additive (Laplace/Lidstone) smoothing; 0 = no smoothing
    force_alpha=True,                # Raise error if alpha=0 and feature has zero count
    fit_prior=True,                  # Learn class prior probabilities from data
    class_prior=None                 # Manual class priors; overrides fit_prior if set
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
    hidden_layer_sizes=(100,),       # Neurons per hidden layer: (100,) = 1 layer, (100,50) = 2 layers
    activation='relu',               # Activation: 'identity', 'logistic', 'tanh', 'relu'
    solver='adam',                   # Optimizer: 'lbfgs' (small data), 'sgd', 'adam' (large data)
    alpha=0.0001,                    # L2 regularization penalty
    batch_size='auto',               # Mini-batch size; 'auto' = min(200, n_samples)
    learning_rate='constant',        # LR schedule: 'constant', 'invscaling', 'adaptive' (sgd only)
    learning_rate_init=0.001,        # Initial learning rate
    power_t=0.5,                     # Exponent for inverse scaling schedule (sgd only)
    max_iter=200,                    # Max epochs
    shuffle=True,                    # Shuffle samples each iteration
    random_state=None,
    tol=1e-4,                        # Optimization tolerance
    verbose=False,
    warm_start=False,
    momentum=0.9,                    # SGD momentum (sgd solver only)
    nesterovs_momentum=True,         # Use Nesterov momentum
    early_stopping=False,            # Stop when validation score stops improving
    validation_fraction=0.1,         # Fraction for early stopping validation set
    beta_1=0.9,                      # Adam: decay rate for 1st moment estimates
    beta_2=0.999,                    # Adam: decay rate for 2nd moment estimates
    epsilon=1e-8,                    # Adam: numerical stability constant
    n_iter_no_change=10,             # Early stopping patience (epochs)
    max_fun=15000                    # Max function evaluations (lbfgs only)
)


# =============================================================================
# 2. REGRESSION
# =============================================================================

from sklearn.linear_model import LinearRegression

model = LinearRegression(
    fit_intercept=True,              # Estimate intercept; False if data is already centered
    copy_X=True,                     # Copy X before fitting; False modifies in-place (saves memory)
    n_jobs=None,                     # Parallel jobs for large arrays (-1 = all cores)
    positive=False                   # Constrain all coefficients >= 0 (non-negative LS)
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import Ridge

model = Ridge(
    alpha=1.0,                       # L2 regularization strength; larger = more shrinkage
    fit_intercept=True,
    copy_X=True,
    max_iter=None,                   # Max iterations for CG/sparse solvers
    tol=1e-4,                        # Precision of solution
    solver='auto',                   # 'auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'
    positive=False,
    random_state=None
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import Lasso

model = Lasso(
    alpha=1.0,                       # L1 regularization strength; higher = more zero coefficients
    fit_intercept=True,
    precompute=False,                # Precompute Gram matrix for speed (True or array)
    copy_X=True,
    max_iter=1000,                   # Max iterations; increase if ConvergenceWarning appears
    tol=1e-4,
    warm_start=False,
    positive=False,
    random_state=None,
    selection='cyclic'               # Coefficient update order: 'cyclic' or 'random' (faster convergence)
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import ElasticNet

model = ElasticNet(
    alpha=1.0,                       # Overall regularization strength
    l1_ratio=0.5,                    # Mix ratio: 0 = Ridge (L2), 1 = Lasso (L1)
    fit_intercept=True,
    precompute=False,
    max_iter=1000,
    copy_X=True,
    tol=1e-4,
    warm_start=False,
    positive=False,
    random_state=None,
    selection='cyclic'
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.svm import SVR

model = SVR(
    kernel='rbf',                    # Kernel: 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    degree=3,                        # Degree for polynomial kernel
    gamma='scale',                   # Kernel coefficient: 'scale' or 'auto' or float
    coef0=0.0,                       # Independent term in poly/sigmoid
    tol=1e-3,
    C=1.0,                           # Penalty for errors outside the epsilon tube
    epsilon=0.1,                     # Width of the insensitive tube; points inside not penalized
    shrinking=True,
    cache_size=200,
    verbose=False,
    max_iter=-1
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    criterion='squared_error',       # Split quality: 'squared_error','absolute_error','friedman_mse','poisson'
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=1.0,                # Default 1.0 (all features) for regression; 'sqrt' for classification
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    loss='squared_error',            # Loss: 'squared_error','absolute_error','huber','quantile'
    learning_rate=0.1,
    n_estimators=100,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,                       # Quantile for 'huber' and 'quantile' losses
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0
)


# =============================================================================
# 3. CLUSTERING
# =============================================================================

from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=8,                    # Number of clusters (the K value)
    init='k-means++',                # Centroid init: 'k-means++', 'random', or array
    n_init='auto',                   # Runs with different seeds; keeps best inertia
    max_iter=300,                    # Max iterations per run
    tol=1e-4,                        # Relative tolerance for convergence
    verbose=0,
    random_state=None,
    copy_x=True,                     # Pre-center data for numerical stability
    algorithm='lloyd'                # 'lloyd' standard EM, 'elkan' faster for well-separated clusters
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,                         # Max distance between two points to be considered neighbors
    min_samples=5,                   # Min neighbors for a point to be a core point
    metric='euclidean',              # Distance metric (normalize data first!)
    metric_params=None,              # Extra keyword args for metric
    algorithm='auto',                # NN algorithm: 'auto','ball_tree','kd_tree','brute'
    leaf_size=30,
    p=None,                          # Minkowski power (if metric='minkowski')
    n_jobs=None
)
# Note: cluster label -1 = noise/outlier

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=2,                    # Number of clusters; set None when using distance_threshold
    metric='euclidean',              # Distance metric: 'euclidean','l1','l2','manhattan','cosine'
    memory=None,                     # Cache computation (path string or joblib.Memory)
    connectivity=None,               # Connectivity matrix to constrain merges
    compute_full_tree='auto',        # Compute full tree even if early stop possible
    linkage='ward',                  # Merge strategy: 'ward','complete','average','single'
    distance_threshold=None,         # Cut dendrogram at this distance (n_clusters must be None)
    compute_distances=False          # Compute distances between clusters (needed for dendrogram plot)
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.cluster import MeanShift

model = MeanShift(
    bandwidth=None,                  # Kernel bandwidth; None = estimated via estimate_bandwidth()
    seeds=None,                      # Seeds for kernels; None = auto
    bin_seeding=False,               # Use binned seeds for speedup (requires bandwidth)
    min_bin_freq=1,                  # Min points per bin to be a seed
    cluster_all=True,                # False = orphan points labeled -1
    n_jobs=None,
    max_iter=300                     # Max iterations per mean shift update
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.cluster import SpectralClustering

model = SpectralClustering(
    n_clusters=8,                    # Number of clusters
    eigen_solver=None,               # Eigenvalue decomposition: None,'arpack','lobpcg','amg'
    n_components=None,               # Dimensions of spectral embedding (default = n_clusters)
    random_state=None,
    n_init=10,                       # KMeans restarts on spectral embedding
    gamma=1.0,                       # Kernel coefficient for rbf/poly/sigmoid/laplacian
    affinity='rbf',                  # Affinity: 'nearest_neighbors','rbf','precomputed', or callable
    n_neighbors=10,                  # Neighbors for nearest_neighbors affinity
    eigen_tol='auto',                # Stopping criterion for eigendecomposition
    assign_labels='kmeans',          # Label assignment: 'kmeans','discretize','cluster_qr'
    degree=3,                        # Poly kernel degree
    coef0=1,                         # Poly/sigmoid kernel term
    kernel_params=None,              # Keyword args for callable affinity
    n_jobs=None,
    verbose=False
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.mixture import GaussianMixture

model = GaussianMixture(
    n_components=1,                  # Number of Gaussian mixture components
    covariance_type='full',          # Covariance: 'full','tied','diag','spherical'
    tol=1e-3,                        # EM convergence threshold
    reg_covar=1e-6,                  # Regularization added to covariance diagonal (prevents singular matrix)
    max_iter=100,                    # Max EM iterations
    n_init=1,                        # Number of initializations; keeps best lower bound
    init_params='kmeans',            # Init method: 'kmeans','k-means++','random','random_from_data'
    weights_init=None,               # Initial mixture weights
    means_init=None,                 # Initial means
    precisions_init=None,            # Initial precisions (inverse covariances)
    random_state=None,
    warm_start=False,
    verbose=0,
    verbose_interval=10
)


# =============================================================================
# 4. DIMENSIONALITY REDUCTION
# =============================================================================

from sklearn.decomposition import PCA

model = PCA(
    n_components=None,               # Dims to keep: int, float (variance %), 'mle', None (keep all)
    copy=True,                       # Copy X before fitting
    whiten=False,                    # Divide by sqrt(eigenvalue) → unit variance per component
    svd_solver='auto',               # 'auto','full','arpack','randomized','covariance_eig'
    tol=0.0,                         # Tolerance for arpack solver
    iterated_power='auto',           # Power iterations for randomized solver
    n_oversamples=10,                # Extra random vectors for randomized SVD accuracy
    power_iteration_normalizer='auto',
    random_state=None
)
# Tip: pass n_components=0.95 to keep 95% of variance automatically

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.manifold import TSNE

model = TSNE(
    n_components=2,                  # Output dimensions (usually 2 or 3 for visualization)
    perplexity=30.0,                 # Balance local vs global structure; typical 5–50
    early_exaggeration=12.0,         # Cluster tightness in early optimization phase
    learning_rate='auto',            # 'auto' or float; 'auto' = max(N/early_exaggeration/4, 50)
    max_iter=1000,                   # Max optimization iterations
    n_iter_without_progress=300,     # Stop if no progress after N iterations
    min_grad_norm=1e-7,              # Stop if gradient norm below this
    metric='euclidean',              # Distance metric
    metric_params=None,
    init='pca',                      # 'pca' recommended (stable), 'random'
    verbose=0,
    random_state=None,
    method='barnes_hut',             # 'barnes_hut' O(NlogN), 'exact' O(N²) for small datasets
    angle=0.5,                       # Speed/accuracy tradeoff for barnes_hut (0.2–0.8)
    n_jobs=None
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.decomposition import TruncatedSVD  # Also known as LSA for text

model = TruncatedSVD(
    n_components=2,                  # Output dimensions (must be < n_features)
    algorithm='randomized',          # 'randomized' (fast, large sparse), 'arpack' (exact)
    n_iter=5,                        # Power iterations for randomized SVD
    n_oversamples=10,                # Extra random samples for accuracy
    power_iteration_normalizer='auto',
    random_state=None,
    tol=0.0                          # Tolerance for arpack solver
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.decomposition import NMF  # Non-Negative Matrix Factorization

model = NMF(
    n_components=None,               # Number of components (latent features)
    init=None,                       # Init method: None,'random','nndsvd','nndsvda','nndsvdar','custom'
    solver='cd',                     # Solver: 'cd' coordinate descent, 'mu' multiplicative update
    beta_loss='frobenius',           # Beta loss: 'frobenius','kullback-leibler','itakura-saito'
    tol=1e-4,
    max_iter=200,
    random_state=None,
    alpha_W=0.0,                     # L1/L2 regularization on W matrix
    alpha_H='same',                  # Regularization on H matrix; 'same' copies alpha_W
    l1_ratio=0.0,                    # L1 vs L2 mix: 0 = L2, 1 = L1
    verbose=0,
    shuffle=False                    # Shuffle coordinates in cd solver
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

model = LinearDiscriminantAnalysis(
    solver='svd',                    # 'svd','lsqr','eigen'
    shrinkage=None,                  # Shrinkage: None,'auto' (Ledoit-Wolf), or float 0–1
    priors=None,                     # Class prior probabilities
    n_components=None,               # Dimensions for dimensionality reduction
    store_covariance=False,          # Store estimated covariance matrix
    tol=1e-4,                        # Threshold for rank estimation in SVD solver
    covariance_estimator=None        # Custom covariance estimator object
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.manifold import Isomap

model = Isomap(
    n_neighbors=5,                   # Number of neighbors for manifold construction
    n_components=2,                  # Output dimensions
    eigen_solver='auto',             # 'auto','arpack','dense'
    tol=0,                           # Convergence tolerance (arpack)
    max_iter=None,                   # Max iterations (arpack)
    path_method='auto',              # Shortest path: 'auto','FW','D'
    neighbors_algorithm='auto',      # NN algorithm: 'auto','brute','kd_tree','ball_tree'
    n_jobs=None,
    metric='minkowski',              # Distance metric
    p=2,                             # Minkowski power parameter
    metric_params=None
)


# =============================================================================
# 5. PREPROCESSING
# =============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(
    copy=True,                       # Copy X; False modifies in place (saves memory)
    with_mean=True,                  # Subtract mean; set False for sparse matrices
    with_std=True                    # Scale to unit variance
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(
    feature_range=(0, 1),            # Desired output range; use (-1, 1) for tanh activations
    copy=True,
    clip=False                       # Clip test values that fall outside the training range
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(
    with_centering=True,             # Subtract median before scaling
    with_scaling=True,               # Scale by interquartile range
    quantile_range=(25.0, 75.0),     # IQR range used for scaling; widen to be less sensitive to outliers
    copy=True,
    unit_variance=False              # Rescale to match StandardScaler's output variance
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(
    categories='auto',               # 'auto' infer categories, or list of arrays per feature
    drop=None,                       # Drop redundant column: None, 'first', 'if_binary'
    sparse_output=True,              # Return sparse matrix; set False for dense array
    dtype=float,                     # Output dtype
    handle_unknown='error',          # 'error', 'ignore' (zeros), 'infrequent_if_exist'
    min_frequency=None,              # Min count to be own category (else → infrequent)
    max_categories=None,             # Max categories per feature; extras → infrequent
    feature_name_combiner='concat'   # How to build output column names
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
# No constructor parameters.
# Use for encoding the target y (not features X — use OrdinalEncoder for that).
# le.fit_transform(y)  →  array of integers 0..n_classes-1
# le.inverse_transform(encoded)  →  original labels
# le.classes_  →  ordered array of unique labels

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder(
    categories='auto',               # 'auto' or list of arrays defining category order
    dtype=float,
    handle_unknown='error',          # 'error', 'use_encoded_value'
    unknown_value=None,              # Value for unknown categories (handle_unknown='use_encoded_value')
    encoded_missing_value=float('nan'),  # Value for NaN entries
    min_frequency=None,
    max_categories=None
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,                        # Polynomial degree; 2 adds x², x*y terms
    interaction_only=False,          # True = only cross terms (no x²), False = include powers
    include_bias=True,               # Include column of ones (bias/intercept term)
    order='C'                        # Output array memory order: 'C' row-major, 'F' column-major
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import Normalizer

norm = Normalizer(
    norm='l2',                       # Normalization: 'l1' sum=1, 'l2' unit norm, 'max' max=1
    copy=True
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=float('nan'),     # Placeholder for missing values (default NaN)
    strategy='mean',                 # 'mean','median','most_frequent','constant'
    fill_value=None,                 # Value used when strategy='constant'
    copy=True,
    add_indicator=False,             # Add binary columns indicating imputed rows
    keep_empty_features=False        # Keep features that are entirely NaN after fit
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.impute import KNNImputer

imputer = KNNImputer(
    missing_values=float('nan'),
    n_neighbors=5,                   # Number of nearest neighbors for imputation
    weights='uniform',               # 'uniform' equal, 'distance' inverse-distance weighting
    metric='nan_euclidean',          # Distance metric ignoring NaN positions
    copy=True,
    add_indicator=False,
    keep_empty_features=False
)


# =============================================================================
# 6. FEATURE SELECTION
# =============================================================================

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

selector = SelectKBest(
    score_func=f_classif,            # Scoring function: f_classif, chi2, mutual_info_classif, f_regression
    k=10                             # Number of top features to select; 'all' keeps all
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.feature_selection import SelectPercentile

selector = SelectPercentile(
    score_func=f_classif,            # Same scoring functions as SelectKBest
    percentile=10                    # Percentage of features to keep (top 10% = best 10%)
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.feature_selection import RFE  # Recursive Feature Elimination

selector = RFE(
    estimator=None,                  # Estimator with coef_ or feature_importances_ (e.g. LinearSVC)
    n_features_to_select=None,       # Target number of features; None = half of input features
    step=1,                          # Features removed per iteration; float = fraction
    verbose=0
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.feature_selection import RFECV  # RFE with cross-validation

selector = RFECV(
    estimator=None,                  # Estimator with coef_ or feature_importances_
    step=1,                          # Features removed per iteration
    min_features_to_select=1,        # Lower bound on features to select
    cv=5,                            # Cross-validation strategy
    scoring=None,                    # Scoring metric
    verbose=0,
    n_jobs=None,
    importance_getter='auto'         # 'auto' uses coef_ or feature_importances_; or callable
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(
    threshold=0.0                    # Min variance to keep a feature; 0 removes zero-variance features
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    estimator=None,                  # Fitted or unfitted estimator with coef_/feature_importances_
    threshold=None,                  # Threshold string: 'mean','median','1.25*mean', or float
    prefit=False,                    # True if estimator is already fitted
    norm_order=1,                    # Norm for multi-output coef_ (l1 or l2)
    max_features=None,               # Max features to select
    importance_getter='auto'
)


# =============================================================================
# 7. MODEL SELECTION & EVALUATION
# =============================================================================

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(
    estimator=None,                  # Model object to tune
    param_grid=None,                 # Dict or list of dicts: {'C':[0.1,1,10],'kernel':['rbf','linear']}
    scoring=None,                    # Metric: 'accuracy','roc_auc','f1','neg_mean_squared_error',etc.
    n_jobs=None,                     # Parallel jobs (-1 = all cores)
    refit=True,                      # Refit best model on full data so .predict() works directly
    cv=5,                            # CV strategy: int, KFold(), StratifiedKFold(), etc.
    verbose=0,
    pre_dispatch='2*n_jobs',         # Controls jobs dispatched to limit memory usage
    error_score=float('nan'),        # Value assigned if fit fails
    return_train_score=False         # Include training scores in cv_results_
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import RandomizedSearchCV

rs = RandomizedSearchCV(
    estimator=None,
    param_distributions=None,        # Dict: param → scipy distribution or list to sample from
    n_iter=10,                       # Number of parameter settings to sample; higher = more thorough
    scoring=None,
    n_jobs=None,
    refit=True,
    cv=5,
    verbose=0,
    pre_dispatch='2*n_jobs',
    random_state=None,               # Seed for reproducibility
    error_score=float('nan'),
    return_train_score=False
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import cross_val_score

# cross_val_score is a function, not a class:
# scores = cross_val_score(
#     estimator,                     # Model to evaluate
#     X,                             # Feature matrix
#     y=None,                        # Target vector
#     groups=None,                   # Group labels for GroupKFold
#     scoring=None,                  # Scoring metric string or callable
#     cv=None,                       # CV strategy; default = 5-fold stratified for classifiers
#     n_jobs=None,                   # Parallel jobs
#     verbose=0,
#     fit_params=None,               # Extra params passed to estimator.fit()
#     pre_dispatch='2*n_jobs',
#     error_score=float('nan')
# )

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import KFold

kf = KFold(
    n_splits=5,                      # Number of folds
    shuffle=False,                   # Shuffle before splitting; set True with random_state
    random_state=None
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(
    n_splits=5,                      # Number of folds
    shuffle=False,                   # Shuffle data before splitting
    random_state=None
)
# Preserves class distribution in each fold — preferred for classification

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.pipeline import Pipeline

pipe = Pipeline(
    steps=None,                      # List of (name, estimator) tuples; last step = final estimator
    memory=None,                     # Cache fitted transformers (path or joblib.Memory)
    verbose=False                    # Print elapsed time for each step
)
# Example usage:
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('pca',    PCA(n_components=10)),
#     ('clf',    SVC(C=1.0))
# ])

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.pipeline import FeatureUnion

fu = FeatureUnion(
    transformer_list=None,           # List of (name, transformer) tuples
    n_jobs=None,                     # Parallel jobs
    transformer_weights=None,        # Dict {name: weight} multiplying transformer output
    verbose=False,
    verbose_feature_names_out=True   # Prefix output feature names with transformer name
)


# =============================================================================
# 8. ENSEMBLE METHODS
# =============================================================================

from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(
    estimators=None,                 # List of (name, estimator) tuples
    voting='hard',                   # 'hard' majority vote, 'soft' average probabilities
    weights=None,                    # Per-estimator weights (default = equal)
    n_jobs=None,
    flatten_transform=True,          # Flatten transform output for soft voting
    verbose=False
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import VotingRegressor

vr = VotingRegressor(
    estimators=None,                 # List of (name, estimator) tuples
    weights=None,                    # Per-estimator weights for averaging predictions
    n_jobs=None,
    verbose=False
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import StackingClassifier

sc = StackingClassifier(
    estimators=None,                 # Base-level (name, estimator) list; predictions become meta-features
    final_estimator=None,            # Meta-classifier (default LogisticRegression)
    cv=5,                            # Folds for out-of-fold meta-feature generation
    stack_method='auto',             # 'auto','predict_proba','decision_function','predict'
    n_jobs=None,
    passthrough=False,               # Also pass original X to final_estimator
    verbose=0
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import StackingRegressor

sr = StackingRegressor(
    estimators=None,                 # Base-level (name, estimator) list
    final_estimator=None,            # Meta-regressor (default RidgeCV)
    cv=5,
    n_jobs=None,
    passthrough=False,
    verbose=0
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(
    estimator=None,                  # Base estimator (default DecisionTreeClassifier)
    n_estimators=10,                 # Number of base estimators
    max_samples=1.0,                 # Samples per base estimator: int or float fraction
    max_features=1.0,                # Features per base estimator: int or float fraction
    bootstrap=True,                  # Sample with replacement
    bootstrap_features=False,        # Sample features with replacement
    oob_score=False,                 # Use out-of-bag samples for score estimate
    warm_start=False,
    n_jobs=None,
    random_state=None,
    verbose=0
)

# ─────────────────────────────────────────────────────────────────────────────

from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    estimator=None,                  # Base estimator (default DecisionTreeClassifier depth=1)
    n_estimators=50,                 # Max number of estimators (early stop if perfect fit)
    learning_rate=1.0,               # Shrinks contribution of each classifier; trade-off with n_estimators
    algorithm='SAMME',               # 'SAMME' discrete boosting (only option since sklearn 1.6)
    random_state=None
)


# =============================================================================
# 9. XGBOOST  (pip install xgboost)
# =============================================================================

import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=100,                # Number of boosting rounds
    max_depth=6,                     # Max tree depth; 3–10 typical
    learning_rate=0.3,               # Alias eta; shrinks contribution per step
    subsample=1.0,                   # Row subsampling ratio per tree
    colsample_bytree=1.0,            # Column subsampling ratio per tree
    colsample_bylevel=1.0,           # Column subsampling ratio per level
    colsample_bynode=1.0,            # Column subsampling ratio per split
    reg_alpha=0,                     # L1 regularization on weights
    reg_lambda=1,                    # L2 regularization on weights
    gamma=0,                         # Min loss reduction to make a split (alias min_split_loss)
    min_child_weight=1,              # Min sum of instance weight in child; higher = more conservative
    max_delta_step=0,                # Max delta step per tree; 0 = unconstrained
    scale_pos_weight=1,              # Balances positive/negative weights (use sum_neg/sum_pos)
    base_score=0.5,                  # Initial prediction score (global bias)
    booster='gbtree',                # 'gbtree','gblinear','dart'
    n_jobs=-1,                       # Parallel threads (-1 = all cores)
    random_state=42,
    objective='binary:logistic',     # Loss: 'binary:logistic','multi:softmax','reg:squarederror',etc.
    eval_metric=None,                # Evaluation metric for watchlist
    early_stopping_rounds=None,      # Stop if eval metric flat for N rounds
    use_label_encoder=False,
    tree_method='hist',              # 'auto','exact','approx','hist' (fastest for large data)
    device='cpu',                    # 'cpu' or 'cuda' for GPU
    importance_type='gain'           # Feature importance type: 'gain','weight','cover','total_gain'
)

# XGBRegressor uses the same parameters with objective='reg:squarederror' by default
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.3,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0,
    reg_lambda=1,
    objective='reg:squarederror',    # 'reg:squarederror','reg:absoluteerror','reg:tweedie',etc.
    n_jobs=-1,
    random_state=42
)


# =============================================================================
# 10. LIGHTGBM  (pip install lightgbm)
# =============================================================================

import lightgbm as lgb

model = lgb.LGBMClassifier(
    boosting_type='gbdt',            # 'gbdt','dart','goss','rf'
    num_leaves=31,                   # Max leaves per tree; increase for complex data
    max_depth=-1,                    # Max depth (-1 = unlimited); controls num_leaves
    learning_rate=0.1,               # Shrinkage rate; lower needs more n_estimators
    n_estimators=100,                # Number of boosted trees
    subsample_for_bin=200000,        # Samples for constructing feature bins
    objective=None,                  # 'binary','multiclass','regression',etc. (auto-inferred)
    class_weight=None,               # 'balanced' or dict
    min_split_gain=0.0,              # Min gain to perform a split
    min_child_weight=1e-3,           # Min sum hessian in leaf
    min_child_samples=20,            # Min samples in leaf; key regularization parameter
    subsample=1.0,                   # Row subsampling fraction
    subsample_freq=0,                # Subsampling frequency; 0 = disabled
    colsample_bytree=1.0,            # Feature subsampling fraction per tree
    reg_alpha=0.0,                   # L1 regularization
    reg_lambda=0.0,                  # L2 regularization
    random_state=None,
    n_jobs=-1,
    importance_type='split',         # Feature importance: 'split' (times used) or 'gain' (total gain)
    device_type='cpu',               # 'cpu' or 'gpu'
    # Additional important params (pass via **kwargs or set_params):
    # num_class                       # For multiclass; required when objective='multiclass'
    # early_stopping_rounds           # Stop if metric flat for N rounds
    # verbose=-1                      # Suppress training output
)

# LGBMRegressor uses same params; objective defaults to 'regression'
model = lgb.LGBMRegressor(
    n_estimators=100,
    max_depth=-1,
    num_leaves=31,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None,
    n_jobs=-1
)
