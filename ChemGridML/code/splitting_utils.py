import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

def gen_butina_clusters(smiles_list, cutoff=0.6, fp_radius=2, fp_bits=1024):
    """
    Generates Butina clusters for a list of SMILES.
    Returns an array of integer group IDs.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, fp_radius, nBits=fp_bits)
           for x in mols if x is not None]

    if not fps:
        return np.arange(len(smiles_list))

    # Calculate distance matrix (1 - Similarity)
    # This is the expensive step we want to do only once!
    dists = []
    n_fps = len(fps)
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - x for x in sims])

    clusters = Butina.ClusterData(dists, n_fps, cutoff, isDistData=True, reordering=True)

    group_ids = np.arange(len(smiles_list), dtype=int)
    valid_mask = [x is not None for x in mols]
    valid_indices = np.where(valid_mask)[0]

    for cluster_id, member_indices in enumerate(clusters):
        for member_idx in member_indices:
            original_idx = valid_indices[member_idx]
            group_ids[original_idx] = cluster_id

    return group_ids

def butina_train_test_split(X, Y, smiles, test_size=0.2, random_state=42, cutoff=0.6):
    """Helper for the outer split (runs once per experiment)"""
    groups = gen_butina_clusters(smiles, cutoff=cutoff)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, Y, groups=groups))
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx], train_idx, test_idx

class ButinaKFold:
    """
    Efficient Cluster-aware CV splitter.
    Computes clusters ONCE during initialization.
    """
    def __init__(self, n_splits=5, smiles=None, random_state=42, shuffle=True, cutoff=0.6):
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

        if smiles is None:
            raise ValueError("ButinaKFold requires 'smiles' in __init__")

        # --- OPTIMIZATION: Compute clusters once and store them ---
        self.groups = gen_butina_clusters(smiles, cutoff=cutoff)
        self.indices = np.arange(len(smiles))

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Yields train/test indices based on pre-calculated clusters"""

        # Shuffle logic (Pat Walters' fix for GroupKFold deterministic behavior)
        working_indices = self.indices
        working_groups = self.groups

        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(len(working_indices))
            working_indices = working_indices[perm]
            working_groups = working_groups[perm]

        # Use GroupKFold with our pre-calculated groups
        gkf = GroupKFold(n_splits=self.n_splits)

        for train_idx_perm, val_idx_perm in gkf.split(working_indices, y=None, groups=working_groups):
            # Map back to original indices
            if self.shuffle:
                yield working_indices[train_idx_perm], working_indices[val_idx_perm]
            else:
                yield train_idx_perm, val_idx_perm
