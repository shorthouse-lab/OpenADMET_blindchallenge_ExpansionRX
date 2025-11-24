# features.py
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import MACCSkeys
from rdkit import Chem
import numpy as np
import env
import numpy as np
import functools, hashlib, os, pickle, json

# Optional deepchem import (only needed for MOL2VEC / GRAPH)
try:
    import deepchem as dc
    _DEEPCHEM_AVAILABLE = True
except ImportError:
    dc = None
    _DEEPCHEM_AVAILABLE = False
    print("[features] deepchem not available: MOL2VEC and GRAPH will be skipped unless installed.")

# Simple disk cache for fingerprint generation to speed up reruns.
# Cache key built from feature name + smiles list hash + fp size/radius parameters.
CACHE_DIR = os.getenv('FEATURE_CACHE_DIR', './feature_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def _hash_smiles(smiles_list):
    h = hashlib.sha256('\n'.join(smiles_list).encode()).hexdigest()
    return h

def cached_feature(func):
    @functools.wraps(func)
    def wrapper(mols, *args, **kwargs):
        # Build smiles list for hashing
        smiles_list = []
        for mol in mols:
            if mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol, canonical=True))
            else:
                smiles_list.append('')
        param_repr = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        key = f"{func.__name__}_{_hash_smiles(smiles_list)}_{hashlib.md5(param_repr.encode()).hexdigest()}"
        path = os.path.join(CACHE_DIR, key + '.pkl')
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass  # fall through to regeneration
        result = func(mols, *args, **kwargs)
        try:
            with open(path, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            pass
        return result
    return wrapper

def generate(mols, gen):
    fingerprints = []
    valid_indices = []  # Track which molecules were successfully processed
    failed_count = 0
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = gen.GetFingerprint(mol)
                fingerprints.append(np.array(fp))
                valid_indices.append(i)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return np.stack(fingerprints), valid_indices
    else:
        return np.array([]), []

@cached_feature
def AtomPair(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(includeChirality=True, fpSize=fpSize)
    return generate(mols, gen)

@cached_feature
def ECFP(mols, radius=2, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetMorganGenerator(includeChirality=True, radius=radius, fpSize=fpSize)
    return generate(mols, gen)

@cached_feature
def MACCS(mols):
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprints.append(np.array(fp))
                valid_indices.append(i)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return np.stack(fingerprints), valid_indices
    else:
        return np.array([]), []

@cached_feature
def RDKitFP(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
    return generate(mols, gen)

@cached_feature
def TOPOTOR(mols, fpSize=env.DEFAULT_FP_SIZE):
    gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(includeChirality=True, fpSize=fpSize)
    return generate(mols, gen)

@cached_feature
def MOL2VEC(mols):
    if not _DEEPCHEM_AVAILABLE:
        raise RuntimeError("MOL2VEC requested but deepchem is not installed. Install deepchem or remove MOL2VEC from config.")
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    featurizer = dc.deepchem.feat.Mol2VecFingerprint()
    
    for i, mol in enumerate(mols):
        try:
            if mol is not None:
                fp = featurizer.featurize([mol])
                if fp is not None and len(fp) > 0 and fp[0] is not None:
                    fingerprints.append(fp[0])
                    valid_indices.append(i)
                else:
                    failed_count += 1
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    if fingerprints:
        return np.stack(fingerprints), valid_indices
    else:
        return np.array([]), []

@cached_feature
def GRAPH(mols):
    if not _DEEPCHEM_AVAILABLE:
        raise RuntimeError("GRAPH features requested but deepchem is not installed. Install deepchem to use GCN/GAT.")
    fingerprints = []
    valid_indices = []
    failed_count = 0
    
    featurizer = dc.deepchem.feat.MolGraphConvFeaturizer()

    # Featurize all molecules at once
    all_features = featurizer.featurize(mols)
    
    # Check each featurized result
    for i, fp in enumerate(all_features):
        # Check if it's a valid GraphData object
        if hasattr(fp, 'node_features') and hasattr(fp, 'edge_index'):
            # Additional check to ensure it's not empty
            if fp.node_features.shape[0] > 0:
                fingerprints.append(fp)
                valid_indices.append(i)
            else:
                failed_count += 1
        # Check if it's an empty array (the failure case you observed)
        elif isinstance(fp, np.ndarray) and fp.size == 0:
            failed_count += 1
        # Handle any other unexpected types
        else:
            failed_count += 1
    
    print(f"Feature failed for {failed_count} molecules")
    
    return np.array(fingerprints), valid_indices

def getFeature(mols, fingerprint: str):
    if fingerprint not in globals():
        raise ValueError(f"Unknown fingerprint type: {fingerprint}")
    
    func = globals()[fingerprint]
    if not callable(func):
        raise ValueError(f"{fingerprint} is not a callable function")
    
    try:
        return func(mols)
    except RuntimeError as e:
        # Provide a clearer message upstream
        print(f"[features] Skipping {fingerprint}: {e}")
        return np.array([]), []