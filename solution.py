import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path, delimiter=';', encoding='utf-8-sig')

# Compounds direpresentasikan oleh com_smiles
def com_smiles_to_fingerprint(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # ekstraksi fitur menggunakan fingerprint PaDEL-Pubchem melalui sdk rdkit
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return None

def cluster_fingerprints(data):
    fingerprints = data['com_smiles'].apply(lambda x: com_smiles_to_fingerprint(x))
    fingerprints = data['com_smiles'].apply(lambda x: com_smiles_to_fingerprint(x))
    fingerprint_matrix = np.array(list(fingerprints.dropna()))
    # agglomerative hierarchical clustering
    # clustering menggunakan single linkage, metric (ukuran jarak) yang digunakan adalah Euclidean distance

    linked = linkage(fingerprint_matrix, method='single', metric='euclidean')

    max_d = 0
    for i in range(1, len(linked)):
        if len(np.unique(fcluster(linked, i, criterion='maxclust'))) == 3:
            max_d = linked[i-1, 2]
            break

    plt.figure(figsize=(10, 5))
    dendrogram(linked,
               orientation='top',
               labels=data['com_id'].values,
               distance_sort='descending',
               show_leaf_counts=True,
               color_threshold=max_d)
    plt.show()

    cluster_assignments = fcluster(linked, 3, criterion='maxclust')
    data['cluster'] = cluster_assignments
    print("========== Hasil Cluter ==========")
    print(data[['com_id', 'cluster']])

def main():
    data = load_data("dataset.csv")
    cluster_fingerprints(data)

if __name__ == "__main__":
    main()
