import faiss
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
import numpy as np

base = pd.read_csv("data/base.csv", index_col="Id")
scaler = RobustScaler()
base_scale = scaler.fit_transform(base)
base_data = base_scale.astype(np.float32)

N_CLASTERS = 460
USE_PCA = True
PCA_DIMENSIONS = 65

def create_and_train_index(data, 
                           n_clusters=None, 
                           use_pca=False, 
                           pca_dimensions=None, 
                           metric=faiss.METRIC_L2):
    d = data.shape[1]
    
    # Применение PCA, если требуется
    if use_pca and pca_dimensions:
        pca_matrix = faiss.PCAMatrix(d, pca_dimensions, eigen_power=-0.5)
        pca_matrix.train(data)
        data = pca_matrix.apply_py(data)
        d = pca_dimensions  # Новая размерность после PCA

    # Создание индекса
    if n_clusters:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, metric)
        index.train(data)
    else:
        index = faiss.IndexFlatL2(d)

    index.add(data)
    return index, (pca_matrix if use_pca and pca_dimensions else None)
    
index, pca_matrix = create_and_train_index(base_data, 
                                           n_clusters=N_CLASTERS,
                                           use_pca=USE_PCA,
                                           pca_dimensions=PCA_DIMENSIONS
                                          )

# Сохранение
faiss.write_index(index, 'server_matching/model/index.faiss')

with open('server_matching/model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

if pca_matrix:
    faiss.write_VectorTransform(pca_matrix, 'server_matching/model/pca_matrix.faiss')
