from fastapi import FastAPI, HTTPException
import faiss
import pickle
import numpy as np
import uvicorn
from pydantic import BaseModel
import pandas as pd

class SampleModel(BaseModel):
    sample: list

app = FastAPI()

base = pd.read_csv("../data/base.csv", index_col="Id")

# Загрузка индекса, PCA матрицы и скейлера
index = faiss.read_index("model/index.faiss")
pca_matrix = faiss.read_VectorTransform("model/pca_matrix.faiss")
with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.post("/find_neighbors/")
def find_neighbors(sample_data: SampleModel):
    try:
        sample = sample_data.sample
        print("Полученный sample:", sample)

        sample_array = np.array(sample).astype(np.float32)
        sample_scaled = scaler.transform(sample_array)
        sample_pca = pca_matrix.apply_py(sample_scaled)

        # Поиск ближайших соседей
        k = 5
        distances, indices = index.search(sample_pca, k)

        # Получение меток для найденных индексов
        labels = [base.index[idx] for idx in indices[0]]
        return {"neighbors": labels}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Запуск сервера FastAPI на порту 8000...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
