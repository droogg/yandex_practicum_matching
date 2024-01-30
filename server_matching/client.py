import requests
import numpy as np
import pandas as pd

# URL сервера FastAPI
url = "http://127.0.0.1:8000/find_neighbors/"

data = pd.read_csv("data/train.csv", index_col="Id")
# Подготовка образца данных
sample = data.sample(n=1)
try:
    sample = sample.drop("Target", axis=1)
except:
    pass
sample_list = sample.values.tolist()

sample_data = {
    "sample": sample_list
}
# Отправка запроса на сервер
response = requests.post(url, json=sample_data)

if response.status_code == 200:
    neighbors = response.json()["neighbors"]
    print("Ближайшие соседи:", neighbors)
else:
    print("Ошибка:", response.status_code, response.text)
