import pandas as pd
import os


def read_files_per_documents():
    all_files = os.listdir("../Dataset/")
    only_text = []
    id_10_clusters = []
    url = []
    size = []
    date = []
    time = []
    dataset = []
    for file in all_files:
        try:
            f = open(f"../Dataset/{file}", "r")
            text = f.read().split("\n")
            id_10_clusters.append(text[0].split("=")[1][0])
            url.append(text[1].split("=")[1])
            size.append(text[2].split("=")[1])
            date.append(text[3].split("=")[1])
            time.append(text[4].split("=")[1])
            dataset.append(text[5].split("=")[1])
            flat_text = ""
            # Como lo que hay de aqui en adelante es un html podríamos simplemente coger lo que dentro de la etiqueta
            # raiz <html>
            for i in text[6:]:
                flat_text = flat_text + i
            " ".join(flat_text.split())
            only_text.append(flat_text)
            only_text = only_text.strip()
            f.close()
        except:
            pass
    return id_10_clusters, url, size, date, time, dataset, only_text


id_10_clusters, url, size, date, time, dataset, only_text = read_files_per_documents()


def create_dataset(id_10_clusters, url, size, date, time, dataset, only_text):
    #  Con esto ya metemos todos los datos en un dataframe de Pandas
    df = pd.DataFrame([id_10_clusters, url, size, date, time, dataset, only_text]).transpose()
    df.columns = ['id_10_clusters', 'url', 'size', 'date', 'time', 'dataset', 'only_text']
    map_dict = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "J": 10}
    df["id_10_clusters"].replace(map_dict, inplace=True)
    map_dict = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 4, 10: 4}
    df["id_4_clusters"] = df["id_10_clusters"].replace(map_dict)
    return df


df = create_dataset(id_10_clusters, url, size, date, time, dataset, only_text)
df.to_csv("../dataset_crudo.csv")
