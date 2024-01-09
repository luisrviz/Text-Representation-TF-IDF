from crear_dataframe import create_dataset, read_files_per_documents
from preproceso_intermedio import clean_text
from representación_texto import representation_tf, dimensionality_reduction_lsi, representation_tf_idf, \
     representation_binary_weigth, representation_log_tf
from clustering import clustering_bisecting_kmeans, clustering_kmeans, clustering_hierarchical
from evaluate import fit_and_evaluate

if __name__ == "__main__":
     # Leemos los datos y creamos el dataset
     id_10_clusters, url, size, date, time, dataset, only_text = read_files_per_documents()
     df = create_dataset(id_10_clusters, url, size, date, time, dataset, only_text)

     # Tenemos que seleccionar el número de clústeres deseados y el algoritmo a emplear
     num_clusters = 10
     labels = df.id_10_clusters
     algorithm = clustering_hierarchical

     # Realizamos el preproceso del texto
     df_clean = clean_text(df)

     # Guardamos el conjunto de datos preprocesado pues lo utilizaremos más adelante
     df.to_csv("../dataset_preproceso.csv")

     # Dado que nuestra "variable" de interés es la representación vemos los resultados para cada tipo
     for modelo_representacion in [representation_tf_idf, representation_tf, representation_binary_weigth,
               representation_log_tf]:
     # Realizamos la representación del texto
          training_text_features = modelo_representacion(df_clean)

          # Realizamos la reducción de dimensionalidad
          training_text_features_reduced = dimensionality_reduction_lsi(training_text_features, 800)

          # Hacemos el clustering
          results, dbscan = algorithm(training_text_features_reduced, num_clusters=num_clusters)

          # Evaluamos los resultados
          evaluation, evaluations_std = fit_and_evaluate(dbscan, training_text_features_reduced, labels)
