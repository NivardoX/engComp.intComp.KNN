import numpy as np
import pandas as pd
from statistics import mode

def euclidean_distance(data_1, data_2):
  dist = 0
  for i in range(len(data_1)):
      dist = dist + np.square(data_1[i] - data_2[i])
  return np.sqrt(dist)

class Knn:
  def __init__(self,train_data, K=7):
    self.k = K
    self.train_data_attr = train_data[0]
    self.train_data_classes = train_data[1].to_numpy()


  def predict(self,unclassified_data):
    distances = []
    classes = []

    for i in range(self.train_data_attr.shape[0]):
      distances.append(euclidean_distance(unclassified_data,self.train_data_attr.iloc[[i]].to_numpy()[0]))
      classes.append(self.train_data_classes[i][0])

    sorted_indexes = np.argsort(distances)
    sorted_classes = np.array(classes)[sorted_indexes]


    return mode(sorted_classes[:self.k])
    





if __name__ == '__main__':
  iris_dataset =pd.read_csv("iris_dataset.csv")
  # SHUFFLE DF
  iris_dataset = iris_dataset.sample(frac=1).reset_index(drop=True)

  iris_dataset_test = iris_dataset.iloc[-10:,:]
  iris_dataset_train = iris_dataset.iloc[:-10,:]


  iris_attributes = iris_dataset_train[["sepal.length","sepal.width","petal.length","petal.width"]]
  iris_attributes_test = iris_dataset_test[["sepal.length","sepal.width","petal.length","petal.width"]]

  iris_classes = iris_dataset_train[["variety"]]
  iris_classes_test = iris_dataset_test[["variety"]].to_numpy()


  knn = Knn((iris_attributes,iris_classes))
  
  for test_data_idx in range(iris_attributes_test.shape[0]):

    predicted = knn.predict(iris_attributes_test.iloc[[test_data_idx]].to_numpy()[0])
    print(iris_attributes_test.iloc[[test_data_idx]].to_numpy()[0])
    print(f"Predicted {predicted}, Expected {iris_classes_test[test_data_idx][0]} ")