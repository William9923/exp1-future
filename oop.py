import csv
from copy import copy

from random import randrange

# Asumsi data indeks 0 : X, data indeks 1 : y

class Supervised():
    def training(self, train):
        pass
    def predict(self, test):
        pass


class LinearRegression(Supervised):
   
  '''
      f(x) = w1*x +w0 
  '''
  #Calculate the mean value of a list of numbers
  
  # Utility Function

  def __init__(self):
    self.b0 = None
    self.b1 = None 

  def mean(values):
    return sum(values) / float(len(values))

  # Calculate covariance between x and y
  def covariance(x, mean_x, y, mean_y):
    cov = 0.0
    for i in range(len(x)):
      cov += (x[i] - mean_x) * (y[i] - mean_y)
    return cov

  # Calculate the variance of a list of numbers
  def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

  # Calculate coefficients 
  def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = LinearRegression.mean(x), LinearRegression.mean(y)
    b1 = LinearRegression.covariance(x, x_mean, y, y_mean) / LinearRegression.variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]

  def training(self, train):
    b0, b1 = LinearRegression.coefficients(train)
    self.b0 = b0
    self.b1 = b1
    return self 


  def predict(self, test):

    if self.b0 == None or self.b1 == None:
      raise ValueError("Model had not been trained")
    
    result_set = list()
    for row in test:
      result_set.append(self.b0 + self.b1 * row)
    return result_set

class Preprocessing():
    def run(self, dataset):
        pass

# implementation
class Normalization(Preprocessing):

    def find_min_max(self, dataset):
      col_values = [row[0] for row in dataset]
      value_min = min(col_values)
      value_max = max(col_values)
      return [value_min, value_max]

    def run(self, dataset):
        return self.normalize_dataset(dataset)

    def normalize_dataset(self, dataset) :
      value_min, value_max = self.find_min_max(dataset)
      normalize_dataset = copy(dataset)
      range_val = value_max-value_min
      for i in range(len(dataset)):
        normalize_dataset[i][0] = (dataset[i][0] - value_min) / range_val
      return normalize_dataset 

# implementation
class Standarization(Preprocessing):
  pass

# Abstract class
class Validation():

    def __init__(self, metric):
      self.metric = metric
    # virtual function
    def validate(self, method, model, dataset):
        pass 

class SplitValidation(Validation):

  def __init__(self, metric):
    super().__init__(metric)

  def train_test_split(self, dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
      index = randrange(len(dataset_copy))
      train.append(dataset_copy.pop(index))

    test = list(dataset_copy)
    return train, test
  
  def validate(self, method, dataset):
    test_set = list()
    train, test = self.train_test_split(dataset, 0.75)
    model = method.training(train)
    for row in test:
      dup_row = list(row)
      test_set.append(dup_row[0])
    
    prediction = model.predict(test_set)
    validation_score = self.metric.run([row[1] for row in test], prediction)
    return validation_score

class Metric():
  def run(self, actual, predicted):
    pass 

class MSEMetric(Metric):
  def run(self, actual, predicted):
    sum_error = 0.0

    if len(actual) != len(predicted):
      raise ValueError("Not same length")

    for i in range(len(actual)):
      prediction_error = predicted[i] - actual[i]
      sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return mean_error


class PipelineSupervied():

    def __init__(self, file = '', preprocessings = list(), method = None, validation = None):
        self.list_prep = preprocessings
        self.method = method
        self.validation = validation
        self.file = file

    # Load a CSV file
    def load_data(self, file = ''):
        dataset = []
        with open(file, 'r') as csv_file:
          csv_reader = csv.reader(csv_file, delimiter=',')
          for row in csv_reader:
            if not row:
              continue
            dataset.append([float(row[0]), float(row[1])])
        return dataset

    def preprocessing(self, dataset):
        temp = dataset
        for prep in self.list_prep:
            temp = prep.run(temp)
        return temp

    def run(self):
        # retrieve data
        dataset = self.load_data(self.file) 
        # preprocess data
        post_preprocess = self.preprocessing(dataset)

        # generate model
        model = self.train(post_preprocess)

        print(f"Validation Score: {self.validation.validate(self.method, post_preprocess)}")
        return model

    def train(self, dataset):
        return self.method.training(dataset)

    # def validation(self, method, model, dataset):
    #     return self.validation.validate(method, model, dataset)

if __name__ == '__main__':
    linear_regression = LinearRegression()
    split_validation = SplitValidation(MSEMetric())

    pipe = PipelineSupervied('insurance.csv', [Normalization()], linear_regression, split_validation)
    model = pipe.run()

    print(model.b0, model.b1)


