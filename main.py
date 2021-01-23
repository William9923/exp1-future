import csv
from oop import LinearRegression, Normalization, SplitValidation, MSEMetric, PipelineSupervied
import argparse

# Cara pakai python main.py -p "Normalization, dll" -v SplitValidation -d insurance.csv
# Dari dion :D

def factory_method(method):
  return {
    'linear' : LinearRegression()
  }.get(method)

def factory_preprocessor(preprocessor):
  return {
    'normalization' : Normalization()
  }.get(preprocessor)

def factory_validation(validation, metric):
  return {
    'split_validation' : SplitValidation(metric)
  }.get(validation)

if __name__ == '__main__':

  # Argument Parser
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', dest='method', action='store', type=str)
  parser.add_argument('-p', dest='preprocessing', action='store', type=str)
  parser.add_argument('-v', dest='validation', action='store', type=str)
  parser.add_argument('-d', dest='data', action='store', type=str)

  args = parser.parse_args()

  method = factory_method(args.method)
  preprocessing = [ factory_preprocessor(preprocessor) for preprocessor in args.preprocessing.split(", ") ]
  validation = factory_validation(args.validation, MSEMetric())
  
  data = args.data.strip()

  # method = eval(args.method)
  # preprocessing = [ eval(preprocessor) for preprocessor in args.preprocessing.split(", ") ]
  # validation = eval(args.validation)
  # data = args.data
  
  pipe = PipelineSupervied(data, preprocessing, method, validation)

  # print(preprocessing)
  # print(validation)

  model = pipe.run()

  print(model.b0, model.b1)

  # linear_regression = LinearRegression()

  # linear_regression.training(dataset)

  # print(linear_regression.b0, linear_regression.b1)

  # print(linear_regression.predict([10, 19, 123, 19]))

  # python3 main.py -v split_val -d data.csv