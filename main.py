import csv
from oop import LinearRegression
import argparse
from oop import Normalization, SplitValidation

# Cara pakai python main.py -p "Normalization, dll" -v SplitValidation -d insurance.csv
# Dari dion :D

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='preprocessing', action='store', type=str)
parser.add_argument('-v', dest='validation', action='store', type=str)
parser.add_argument('-d', dest='data', action='store', type=str)

args = parser.parse_args()
preprocessing = [ eval(preprocessor) for preprocessor in args.preprocessing.split(", ") ]
validation = eval(args.validation)
data = args.data

if __name__ == '__main__':
  dataset = load_data()
  
  linear_regression = LinearRegression()

  linear_regression.training(dataset)

  print(linear_regression.b0, linear_regression.b1)

  print(linear_regression.predict([10, 19, 123, 19]))

  # python3 main.py -v split_val -d data.csv