import argparse
from oop import Normalization

parser = argparse.ArgumentParser()
parser.add_argument('-p', dest='preprocessing', action='store', type=str)
parser.add_argument('-v', dest='validation', action='store', type=str)
parser.add_argument('-d', dest='data', action='store', type=str)

args = parser.parse_args()
preprocessing = [ eval(preprocessor) for preprocessor in args.preprocessing.split(", ") ]
validation = eval(args.validation)
data = args.data

print(preprocessing)
print(validation)
print(data)