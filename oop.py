class Supervised(ML):

    def training(self):
        pass
    def predict(self):
        pass

class Backpro(Supervised):
    pass

class SVM(Supervised):
    pass

class IDTree(Supervised):
    pass

class LinearRegression(Supervised):
   
    '''
        f(x) = w1*x +w0 
    '''
    def training(self):
        pass
    def predict(self, model):
        pass



class Preprocessing():
    def run(self, dataset):
        pass


class Normalization(Preprocessing):

    def run(self, dataset):
        return dataset

class Validation():

    def validate(self):
        pass

class SplitValidation(Validation):

    def validate(self):
        pass

class Model():
    pass

class WeightsModel(Model):
    pass

class PipelineSupervied():

    def __init__(self, file = '', preprocessings = list(), method = None, validation = None):
        self.list_prep = preprocessings
        self.method = method
        self.validation = validation
        self.file = file

    def load_data(self, file = ''):
        pass

    def run(self):
        # retrieve data
        dataset = self.load_data(self.file) 
        # preprocess data
        post_preprocess = self.preprocessing(dataset)

        # generate model
        model = self.method(post_preprocess)

        self.validation(method, model, post_preprocess)
        return model

    def preprocessing(self, dataset):
        temp = dataset
        for prep in self.list_prep:
            temp = prep.run(temp)

        return temp

    def method(self, dataset):
        return self.method.training(dataset)

    def validation(self, method, model, dataset):
        pass

if __name__ == '__main__':
    pipe = PipelineSupervied('housing.csv', [Normalization()], LinearRegression(), SplitValidation())
    model = pipe.run()
