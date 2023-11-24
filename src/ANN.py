
import numpy as np
from src.functions import activation_functions, loss_functions
from src.utils import count_true
#Activation functions


#class for  encapsulation of  the layers
class create_layer:
    def __init__(self, function, n_perceptrons) -> None:
        #number of perceptrons
        self.n_perceptrons = n_perceptrons
        #function activation
        self.function = function

# perceptron class
class Perceptron:
    
    def __init__(self, n_weights: int, act_function: callable):
        #initialize the weights random
        self.W = np.random.rand(n_weights)
        #bias
        self.b = 1
        #function activation
        self.act_function = act_function

    #calculates the perceptron  
    def input(self, X):
        #in case the activation function is softmax
        if self.act_function == activation_functions["softmax"]:
            return np.dot(X, self.W) + self.b
        
        self.output = self.act_function(np.dot(X, self.W) + self.b)
        return self.output
    
    def print_wb(self):
        # print (f"ID({self.id})\n|\t\t\tW:  {self.W.shape}")
        print (f"\t\tW:  {self.W}")
        print (f"\t\tb:  {self.b}")

    def print(self):
        print (f"\tID({self.id})\n\t|\tW shape: {self.W.shape}\n\t|\toutput: {self.output}")
        self.print_wb()
        print ("\t\t-")


# layer class
class Layer:
    
    def __init__(self, function: str, batch_size: int,n_perceptrons: int, n_inputs: int):
        #activation function
        self.function = function
        #number of perceptrons
        self.n_perceptrons = n_perceptrons
        #number of inputs recived
        self.n_inputs = n_inputs
        #batch size
        self.batch_size = batch_size
        #perceptrons
        self.perceptrons = [Perceptron(n_weights=n_inputs, act_function=activation_functions[self.function]) 
                            for perceptron in range(n_perceptrons)]

     #returns the matrix of results of the perceptrons       
    def forward_pass(self, X):
        #initialize the matrix of the perceptrons
        output = np.empty((self.n_perceptrons, self.batch_size))
       #calculates the matrix of the perceptrons
        index = 0
        for perceptron in self.perceptrons:
            out = perceptron.input(X)
            output[index] = out
            index += 1
        # in case of softmax, return the output after compute in a softmax
        if self.function == "softmax":
            return activation_functions["softmax"](output.T)
        #returns the output
        return output.T


    def print_perceptrons(self):
        for perceptron in self.perceptrons:
            perceptron.print()
    def print(self):
        print(f"ID({self.id})\n|\tactivation_function: {self.function}\t\n|\tperceptrons: {self.n_perceptrons}\t")
        self.print_perceptrons()

#ANN class
class ANN:
    
    def __init__(self,  layers: list, Xdata, Ydata, cost_fn:str = "mse", batch = None):
        #number of layers
        self.n_layers :int = len(layers)
        #data
        self.X = Xdata
        #prediction
        self.Y = Ydata
        #index of the samples, to know in which batch is the model
        self.index_samples : int = 0
        #variable that checks if the batch has to be reset to 0
        self.finished_batch : bool = False
        #batch size
        self.batch = batch if batch else self.X.shape[0]
        #output
        self.output :float = None
        #cost
        self.cost = None
        #accuracity
        self.accuracy = 0
        #layers
        self.layers = self.create_layers(layers)
        # output function
        self.output_function = layers[-1].function
        # cost function
        self.cost_fn = cost_fn
        # loss function
        self.loss_functions = loss_functions
        # dict of to transform the output in a binary format
        self.parse_output_functions = {
                'sigmoid': lambda o: o > 0.5,
                'softmax': lambda o: np.argmax(o, axis=1),
                'tanh': lambda o: o,
                'relu': lambda o: o
        }
    
    # creates the layers for the ANN
    def create_layers(self, layers_info: list):
        
        #input layer
        layers = [Layer(function=layers_info[0].function, n_perceptrons=layers_info[0].n_perceptrons, n_inputs=self.X.shape[1], batch_size=self.batch)]
        
        for i in range(1, len(layers_info)):
            layers.append(Layer(function=layers_info[i].function, n_perceptrons=layers_info[i].n_perceptrons, n_inputs= layers_info[i- 1].n_perceptrons, batch_size=self.batch))
        
        return layers

    # gets the accuracy 
    def get_accuracy(self, printable=False):
        #refresh the samples
        self.take_samples()
        #runs the ANN
        self.forward_pass()
        #calculates how many values has been predicted
        correct_predictions = count_true(y_pred=self.parse_output_functions[self.output_function](self.output), y_true=self.samples_y)

        accuracy = correct_predictions / len(self.samples_y)
        return_value = (f"{correct_predictions}/{len(self.samples_y)}", accuracy) if printable else accuracy
        return return_value

    #takes the corresponding samples for the batch
    def take_samples(self):
        # when there is not more samples to take
        if (self.index_samples + self.batch) > self.X.shape[0]:
            self.finished_batch = True
            self.index_samples = 0
            return False
        #samples of the data
        self.samples = self.X[self.index_samples: self.index_samples + self.batch]
        #samples of the prediction
        self.samples_y = self.Y[self.index_samples: self.index_samples + self.batch]
        #refresh the index of the samples
        self.index_samples += self.batch
        return True
    
    #calculates the output of the ANN
    def forward_pass(self):
        #calculates the input layer
        input_layer = self.layers[0]
        x = input_layer.forward_pass(X=self.samples)
        #calculates the hidden layers
        for layer in self.layers[1:]:
                x = layer.forward_pass(X=x)
        #save the output 
        self.output = x
    
    
    def get_cost(self, output = None):
        """Mean squared error derivative on the output of the last layer

        Returns:
            np.array: matrix with the cost of the output
        """
        
        return self.loss_functions[self.cost_fn](y_true=self.samples_y, y_pred=self.output)

    #returns the total particles in the ANN
    def get_total_parameters(self):
        n_parameters = 0
        
        for layer in self.layers:
            n_parameters += (layer.n_inputs * layer.n_perceptrons) + layer.n_perceptrons # total weights per layer + total biases
    
        return n_parameters
    
    #fill the weights with the particles
    def fill_weights(self, particle):

        p_copy = np.copy(particle)

        for l in range(len(self.layers)):
            for p in range(len(self.layers[l].perceptrons)):
                for w in range(len(self.layers[l].perceptrons[p].W)):
                    self.layers[l].perceptrons[p].W[w] = p_copy[0]
                    p_copy = p_copy[1:]
                self.layers[l].perceptrons[p].b = p_copy[0]
                p_copy = p_copy[1:]    

    def print (self):
        print (f"Layers ({len(self.layers)}) :")
        for layer in self.layers:
            layer.print()
            print ("\t------")
        print (f"Inputs: {self.X[1]}")
