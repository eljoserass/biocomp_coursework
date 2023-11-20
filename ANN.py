

def sigmoid(x):
        import numpy as np
        return 1 / (1 + np.exp(-x))
    
def dsigmoid(x):
        return sigmoid(x=x) * (1 - sigmoid(x=x))
    
def relu(x):
        import numpy as np
        return np.maximum(0.0, x)
    
def drelu(x):
        import numpy as np
        return np.where(x > 0, 1, 0)
    
def tanh(x):
        import numpy as np
        return np.tanh(x)
    
def dtanh(x):
        import numpy as np
        return 1 - np.square(np.tanh(x)) 


activation_functions: dict = {"sigmoid": sigmoid,
                              "dsigmoid": dsigmoid,
                              "relu": relu,
                              "drelu": drelu,
                              "tanh": tanh,
                              "dtanh": dtanh
                              }

class create_layer:
    def __init__(self, function, n_perceptrons) -> None:
        self.n_perceptrons = n_perceptrons
        self.function = function


class Layer:
    
    def __init__(self, function: str, batch_size: int,n_perceptrons: int, n_inputs: int, id: int):
        
        self.id = id
        self.function = function
        self.n_perceptrons = n_perceptrons
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.perceptrons = [Perceptron(n_weights=n_inputs, id=perceptron, act_function=activation_functions[self.function]) 
                            for perceptron in range(n_perceptrons)]
        
    def print_perceptrons(self):
        for perceptron in self.perceptrons:
            perceptron.print()
            
    def forward_pass(self, X):
        import numpy as np
        output = np.empty((self.n_perceptrons, self.batch_size))
        index = 0
        for perceptron in self.perceptrons:
            out = perceptron.input(X)
            output[index] = out
            index += 1
        return output.T
    
    def print(self):
        print(f"ID({self.id})\n|\tactivation_function: {self.function}\t\n|\tperceptrons: {self.n_perceptrons}\t")
        self.print_perceptrons()
    
class Perceptron:
    
    def __init__(self, n_weights: int, act_function: callable, id: int):
        
        import numpy as np
        self.id = id
        self.W = np.random.rand(n_weights)
        self.b = 1
        self.act_function = act_function
        self.output = 0
        
    def input(self, X):
        import numpy as np
        # print (f"X.shape {X.shape}  W.shape {self.W.shape}  b {self.b}")
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
    
        

class ANN:
    
    def __init__(self,  layers: list, Xdata, Ydata):
        
        self.inputs = Xdata[1]
        self.n_layers = len(layers)
        self.X = Xdata
        self.Y = Ydata
        self.batch = self.X.shape[0]
        self.output :float = None
        self.cost = None
        self.accuracy = 0
        self.layers = self.create_layers(layers)
    
    def create_layers(self, layers_info: list):
        layers = [Layer(function=layers_info[0].function, n_perceptrons=layers_info[0].n_perceptrons, n_inputs=self.X.shape[1], id=0, batch_size=self.batch)]
        
        for i in range(1, len(layers_info)):
            layers.append(Layer(function=layers_info[i].function, n_perceptrons=layers_info[i].n_perceptrons, n_inputs= layers_info[i- 1].n_perceptrons, id=i, batch_size=self.batch))
        
        return layers
    def get_accuracy(self, validate_Y = None):
        import numpy as np
        threshold = 0.86
        output_pred = (self.output > threshold).astype(int)
        correct_predictions = np.sum(np.argmax(output_pred, axis=1) == np.argmax(self.Y, axis=1))
        
        return correct_predictions / len(self.Y)
        
    def forward_pass(self):
        import numpy as np
        
        output = np.empty_like(self.X.shape)
        for layer in self.layers:
            # print (f"ID({layer.id})")
            if layer.id == 0:
                output = layer.forward_pass(X=self.X)
            else:
                output = layer.forward_pass(X=output)
            # print (f"output.shape {output.shape}")
            # print("-----------")
        self.output = output
        print(output)
        self.cost = self.get_cost()
    
    
    def get_cost(self):
        """Mean squared error derivative on the output of the last layer

        Returns:
            np.array: matrix with the cost of the output
        """
        import numpy as np

        return np.mean(self.Y * np.log(self.output)  + (1 - self.output) * np.log(1 - self.output)) / self.batch

    def get_total_parameters(self):
        n_parameters = 0
        
        for layer in self.layers:
            n_parameters += (layer.n_inputs * layer.n_perceptrons) + layer.n_perceptrons # total weights per layer + total biases
    
        return n_parameters
    
    def fill_weights(self, particle):
        import numpy as np
        p_copy = np.copy(particle)

        for l in range(len(self.layers)):
            for p in range(len(self.layers[l].perceptrons)):
                for w in range(len(self.layers[l].perceptrons[p].W)):
                    self.layers[l].perceptrons[p].W[w] = p_copy[0]
                    p_copy = p_copy[1:]
                self.layers[l].perceptrons[p].b = p_copy[0]
                p_copy = p_copy[1:]
            

    def gradient(self, output):
        import numpy as np
        self.grad = np.dot(output.T, self.cost) / self.Y.shape[0]
        return self.grad
    
    def print (self):
        print (f"Layers ({len(self.layers)}) :")
        for layer in self.layers:
            layer.print()
            print ("\t------")
        print (f"Inputs: {self.inputs}")
