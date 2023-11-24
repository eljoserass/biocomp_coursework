

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


def softmax(x):
    import numpy as np
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Subtracting max for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
def dtanh(x):
    import numpy as np
    return 1 - np.square(np.tanh(x))


activation_functions: dict = {"sigmoid": sigmoid,
                              "dsigmoid": dsigmoid,
                              "relu": relu,
                              "drelu": drelu,
                              "tanh": tanh,
                              "dtanh": dtanh,
                              "softmax": softmax
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
        if self.function == "softmax":
            return activation_functions["softmax"](output.T)
        return output.T
    
    def print(self):
        print(f"ID({self.id})\n|\tactivation_function: {self.function}\t\n|\tperceptrons: {self.n_perceptrons}\t")
        self.print_perceptrons()
    
class Perceptron:
    
    def __init__(self, n_weights: int, act_function: callable, id: int):
        
        import numpy as np
        import decimal
        self.id = id
        self.W = np.random.rand(n_weights)
        self.W
        self.b = 1
        self.act_function = act_function
        self.output = 0
        
    def input(self, X):
        import numpy as np
        # print (f"X.shape {X.shape}  W.shape {self.W.shape}  b {self.b}")
        if self.act_function == softmax:
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
    
def mean_squared_error(y_true, y_pred):
    import numpy as np
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred):
    import numpy as np
    epsilon = 1e-15 
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def count_true(y_true, y_pred):
    correct_predictions = 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            correct_predictions += 1
    return correct_predictions


class ANN:
    
    def __init__(self,  layers: list, Xdata, Ydata, cost_fn:str = "mse", batch = None):
        self.inputs = Xdata[1]
        import numpy as np
        self.n_layers = len(layers)
        self.X = Xdata
        self.Y = Ydata
        self.index_samples = 0
        self.finished_batch = False
        if not batch:
            self.batch = self.X.shape[0]
        else: 
            self.batch = batch    
        self.output :float = None
        self.cost = None
        self.accuracy = 0
        self.layers = self.create_layers(layers)
        self.output_function = layers[-1].function
        self.cost_fn = cost_fn
        self.cost_functions = {
                'mse': lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
                'cross-entropy': lambda y_true, y_pred: binary_cross_entropy(y_true, y_pred)
            }
        self.parse_output_functions = {
                'sigmoid': lambda o: o > 0.5,
                'softmax': lambda o: np.argmax(o, axis=1),
                'tanh': lambda o: o,
                'relu': lambda o: o
        }
    
    def create_layers(self, layers_info: list):
        layers = [Layer(function=layers_info[0].function, n_perceptrons=layers_info[0].n_perceptrons, n_inputs=self.X.shape[1], id=0, batch_size=self.batch)]
        
        for i in range(1, len(layers_info)):
            layers.append(Layer(function=layers_info[i].function, n_perceptrons=layers_info[i].n_perceptrons, n_inputs= layers_info[i- 1].n_perceptrons, id=i, batch_size=self.batch))
        
        return layers

    def get_accuracy(self, printable=False):
        self.take_samples()
        self.forward_pass()
        correct_predictions = count_true(y_pred=self.parse_output_functions[self.output_function](self.output), y_true=self.samples_y)

        accuracy = correct_predictions / len(self.samples_y)
        return_value = (f"{correct_predictions}/{len(self.samples_y)}", accuracy) if printable else accuracy
        return return_value
    
    def get_accuracy_old(self, printable=False):
        # TODO change to be able to support all activatoin functions
        def count_sigmoid():
            correct_predictions = 0
            for i in range(len(self.Y)):
                self.take_samples()
                self.forward_pass()
                if self.output[i] > 0.5:
                    output_pred = 1
                else:
                    output_pred = 0
                if output_pred == self.Y[i]:
                    correct_predictions += 1
            if printable:
                print (f"correct_predictions {correct_predictions}/{len(self.Y)}")
            return correct_predictions
        
        def count_softmax():
            import numpy as np
            correct_predictions = 0
            for i in range(len(self.samples_y)):
                if np.argmax(self.output[i]) == self.samples_y[i]:
                    correct_predictions += 1  
            if printable:  
                print (f"correct_predictions {correct_predictions}/{len(self.samples_y)}")
            return correct_predictions

        correct_predictions = 0
        #take samples y
        self.take_samples()
        #take output
        self.forward_pass()
        if self.output_function == "sigmoid":
            correct_predictions =  count_sigmoid()
        if self.output_function == "softmax":
            correct_predictions = count_softmax()
        accuracy = correct_predictions / len(self.samples_y)
        return_value = (f"{correct_predictions}/{len(self.samples_y)}", accuracy) if printable else accuracy
        return return_value
    
    
    def take_samples(self):
        if (self.index_samples + self.batch) > self.X.shape[0]:
            self.finished_batch = True
            self.index_samples = 0
            return False
        self.samples = self.X[self.index_samples: self.index_samples + self.batch]
        self.samples_y = self.Y[self.index_samples: self.index_samples + self.batch]
        self.index_samples += self.batch
        return True
    
    def forward_pass(self):
        import numpy as np
        output = np.empty_like(self.samples)
        for layer in self.layers:
            if layer.id == 0:
                output = layer.forward_pass(X=self.samples)
            else:
                output = layer.forward_pass(X=output)
        self.output = output
    
    
    def get_cost(self):
        """Mean squared error derivative on the output of the last layer

        Returns:
            np.array: matrix with the cost of the output
        """
        
        return self.cost_functions[self.cost_fn](y_true=self.samples_y, y_pred=self.output)

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
