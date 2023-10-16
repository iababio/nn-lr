# -*- coding: utf-8 -*-
"""ML-HW1.ipynb

Author: @Innocent Boakye Ababio

Colab file is located at
    https://colab.research.google.com/drive/1A1gB_oj5f4g6yT7YEF6L2hfKfK-_nOcB

1. Title: Pima Indians Diabetes Database

2. Sources:
   (a) Original owners: National Institute of Diabetes and Digestive and
                        Kidney Diseases
   (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
                          Research Center, RMI Group Leader
                          Applied Physics Laboratory
                          The Johns Hopkins University
                          Johns Hopkins Road
                          Laurel, MD 20707
                          (301) 953-6231
   (c) Date received: 9 May 1990

3. Past Usage:
    1. Smith,~J.~W., Everhart,~J.~E., Dickson,~W.~C., Knowler,~W.~C., \&
       Johannes,~R.~S. (1988). Using the ADAP learning algorithm to forecast
       the onset of diabetes mellitus.  In {\it Proceedings of the Symposium
       on Computer Applications and Medical Care} (pp. 261--265).  IEEE
       Computer Society Press.

       The diagnostic, binary-valued variable investigated is whether the
       patient shows signs of diabetes according to World Health Organization
       criteria (i.e., if the 2 hour post-load plasma glucose was at least
       200 mg/dl at any survey  examination or if found during routine medical
       care).   The population lives near Phoenix, Arizona, USA.

       Results: Their ADAP algorithm makes a real-valued prediction between
       0 and 1.  This was transformed into a binary decision using a cutoff of
       0.448.  Using 576 training instances, the sensitivity and specificity
       of their algorithm was 76% on the remaining 192 instances.

4. Relevant Information:
      Several constraints were placed on the selection of these instances from
      a larger database.  In particular, all patients here are females at
      least 21 years old of Pima Indian heritage.  ADAP is an adaptive learning
      routine that generates and executes digital analogs of perceptron-like
      devices.  It is a unique algorithm; see the paper for details.

5. Number of Instances: 768

6. Number of Attributes: 8 plus class

7. For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

8. Missing Attribute Values: Yes

9. Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268

10. Brief statistical analysis:

| Attribute number | Mean | Standard Deviation  |
|-------------------|------|---------------------|
| 1.                | 3.8  | 3.4                 |
| 2.                | 120.9| 32.0                |
| 3.                | 69.1 | 19.4                |
| 4.                | 20.5 | 16.0                |
| 5.                | 79.8 | 115.2               |
| 6.                | 32.0 | 7.9                 |
| 7.                | 0.5  | 0.3                 |
| 8.                | 33.2 | 11.8                |
"""

# Commented out IPython magic to ensure Python compatibility.
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import requests


# %matplotlib inline


# class Value:
#     """ stores a single scalar value and its gradient """
#
#     def __init__(self, data, _children=(), _op=''):
#         self.data = data
#         self.grad = 0
#
#         # internal variables used for autograd graph construction
#         self._backward = lambda: None
#         self._prev = set(_children)
#         self._op = _op # the op that produced this node, for graphviz / debugging / etc
#         self.m = np.zeros_like(data)
#         self.v = np.zeros_like(data)
#
#     def __add__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data + other.data, (self, other), '+')
#
#         def _backward():
#             self.grad += out.grad
#             other.grad += out.grad
#         out._backward = _backward
#
#         return out
#
#     def __mul__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         out = Value(self.data * other.data, (self, other), '*')
#
#         def _backward():
#             self.grad += other.data * out.grad
#             other.grad += self.data * out.grad
#         out._backward = _backward
#
#         return out
#
#     def __pow__(self, other):
#         assert isinstance(other, (int, float)), "only supporting int/float powers for now"
#         out = Value(self.data**other, (self,), f'**{other}')
#
#         def _backward():
#             self.grad += (other * self.data**(other-1)) * out.grad
#         out._backward = _backward
#
#         return out
#
#     def relu(self):
#         out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
#
#         def _backward():
#             self.grad += (out.data > 0) * out.grad
#         out._backward = _backward
#
#         return out
#
#
#     def backward(self):
#
#         # topological order all of the children in the graph
#         topo = []
#         visited = set()
#         def build_topo(v):
#             if v not in visited:
#                 visited.add(v)
#                 for child in v._prev:
#                     build_topo(child)
#                 topo.append(v)
#         build_topo(self)
#
#
#         # go one variable at a time and apply the chain rule to get its gradient
#         self.grad = 1
#         for v in reversed(topo):
#             v._backward()
#
#     def __neg__(self): # -self
#         return self * -1
#
#     def __radd__(self, other): # other + self
#         return self + other
#
#     def __sub__(self, other): # self - other
#         return self + (-other)
#
#     def __rsub__(self, other): # other - self
#         return other + (-self)
#
#     def __rmul__(self, other): # other * self
#         return self * other
#
#     def __truediv__(self, other): # self / other
#         return self * other**-1
#
#     def __rtruediv__(self, other): # other / self
#         return other * self**-1
#
#     def __repr__(self):
#         return f"Value(data={self.data}, grad={self.grad})"


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        # Initialize the Value instance with data, gradient, and autograd-related variables.
        self.data = data
        self.grad = 0

        # Internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc
        self.m = np.zeros_like(data)  # Initialize the 'm' variable for Adam optimizer.
        self.v = np.zeros_like(data)  # Initialize the 'v' variable for Adam optimizer.

    def __add__(self, other):
        # Overloaded addition operator to support Value + Value or Value + scalar
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # Chain rule: derivative of the sum is 1 for both operands.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other):
        # Overloaded multiplication operator to support Value * Value or Value * scalar
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # Chain rule for multiplication
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        # Overloaded power operator to support Value ** Value or Value ** scalar
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # Chain rule for power operation
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def relu(self):
        # Rectified Linear Unit (ReLU) activation function
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            # Derivative of ReLU: 0 for negative input, 1 for positive input.
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        # Perform backpropagation using reverse-mode autograd

        # Topological order all the children in the graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self):  # Unary negation operator: -self
        return self * -1

    def __radd__(self, other):  # Overloaded addition for reverse order: other + self
        return self + other

    def __sub__(self, other):  # Overloaded subtraction: self - other
        return self + (-other)

    def __rsub__(self, other):  # Overloaded reverse subtraction: other - self
        return other + (-self)

    def __rmul__(self, other):  # Overloaded multiplication for reverse order: other * self
        return self * other

    def __truediv__(self, other):  # Overloaded division: self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # Overloaded reverse division: other / self
        return other * self ** -1

    def __repr__(self):
        # String representation of the Value instance for debugging purposes.
        return f"Value(data={self.data}, grad={self.grad})"


# class Module:
#
#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = 0
#
#     def parameters(self):
#         return []

class Module:  # Define a class named Module.

    def zero_grad(self):  # Method to zero out gradients for all parameters.
        for p in self.parameters():  # Iterate through the parameters of the module.
            p.grad = 0  # Set the gradient of the current parameter to zero.

    def parameters(self):  # Method to retrieve the parameters of the module.
        return []  # Return an empty list, to be overridden by subclasses.


# class Neuron(Module):
#
#     def __init__(self, nin, nonlin=True):
#         self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
#         self.b = Value(0)
#         self.nonlin = nonlin
#
#     def __call__(self, x):
#         act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
#         return act.relu() if self.nonlin else act
#
#     def parameters(self):
#         return self.w + [self.b]
#
#     def __repr__(self):
#         return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Neuron(Module):  # Define a class named Neuron that inherits from the Module class.

    def __init__(self, nin, nonlin=True):  # Constructor method to initialize a Neuron instance.
        # Initialize weights (w) with random values between -1 and 1 for each input.
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # Initialize bias (b) with a scalar value of 0.
        self.b = Value(0)
        # Boolean flag to determine whether to apply a non-linear activation function.
        self.nonlin = nonlin

    def __call__(self, x):  # Method to make the Neuron instance callable.
        # Calculate the weighted sum of inputs (x) and the bias, forming the linear activation.
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Apply ReLU activation if nonlin is True, otherwise return the linear activation.
        return act.relu() if self.nonlin else act

    def parameters(self):  # Method to retrieve the parameters of the Neuron (weights and bias).
        return self.w + [self.b]

    def __repr__(self):  # Method to represent the Neuron instance as a string.
        # Return a string indicating whether the neuron has ReLU or Linear activation and the number of weights.
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


# class Layer(Module):
#
#     def __init__(self, nin, nout, **kwargs):
#         self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
#
#     def __call__(self, x):
#         out = [n(x) for n in self.neurons]
#         return out[0] if len(out) == 1 else out
#
#     def parameters(self):
#         return [p for n in self.neurons for p in n.parameters()]
#
#     def __repr__(self):
#         return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class Layer(Module):  # Define a class named Layer that inherits from the Module class.

    def __init__(self, nin, nout, **kwargs):  # Constructor method to initialize a Layer instance.
        # Create a list of neurons using the Neuron class with specified input size and additional arguments.
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):  # Method to make the Layer instance callable.
        # Calculate the output of each neuron in the layer for the given input (x).
        out = [n(x) for n in self.neurons]
        # Return a single output if there is only one neuron, otherwise return a list of outputs.
        return out[0] if len(out) == 1 else out

    def parameters(self):  # Method to retrieve the parameters of the Layer (neuron parameters).
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):  # Method to represent the Layer instance as a string.
        # Return a string indicating the structure of the layer, including information about each neuron.
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


# class MLP(Module):
#
#     def __init__(self, nin, nouts):
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
#
#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#
#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]
#
#     def __repr__(self):
#         return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

class MLP(Module):  # Define a class named MLP that inherits from the Module class.

    def __init__(self, nin, nouts):  # Constructor method to initialize an MLP instance.
        # Define the sizes of the layers, including the input and output sizes.
        sz = [nin] + nouts
        # Create a list of layers using the Layer class with specified sizes and activation functions.
        self.layers = [Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):  # Method to make the MLP instance callable.
        # Iterate through each layer and calculate the output for the given input (x).
        for layer in self.layers:
            x = layer(x)
        # Return the final output of the MLP.
        return x

    def parameters(self):  # Method to retrieve the parameters of the MLP (layer parameters).
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):  # Method to represent the MLP instance as a string.
        # Return a string indicating the structure of the MLP, including information about each layer.
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


np.random.seed(42)
random.seed(42)

# Create a directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(parents=True, exist_ok=True)

# URL of the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'

# Path to save the downloaded dataset
file_path = data_dir / 'pima-indians-diabetes.csv'

# Download the dataset if it doesn't exist
if not file_path.is_file():
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

df = pd.read_csv('data/pima-indians-diabetes.csv')
df.head()

X_labels = df.iloc[:, :-1].values

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X_labels)

y_label = df.iloc[:, -1].values
y = y_label * 2 - 1  # make y be -1 or 1

# visualize in 2D
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y_label, s=20, cmap='jet')
# print(y_label)


# initialize a model
model = MLP(8, [16, 16, 1])  # 2-layer neural network
print(model)
print("number of parameters", len(model.parameters()))


# loss function
def loss(batch_size=None):
    # inline DataLoader :)
    if batch_size is None:
        Xb, yb = X, y
    else:
        ri = np.random.permutation(X.shape[0])[:batch_size]
        Xb, yb = X[ri], y[ri]
    inputs = [list(map(Value, xrow)) for xrow in Xb]

    # forward the model to get scores
    scores = list(map(model, inputs))

    # svm "max-margin" loss
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
    data_loss = sum(losses) * (1.0 / len(losses))
    # L2 regularization
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in model.parameters()))
    total_loss = data_loss + reg_loss

    # also get accuracy
    accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
    return total_loss, sum(accuracy) / len(accuracy)


total_loss, acc = loss()
print(total_loss, acc)

# optimization
for k in range(100):

    # forward
    total_loss, acc = loss()

    # backward
    model.zero_grad()
    total_loss.backward()

    # update (sgd)
    learning_rate = 1.0 - 0.95 * k / 100
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if k % 10 == 0:
        print(f"step {k} loss {total_loss.data}, accuracy {acc * 100}%")
