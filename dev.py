import numpy as np
import whiterabbit.neural_network as m

n = m.NeuralNetwork.random()
print(list(n))
n.save("dev.npz")
input()
e = m.NeuralNetwork.load("dev.npz")
print(list(e))
for mat in list(n):
    print(type(mat), mat.dtype)
for mat in list(e):
    print(type(mat), mat.dtype)
