import whiterabbit.neural_network as m

n = m.NeuralNetwork.random()
print(hash(n))
n.save("dev.npz")
e = m.NeuralNetwork.load("dev.npz")
print(hash(e))
