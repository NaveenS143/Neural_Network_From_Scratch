from losses import mse, mse_derivative

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, X, Y, epochs , learning_rate ):
    for e in range(epochs):
            error = 0
            for x, y in zip(X, Y):
                output = predict(network, x)

                error += mse(y, output)

                grad = mse_derivative(y, output)
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(X)
            print(f"{e + 1}/{epochs}, error={error}")