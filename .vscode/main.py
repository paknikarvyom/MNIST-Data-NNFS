# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T14:54:10.366916Z","iopub.execute_input":"2026-02-22T14:54:10.367665Z","iopub.status.idle":"2026-02-22T14:54:10.372690Z","shell.execute_reply.started":"2026-02-22T14:54:10.367632Z","shell.execute_reply":"2026-02-22T14:54:10.371691Z"},"jupyter":{"outputs_hidden":false}}
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T14:54:10.374843Z","iopub.execute_input":"2026-02-22T14:54:10.375173Z","iopub.status.idle":"2026-02-22T14:54:12.709445Z","shell.execute_reply.started":"2026-02-22T14:54:10.375149Z","shell.execute_reply":"2026-02-22T14:54:12.708091Z"},"jupyter":{"outputs_hidden":false}}
data = pd.read_csv('digit-recognizer/train.csv')

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T14:54:12.711046Z","iopub.execute_input":"2026-02-22T14:54:12.711370Z","iopub.status.idle":"2026-02-22T14:54:12.726736Z","shell.execute_reply.started":"2026-02-22T14:54:12.711343Z","shell.execute_reply":"2026-02-22T14:54:12.725560Z"},"jupyter":{"outputs_hidden":false}}
data.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T14:54:12.728372Z","iopub.execute_input":"2026-02-22T14:54:12.728773Z","iopub.status.idle":"2026-02-22T14:54:13.560497Z","shell.execute_reply.started":"2026-02-22T14:54:12.728732Z","shell.execute_reply":"2026-02-22T14:54:13.559363Z"},"jupyter":{"outputs_hidden":false}}
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]/255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]/255

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-02-22T14:54:13.562608Z","iopub.execute_input":"2026-02-22T14:54:13.563010Z","iopub.status.idle":"2026-02-22T14:54:13.573746Z","shell.execute_reply.started":"2026-02-22T14:54:13.562979Z","shell.execute_reply":"2026-02-22T14:54:13.572620Z"}}
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z,0)

def softmax(Z):
    Z -= np.max(Z, axis=0)  # Subtract max value for numerical stability
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T14:54:13.574664Z","iopub.execute_input":"2026-02-22T14:54:13.574949Z","iopub.status.idle":"2026-02-22T14:54:13.592353Z","shell.execute_reply.started":"2026-02-22T14:54:13.574922Z","shell.execute_reply":"2026-02-22T14:54:13.591411Z"},"jupyter":{"outputs_hidden":false}}
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def J(w): return w[0]**2 + 2 * w[1]**2
def grad_J(w): return np.array([2 * w[0], 4 * w[1]])
    
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ",i)
            predictions = get_predictions(A2)
            accuracy = round(get_accuracy(predictions, Y)*100, 2)
            print("Accuracy: ", accuracy, "%")
    return W1, b1, W2, b2



# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T14:54:13.593549Z","iopub.execute_input":"2026-02-22T14:54:13.594011Z","iopub.status.idle":"2026-02-22T15:06:33.852139Z","shell.execute_reply.started":"2026-02-22T14:54:13.593975Z","shell.execute_reply":"2026-02-22T15:06:33.850280Z"},"jupyter":{"outputs_hidden":false}}
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 1000)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T19:41:53.471313Z","iopub.execute_input":"2026-02-22T19:41:53.472101Z","iopub.status.idle":"2026-02-22T19:41:53.480661Z","shell.execute_reply.started":"2026-02-22T19:41:53.472064Z","shell.execute_reply":"2026-02-22T19:41:53.479833Z"}}
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T19:41:53.481925Z","iopub.execute_input":"2026-02-22T19:41:53.482597Z","iopub.status.idle":"2026-02-22T19:41:53.953395Z","shell.execute_reply.started":"2026-02-22T19:41:53.482546Z","shell.execute_reply":"2026-02-22T19:41:53.952382Z"}}
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# %% [code] {"execution":{"iopub.status.busy":"2026-02-22T19:41:53.954445Z","iopub.execute_input":"2026-02-22T19:41:53.955043Z","iopub.status.idle":"2026-02-22T19:41:53.976902Z","shell.execute_reply.started":"2026-02-22T19:41:53.955011Z","shell.execute_reply":"2026-02-22T19:41:53.976110Z"}}
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy_dev = np.float64(get_accuracy(dev_predictions, Y_dev))
print(dev_predictions)
print("   ")
print("Accuracy:", accuracy_dev*100, "%")