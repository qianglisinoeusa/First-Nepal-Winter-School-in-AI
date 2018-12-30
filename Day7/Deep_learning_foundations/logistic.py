#!/usr/bin/env python

"""
An introduction to pytorch nn modules via logistic regression. A challenge at
the end for fitting a nonlinear decision boundary.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# Generic functions
###############################################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mixture_of_gaussians(n, k=2, pi=None):
    """
    Simulate 2D Mixture of Gaussians

    :param n: The number of points to simulate.
    :param k: The number of mixture components.
    :param pi: The marginal probabilities of different classes.
    :return mean, data: A tuple giving (1) the means for the mixture components
      and (2) the positions of the mixture of gaussians points
    """
    if not pi:
        pi = np.ones(k) / k

    means = np.array([[0, 1], [-3, -3]])
    z = np.random.choice(range(k), n, True, pi)

    data = np.zeros((n, 2))
    for i in range(n):
        data[i, ] = np.random.normal(means[z[i]])

    return means, data


def mix_gauss_classes(x, theta):
    """
    Label a Mixture of Gaussians

    This simulates points x using a mixture of gaussians and then defines
    binary $y$ according to a sigmoid on theta ^T x

    :param x: An n x 2 numpy array giving the x coordinates of the points of
      interest.
    :param theta: The direction used to define the class probabilities y
    :return pi, y: A tuple giving the class probabilties and assignments for
      each point x, respectively
    """
    n, _ = x.shape
    x_tilde = np.hstack((np.ones((n, 1)), x))
    pi = sigmoid(np.dot(x_tilde, theta))

    y = np.zeros(n)
    for i in range(n):
        y[i] = np.random.choice(range(k), size=1, p=[1 - pi[i], pi[i]])

    return pi, y


###############################################################################
# Logistic regression torch module
###############################################################################

class LogisticRegression(torch.nn.Module):
    def __init__(self, D):
        super(LogisticRegression, self).__init__() # initialize using superclass
        self.linear = torch.nn.Linear(D, 1)
        self.output = torch.nn.Sigmoid()

    def forward(self, x):
        Wx = self.linear(x)
        return self.output(Wx)


###############################################################################
# Simulation experiment
###############################################################################
n = 500
means, x = mixture_of_gaussians(n)
plt.scatter(x[:, 0], x[:, 1])

theta = means[1, :] - means[0, :]
theta = np.hstack((0.2, theta))
theta /= np.sqrt(sum(theta ** 2))

pi, y = mix_gauss_classes(x, theta)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.scatter(x[:, 0], x[:, 1], c=pi)

# fit a logistic regression on this
model = LogisticRegression(2)
x_tens = torch.as_tensor(x, dtype=torch.float32)
y_tens = torch.as_tensor(y, dtype=torch.float32)
y_tens = y_tens.reshape((n, 1))

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train the model
for i in range(500):
    y_hat = model(x_tens)
    loss = loss_fn(input=y_hat, target=y_tens)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Uncomment this to save images of the decision boundary being adjusted
    # if i % 5 == 0:
    #     print(loss.item())
    #     fig = plt.figure()
    #     plt.scatter(x[:, 0], x[:, 1], c=y_hat.detach().numpy().flatten())
    #     name = "image_{}".format(str(i).rjust(4, "0"))
    #     fig.savefig(name)


# now do a nonlinear transformation
z = np.vstack([x[:, 0] ** 2 - x[:, 1] * x[:, 0], x[:, 0] ** 2 + x[:, 1] ** 2 - 2 * x[:, 0] * x[:, 1]]).T
plt.scatter(z[:, 0], z[:, 1], c=pi)

# fit a logistic regression on this new z
# see if you can add a hidden layer to the logistic model to get a better
# decision boundary
