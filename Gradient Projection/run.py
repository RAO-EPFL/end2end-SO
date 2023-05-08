import torch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


class Projection_Element(torch.nn.Module):
    # projection element implement with cvxpylayers
    def __init__(self):
        super(Projection_Element, self).__init__()
        x = cp.Parameter(2)
        y = cp.Variable(2)
        objective = cp.Minimize(cp.norm(x-y, 2))
        problem = cp.Problem(objective, [cp.norm(y) <= 1])
        self.proj_layer = CvxpyLayer(problem, parameters=[x], variables=[y])

    def forward(self, x):
        return self.proj_layer(x)[0]


def path(x, loss):
    # return training path with starting point w=x
    points = []
    points.append(x.detach().numpy().copy())
    optimizer = torch.optim.SGD([x], lr=1e-1)
    for i in range(100):
        optimizer.zero_grad()
        l = loss(projection(x))
        l.backward()
        optimizer.step()
        points.append(x.detach().numpy().copy())
    return points


def plot_contour(loss, step=0.1):
    # plot the contour lines of the loss function, taking into account the projection
    def proj(x):
        if torch.norm(x) <= 1:
            return x
        else:
            return x / torch.norm(x)
    feature_x = np.arange(-1.5, 1.5, step)
    feature_x[np.abs(feature_x) < 1e-5] = 0
    feature_y = np.arange(-1.5, 1.5, step)
    feature_x[np.abs(feature_y) < 1e-5] = 0

    # Creating 2-D grid of features
    [X, Y] = np.meshgrid(feature_x, feature_y)
    Z = X*np.inf
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = loss(proj(torch.Tensor([[X[i, j], Y[i, j]]])))
    cf = plt.contour(X, Y, Z, np.linspace(0, 2.25, 2*int(2.25/0.25)+1), alpha=0.5)
    cmap = cf.get_cmap()
    plt.plot([0, 0], [-1.5, -1], alpha=0.5, color=cmap(np.linspace(0, 2.25, 2 * int(2.25/0.25)+1))[-1])


def plot_circle(radius=1):
    # plot the unit circle
    x = np.arange(-radius, radius+0.001, 0.001)
    y = np.sqrt(radius**2-x**2)
    plt.fill_between(x, -y, y, alpha=0.1)


if __name__ == '__main__':
    x1 = torch.Tensor([[1.1, 0]])
    x1.requires_grad = True
    x2 = torch.Tensor([[-0.95, 0]])
    x2.requires_grad = True

    target = torch.Tensor([[0, 0.5]])
    projection = Projection_Element()
    path1 = path(x1, lambda x: torch.norm(x-target)**2)
    path2 = path(x2, lambda x: torch.norm(x-target)**2)

    plt.figure(figsize=(6, 6))
    plot_circle()
    r = plot_contour(lambda x: torch.norm(x-target)**2, step=0.01)

    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    plt.plot([z[0, 0] for z in path1], [z[0, 1] for z in path1], color='C0', linewidth=3)
    projected = [projection(torch.Tensor(z)).numpy() for z in path1]
    plt.plot([z[0, 0] for z in projected], [z[0, 1] for z in projected], '--', color='C0', linewidth=3)
    plt.arrow(path1[-1][0, 0] + 0.05, path1[-1][0, 1], -0.001, 0, head_width=0.05, color='C0', head_starts_at_zero=False)
    plt.arrow(projected[-1][0, 0] + 0.05, projected[-1][0, 1], -0.001, 0, head_width=0.05, color='C0', head_starts_at_zero=False)

    plt.plot([z[0, 0] for z in path2], [z[0, 1] for z in path2], color='C1', linewidth=2)
    projected = [projection(torch.Tensor(z)).numpy() for z in path2]
    plt.plot([z[0, 0] for z in projected], [z[0, 1] for z in projected], '--', color='C1', linewidth=4)
    plt.arrow(projected[0][0, 0], projected[0][0, 1],
              (projected[-1][0, 0]-projected[0][0, 0])*0.94,
              (projected[-1][0, 1]-projected[0][0, 1])*0.94,
              head_width=0.05, color='C1', head_starts_at_zero=False)

    plt.scatter(target[0, 0], target[0, 1], marker='x', color='red', s=50, zorder=20)
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    plt.xticks([-1, 0, 1], [-1, 0, 1])
    plt.yticks([-1, 0, 1], [-1, 0, 1])
    plt.savefig('gradient_projection.pdf')
    plt.show()
