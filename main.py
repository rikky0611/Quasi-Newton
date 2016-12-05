# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def function(x, y):
    return x**2/4-x*y+2*y**2-x+4*y


def derive_function(x, y):
    return np.array([[x/2-y-1], [-x+4*y+4]])


def get_gradient(x):
    x = x.T
    x = x.reshape(1, 2)
    return derive_function(x[0][0], x[0][1])


def check_if_finish(x, eps=0.00001):
    gradient = get_gradient(x)
    print("norm_x = {0}".format(np.linalg.norm(gradient)))
    return np.linalg.norm(gradient) < eps


class Quasi_Newton():
    def __init__(self, H, x_next, x_now):
        self.H = H
        self.x_next = x_next
        self.x_now = x_now

    def calc_next_H(self):
        I = np.matrix(np.identity(2))
        self._s = self.x_next - self.x_now
        self._y = get_gradient(self.x_next)-get_gradient(self.x_now)
        self._p = 1/np.dot(self._y.T, self._s)
        H_next = (I-self._p*self._s*self._y.T)*self.H*(I-self._p*self._y*self._s.T)+self._p*self._s*self._s.T
        return H_next


if __name__ == '__main__':
    x_now = np.array([0., 0.])[:, np.newaxis]
    x_next = np.array([1., -4.])[:, np.newaxis]
    H = np.matrix(np.identity(2))
    alpha = 1.0

    cnt = 1
    x_array = []
    x_array.append(x_now)
    while True:
        x_array.append(x_next)
        print("-------------------------")
        print("loop{0} x_{0}={1}".format(cnt, x_next))
        if check_if_finish(x_next):
            print("収束点は{0}".format(x_next))
            break
        qn = Quasi_Newton(H, x_next, x_now)
        grad = get_gradient(x_next)
        d = -1.*qn.calc_next_H()*grad
        x_now = x_next
        x_next = np.array(x_next+alpha*d)
        cnt += 1
        x_array.append(x_next)
    
    n = 1000
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    X, Y = np.meshgrid(x, y)
    Z = function(X, Y)
    plt.contour(X, Y, Z)
    plt.gca().set_aspect('equal')
    
    X_arr = [x[0] for x in x_array]
    Y_arr = [x[1] for x in x_array]
    plt.title("alpha={0}".format(alpha))
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.plot(X_arr, Y_arr, '--bo')
    plt.show()
