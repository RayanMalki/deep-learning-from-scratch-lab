import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        # numerical value of the node
        self.data = data

        self.grad = 0.0

        self._backward = lambda: None

        self._prev = set(_children)

        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):

        assert isinstance(other, (int, float)), "Power must be int or float"

        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():

            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return other * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(max(0, self.data), (self,), "Relu")

        def _backward():
            if self.data > 0:
                self.grad += 1.0 * out.grad
            else:
                self.grad += 0.0 * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def built_topo(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    built_topo(child)
                topo.append(node)

        built_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
