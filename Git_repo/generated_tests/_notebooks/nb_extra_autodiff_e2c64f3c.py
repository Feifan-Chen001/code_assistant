# --- notebook cell 6 ---
# To support both python 2 and python 3
from __future__ import absolute_import, division, print_function, unicode_literals

# --- notebook cell 9 ---
def f(x,y):
    return x*x*y + y + 2

# --- notebook cell 11 ---
def df(x,y):
    return 2*x*y, x*x + 1

# --- notebook cell 13 ---
df(3, 4)

# --- notebook cell 16 ---
def d2f(x, y):
    return [2*y, 2*x], [2*x, 0]

# --- notebook cell 17 ---
d2f(3, 4)

# --- notebook cell 21 ---
def gradients(func, vars_list, eps=0.0001):
    partial_derivatives = []
    base_func_eval = func(*vars_list)
    for idx in range(len(vars_list)):
        tweaked_vars = vars_list[:]
        tweaked_vars[idx] += eps
        tweaked_func_eval = func(*tweaked_vars)
        derivative = (tweaked_func_eval - base_func_eval) / eps
        partial_derivatives.append(derivative)
    return partial_derivatives

# --- notebook cell 22 ---
def df(x, y):
    return gradients(f, [x, y])

# --- notebook cell 23 ---
df(3, 4)

# --- notebook cell 26 ---
def dfdx(x, y):
    return gradients(f, [x,y])[0]

def dfdy(x, y):
    return gradients(f, [x,y])[1]

dfdx(3., 4.), dfdy(3., 4.)

# --- notebook cell 28 ---
def d2f(x, y):
    return [gradients(dfdx, [3., 4.]), gradients(dfdy, [3., 4.])]

# --- notebook cell 29 ---
d2f(3, 4)

# --- notebook cell 33 ---
class Const(object):
    def __init__(self, value):
        self.value = value
    def evaluate(self):
        return self.value
    def __str__(self):
        return str(self.value)

class Var(object):
    def __init__(self, name, init_value=0):
        self.value = init_value
        self.name = name
    def evaluate(self):
        return self.value
    def __str__(self):
        return self.name

class BinaryOperator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Add(BinaryOperator):
    def evaluate(self):
        return self.a.evaluate() + self.b.evaluate()
    def __str__(self):
        return "{} + {}".format(self.a, self.b)

class Mul(BinaryOperator):
    def evaluate(self):
        return self.a.evaluate() * self.b.evaluate()
    def __str__(self):
        return "({}) * ({})".format(self.a, self.b)

# --- notebook cell 35 ---
x = Var("x")
y = Var("y")
f = Add(Mul(Mul(x, x), y), Add(y, Const(2))) # f(x,y) = x²y + y + 2

# --- notebook cell 37 ---
x.value = 3
y.value = 4
f.evaluate()

# --- notebook cell 45 ---
from math import sin

def z(x):
    return sin(x**2)

gradients(z, [3])

# --- notebook cell 50 ---
Const.gradient = lambda self, var: Const(0)
Var.gradient = lambda self, var: Const(1) if self is var else Const(0)
Add.gradient = lambda self, var: Add(self.a.gradient(var), self.b.gradient(var))
Mul.gradient = lambda self, var: Add(Mul(self.a, self.b.gradient(var)), Mul(self.a.gradient(var), self.b))

x = Var(name="x", init_value=3.)
y = Var(name="y", init_value=4.)
f = Add(Mul(Mul(x, x), y), Add(y, Const(2))) # f(x,y) = x²y + y + 2

dfdx = f.gradient(x)  # 2xy
dfdy = f.gradient(y)  # x² + 1

# --- notebook cell 51 ---
dfdx.evaluate(), dfdy.evaluate()

# --- notebook cell 53 ---
d2fdxdx = dfdx.gradient(x) # 2y
d2fdxdy = dfdx.gradient(y) # 2x
d2fdydx = dfdy.gradient(x) # 2x
d2fdydy = dfdy.gradient(y) # 0

# --- notebook cell 54 ---
[[d2fdxdx.evaluate(), d2fdxdy.evaluate()],
 [d2fdydx.evaluate(), d2fdydy.evaluate()]]

# --- notebook cell 60 ---
class DualNumber(object):
    def __init__(self, value=0.0, eps=0.0):
        self.value = value
        self.eps = eps
    def __add__(self, b):
        return DualNumber(self.value + self.to_dual(b).value,
                          self.eps + self.to_dual(b).eps)
    def __radd__(self, a):
        return self.to_dual(a).__add__(self)
    def __mul__(self, b):
        return DualNumber(self.value * self.to_dual(b).value,
                          self.eps * self.to_dual(b).value + self.value * self.to_dual(b).eps)
    def __rmul__(self, a):
        return self.to_dual(a).__mul__(self)
    def __str__(self):
        if self.eps:
            return "{:.1f} + {:.1f}ε".format(self.value, self.eps)
        else:
            return "{:.1f}".format(self.value)
    def __repr__(self):
        return str(self)
    @classmethod
    def to_dual(cls, n):
        if hasattr(n, "value"):
            return n
        else:
            return cls(n)

# --- notebook cell 62 ---
3 + DualNumber(3, 4)

# --- notebook cell 64 ---
DualNumber(3, 4) * DualNumber(5, 7)

# --- notebook cell 66 ---
x.value = DualNumber(3.0)
y.value = DualNumber(4.0)

f.evaluate()

# --- notebook cell 68 ---
x.value = DualNumber(3.0, 1.0)  # 3 + ε
y.value = DualNumber(4.0)       # 4

dfdx = f.evaluate().eps

x.value = DualNumber(3.0)       # 3
y.value = DualNumber(4.0, 1.0)  # 4 + ε

dfdy = f.evaluate().eps

# --- notebook cell 69 ---
dfdx

# --- notebook cell 70 ---
dfdy

# --- notebook cell 74 ---
class Const(object):
    def __init__(self, value):
        self.value = value
    def evaluate(self):
        return self.value
    def backpropagate(self, gradient):
        pass
    def __str__(self):
        return str(self.value)

class Var(object):
    def __init__(self, name, init_value=0):
        self.value = init_value
        self.name = name
        self.gradient = 0
    def evaluate(self):
        return self.value
    def backpropagate(self, gradient):
        self.gradient += gradient
    def __str__(self):
        return self.name

class BinaryOperator(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

class Add(BinaryOperator):
    def evaluate(self):
        self.value = self.a.evaluate() + self.b.evaluate()
        return self.value
    def backpropagate(self, gradient):
        self.a.backpropagate(gradient)
        self.b.backpropagate(gradient)
    def __str__(self):
        return "{} + {}".format(self.a, self.b)

class Mul(BinaryOperator):
    def evaluate(self):
        self.value = self.a.evaluate() * self.b.evaluate()
        return self.value
    def backpropagate(self, gradient):
        self.a.backpropagate(gradient * self.b.value)
        self.b.backpropagate(gradient * self.a.value)
    def __str__(self):
        return "({}) * ({})".format(self.a, self.b)

# --- notebook cell 75 ---
x = Var("x", init_value=3)
y = Var("y", init_value=4)
f = Add(Mul(Mul(x, x), y), Add(y, Const(2))) # f(x,y) = x²y + y + 2

result = f.evaluate()
f.backpropagate(1.0)

# --- notebook cell 76 ---
print(f)

# --- notebook cell 77 ---
result

# --- notebook cell 78 ---
x.gradient

# --- notebook cell 79 ---
y.gradient

# --- notebook cell 82 ---
try:
    # %tensorflow_version only exists in Colab.
except Exception:
    pass

import tensorflow as tf

# --- notebook cell 83 ---
tf.reset_default_graph()

x = tf.Variable(3., name="x")
y = tf.Variable(4., name="y")
f = x*x*y + y + 2

jacobians = tf.gradients(f, [x, y])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    f_val, jacobians_val = sess.run([f, jacobians])

f_val, jacobians_val

# --- notebook cell 85 ---
hessians_x = tf.gradients(jacobians[0], [x, y])
hessians_y = tf.gradients(jacobians[1], [x, y])

def replace_none_with_zero(tensors):
    return [tensor if tensor is not None else tf.constant(0.)
            for tensor in tensors]

hessians_x = replace_none_with_zero(hessians_x)
hessians_y = replace_none_with_zero(hessians_y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    hessians_x_val, hessians_y_val = sess.run([hessians_x, hessians_y])

hessians_x_val, hessians_y_val