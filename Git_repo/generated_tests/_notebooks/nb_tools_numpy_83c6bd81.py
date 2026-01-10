# --- notebook cell 4 ---
from __future__ import division, print_function, unicode_literals

# --- notebook cell 6 ---
import numpy as np

# --- notebook cell 9 ---
np.zeros(5)

# --- notebook cell 11 ---
np.zeros((3,4))

# --- notebook cell 13 ---
a = np.zeros((3,4))
a

# --- notebook cell 14 ---
a.shape

# --- notebook cell 15 ---
a.ndim  # equal to len(a.shape)

# --- notebook cell 16 ---
a.size

# --- notebook cell 18 ---
np.zeros((2,3,4))

# --- notebook cell 20 ---
type(np.zeros((3,4)))

# --- notebook cell 22 ---
np.ones((3,4))

# --- notebook cell 24 ---
np.full((3,4), np.pi)

# --- notebook cell 26 ---
np.empty((2,3))

# --- notebook cell 28 ---
np.array([[1,2,3,4], [10, 20, 30, 40]])

# --- notebook cell 30 ---
np.arange(1, 5)

# --- notebook cell 32 ---
np.arange(1.0, 5.0)

# --- notebook cell 34 ---
np.arange(1, 5, 0.5)

# --- notebook cell 36 ---
print(np.arange(0, 5/3, 1/3)) # depending on floating point errors, the max value is 4/3 or 5/3.
print(np.arange(0, 5/3, 0.333333333))
print(np.arange(0, 5/3, 0.333333334))

# --- notebook cell 38 ---
print(np.linspace(0, 5/3, 6))

# --- notebook cell 40 ---
np.random.rand(3,4)

# --- notebook cell 42 ---
np.random.randn(3,4)

# --- notebook cell 44 ---
import matplotlib.pyplot as plt

# --- notebook cell 45 ---
plt.hist(np.random.rand(100000), normed=True, bins=100, histtype="step", color="blue", label="rand")
plt.hist(np.random.randn(100000), normed=True, bins=100, histtype="step", color="red", label="randn")
plt.axis([-2.5, 2.5, 0, 1.1])
plt.legend(loc = "upper left")
plt.title("Random distributions")
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# --- notebook cell 47 ---
def my_function(z, y, x):
    return x * y + z

np.fromfunction(my_function, (3, 2, 10))

# --- notebook cell 50 ---
c = np.arange(1, 5)
print(c.dtype, c)

# --- notebook cell 51 ---
c = np.arange(1.0, 5.0)
print(c.dtype, c)

# --- notebook cell 53 ---
d = np.arange(1, 5, dtype=np.complex64)
print(d.dtype, d)

# --- notebook cell 55 ---
e = np.arange(1, 5, dtype=np.complex64)
e.itemsize

# --- notebook cell 57 ---
f = np.array([[1,2],[1000, 2000]], dtype=np.int32)
f.data

# --- notebook cell 59 ---
if (hasattr(f.data, "tobytes")):
    data_bytes = f.data.tobytes() # python 3
else:
    data_bytes = memoryview(f.data).tobytes() # python 2

data_bytes

# --- notebook cell 62 ---
g = np.arange(24)
print(g)
print("Rank:", g.ndim)

# --- notebook cell 63 ---
g.shape = (6, 4)
print(g)
print("Rank:", g.ndim)

# --- notebook cell 64 ---
g.shape = (2, 3, 4)
print(g)
print("Rank:", g.ndim)

# --- notebook cell 66 ---
g2 = g.reshape(4,6)
print(g2)
print("Rank:", g2.ndim)

# --- notebook cell 68 ---
g2[1, 2] = 999
g2

# --- notebook cell 70 ---
g

# --- notebook cell 72 ---
g.ravel()

# --- notebook cell 74 ---
a = np.array([14, 23, 32, 41])
b = np.array([5,  4,  3,  2])
print("a + b  =", a + b)
print("a - b  =", a - b)
print("a * b  =", a * b)
print("a / b  =", a / b)
print("a // b  =", a // b)
print("a % b  =", a % b)
print("a ** b =", a ** b)

# --- notebook cell 78 ---
h = np.arange(5).reshape(1, 1, 5)
h

# --- notebook cell 80 ---
h + [10, 20, 30, 40, 50]  # same as: h + [[[10, 20, 30, 40, 50]]]

# --- notebook cell 82 ---
k = np.arange(6).reshape(2, 3)
k

# --- notebook cell 84 ---
k + [[100], [200]]  # same as: k + [[100, 100, 100], [200, 200, 200]]

# --- notebook cell 86 ---
k + [100, 200, 300]  # after rule 1: [[100, 200, 300]], and after rule 2: [[100, 200, 300], [100, 200, 300]]

# --- notebook cell 88 ---
k + 1000  # same as: k + [[1000, 1000, 1000], [1000, 1000, 1000]]

# --- notebook cell 90 ---
try:
    k + [33, 44]
except ValueError as e:
    print(e)

# --- notebook cell 93 ---
k1 = np.arange(0, 5, dtype=np.uint8)
print(k1.dtype, k1)

# --- notebook cell 94 ---
k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)

# --- notebook cell 96 ---
k3 = k1 + 1.5
print(k3.dtype, k3)

# --- notebook cell 99 ---
m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]

# --- notebook cell 101 ---
m < 25  # equivalent to m < [25, 25, 25, 25]

# --- notebook cell 103 ---
m[m < 25]

# --- notebook cell 106 ---
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print(a)
print("mean =", a.mean())

# --- notebook cell 108 ---
for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())

# --- notebook cell 110 ---
c=np.arange(24).reshape(2,3,4)
c

# --- notebook cell 111 ---
c.sum(axis=0)  # sum across matrices

# --- notebook cell 112 ---
c.sum(axis=1)  # sum across rows

# --- notebook cell 114 ---
c.sum(axis=(0,2))  # sum across matrices and columns

# --- notebook cell 115 ---
0+1+2+3 + 12+13+14+15, 4+5+6+7 + 16+17+18+19, 8+9+10+11 + 20+21+22+23

# --- notebook cell 117 ---
a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
np.square(a)

# --- notebook cell 119 ---
print("Original ndarray")
print(a)
for func in (np.abs, np.sqrt, np.exp, np.log, np.sign, np.ceil, np.modf, np.isnan, np.cos):
    print("\n", func.__name__)
    print(func(a))

# --- notebook cell 121 ---
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
np.add(a, b)  # equivalent to a + b

# --- notebook cell 122 ---
np.greater(a, b)  # equivalent to a > b

# --- notebook cell 123 ---
np.maximum(a, b)

# --- notebook cell 124 ---
np.copysign(a, b)

# --- notebook cell 126 ---
a = np.array([1, 5, 3, 19, 13, 7, 3])
a[3]

# --- notebook cell 127 ---
a[2:5]

# --- notebook cell 128 ---
a[2:-1]

# --- notebook cell 129 ---
a[:2]

# --- notebook cell 130 ---
a[2::2]

# --- notebook cell 131 ---
a[::-1]

# --- notebook cell 133 ---
a[3]=999
a

# --- notebook cell 135 ---
a[2:5] = [997, 998, 999]
a

# --- notebook cell 137 ---
a[2:5] = -1
a

# --- notebook cell 139 ---
try:
    a[2:5] = [1,2,3,4,5,6]  # too long
except ValueError as e:
    print(e)

# --- notebook cell 141 ---
try:
    del a[2:5]
except ValueError as e:
    print(e)

# --- notebook cell 143 ---
a_slice = a[2:6]
a_slice[1] = 1000
a  # the original array was modified!

# --- notebook cell 144 ---
a[3] = 2000
a_slice  # similarly, modifying the original array modifies the slice!

# --- notebook cell 146 ---
another_slice = a[2:6].copy()
another_slice[1] = 3000
a  # the original array is untouched

# --- notebook cell 147 ---
a[3] = 4000
another_slice  # similary, modifying the original array does not affect the slice copy

# --- notebook cell 149 ---
b = np.arange(48).reshape(4, 12)
b

# --- notebook cell 150 ---
b[1, 2]  # row 1, col 2

# --- notebook cell 151 ---
b[1, :]  # row 1, all columns

# --- notebook cell 152 ---
b[:, 1]  # all rows, column 1

# --- notebook cell 154 ---
b[1, :]

# --- notebook cell 155 ---
b[1:2, :]

# --- notebook cell 158 ---
b[(0,2), 2:5]  # rows 0 and 2, columns 2 to 4 (5-1)

# --- notebook cell 159 ---
b[:, (-1, 2, -1)]  # all rows, columns -1 (last), 2 and -1 (again, and in this order)

# --- notebook cell 161 ---
b[(-1, 2, -1, 2), (5, 9, 1, 9)]  # returns a 1D array with b[-1, 5], b[2, 9], b[-1, 1] and b[2, 9] (again)

# --- notebook cell 163 ---
c = b.reshape(4,2,6)
c

# --- notebook cell 164 ---
c[2, 1, 4]  # matrix 2, row 1, col 4

# --- notebook cell 165 ---
c[2, :, 3]  # matrix 2, all rows, col 3

# --- notebook cell 167 ---
c[2, 1]  # Return matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]

# --- notebook cell 169 ---
c[2, ...]  #  matrix 2, all rows, all columns.  This is equivalent to c[2, :, :]

# --- notebook cell 170 ---
c[2, 1, ...]  # matrix 2, row 1, all columns.  This is equivalent to c[2, 1, :]

# --- notebook cell 171 ---
c[2, ..., 3]  # matrix 2, all rows, column 3.  This is equivalent to c[2, :, 3]

# --- notebook cell 172 ---
c[..., 3]  # all matrices, all rows, column 3.  This is equivalent to c[:, :, 3]

# --- notebook cell 174 ---
b = np.arange(48).reshape(4, 12)
b

# --- notebook cell 175 ---
rows_on = np.array([True, False, True, False])
b[rows_on, :]  # Rows 0 and 2, all columns. Equivalent to b[(0, 2), :]

# --- notebook cell 176 ---
cols_on = np.array([False, True, False] * 4)
b[:, cols_on]  # All rows, columns 1, 4, 7 and 10

# --- notebook cell 178 ---
b[np.ix_(rows_on, cols_on)]

# --- notebook cell 179 ---
np.ix_(rows_on, cols_on)

# --- notebook cell 181 ---
b[b % 3 == 1]

# --- notebook cell 183 ---
c = np.arange(24).reshape(2, 3, 4)  # A 3D array (composed of two 3x4 matrices)
c

# --- notebook cell 184 ---
for m in c:
    print("Item:")
    print(m)

# --- notebook cell 185 ---
for i in range(len(c)):  # Note that len(c) == c.shape[0]
    print("Item:")
    print(c[i])

# --- notebook cell 187 ---
for i in c.flat:
    print("Item:", i)

# --- notebook cell 189 ---
q1 = np.full((3,4), 1.0)
q1

# --- notebook cell 190 ---
q2 = np.full((4,4), 2.0)
q2

# --- notebook cell 191 ---
q3 = np.full((3,4), 3.0)
q3

# --- notebook cell 193 ---
q4 = np.vstack((q1, q2, q3))
q4

# --- notebook cell 194 ---
q4.shape

# --- notebook cell 196 ---
q5 = np.hstack((q1, q3))
q5

# --- notebook cell 197 ---
q5.shape

# --- notebook cell 199 ---
try:
    q5 = np.hstack((q1, q2, q3))
except ValueError as e:
    print(e)

# --- notebook cell 201 ---
q7 = np.concatenate((q1, q2, q3), axis=0)  # Equivalent to vstack
q7

# --- notebook cell 202 ---
q7.shape

# --- notebook cell 205 ---
q8 = np.stack((q1, q3))
q8

# --- notebook cell 206 ---
q8.shape

# --- notebook cell 208 ---
r = np.arange(24).reshape(6,4)
r

# --- notebook cell 210 ---
r1, r2, r3 = np.vsplit(r, 3)
r1

# --- notebook cell 211 ---
r2

# --- notebook cell 212 ---
r3

# --- notebook cell 214 ---
r4, r5 = np.hsplit(r, 2)
r4

# --- notebook cell 215 ---
r5

# --- notebook cell 217 ---
t = np.arange(24).reshape(4,2,3)
t

# --- notebook cell 219 ---
t1 = t.transpose((1,2,0))
t1

# --- notebook cell 220 ---
t1.shape

# --- notebook cell 222 ---
t2 = t.transpose()  # equivalent to t.transpose((2, 1, 0))
t2

# --- notebook cell 223 ---
t2.shape

# --- notebook cell 225 ---
t3 = t.swapaxes(0,1)  # equivalent to t.transpose((1, 0, 2))
t3

# --- notebook cell 226 ---
t3.shape

# --- notebook cell 228 ---
m1 = np.arange(10).reshape(2,5)
m1

# --- notebook cell 229 ---
m1.T

# --- notebook cell 231 ---
m2 = np.arange(5)
m2

# --- notebook cell 232 ---
m2.T

# --- notebook cell 234 ---
m2r = m2.reshape(1,5)
m2r

# --- notebook cell 235 ---
m2r.T

# --- notebook cell 237 ---
n1 = np.arange(10).reshape(2, 5)
n1

# --- notebook cell 238 ---
n2 = np.arange(15).reshape(5,3)
n2

# --- notebook cell 239 ---
n1.dot(n2)

# --- notebook cell 242 ---
import numpy.linalg as linalg

m3 = np.array([[1,2,3],[5,7,11],[21,29,31]])
m3

# --- notebook cell 243 ---
linalg.inv(m3)

# --- notebook cell 245 ---
linalg.pinv(m3)

# --- notebook cell 247 ---
m3.dot(linalg.inv(m3))

# --- notebook cell 249 ---
np.eye(3)

# --- notebook cell 251 ---
q, r = linalg.qr(m3)
q

# --- notebook cell 252 ---
r

# --- notebook cell 253 ---
q.dot(r)  # q.r equals m3

# --- notebook cell 255 ---
linalg.det(m3)  # Computes the matrix determinant

# --- notebook cell 257 ---
eigenvalues, eigenvectors = linalg.eig(m3)
eigenvalues # λ

# --- notebook cell 258 ---
eigenvectors # v

# --- notebook cell 259 ---
m3.dot(eigenvectors) - eigenvalues * eigenvectors  # m3.v - λ*v = 0

# --- notebook cell 261 ---
m4 = np.array([[1,0,0,0,2], [0,0,3,0,0], [0,0,0,0,0], [0,2,0,0,0]])
m4

# --- notebook cell 262 ---
U, S_diag, V = linalg.svd(m4)
U

# --- notebook cell 263 ---
S_diag

# --- notebook cell 265 ---
S = np.zeros((4, 5))
S[np.diag_indices(4)] = S_diag
S  # Σ

# --- notebook cell 266 ---
V

# --- notebook cell 267 ---
U.dot(S).dot(V) # U.Σ.V == m4

# --- notebook cell 269 ---
np.diag(m3)  # the values in the diagonal of m3 (top left to bottom right)

# --- notebook cell 270 ---
np.trace(m3)  # equivalent to np.diag(m3).sum()

# --- notebook cell 273 ---
coeffs  = np.array([[2, 6], [5, 3]])
depvars = np.array([6, -9])
solution = linalg.solve(coeffs, depvars)
solution

# --- notebook cell 275 ---
coeffs.dot(solution), depvars  # yep, it's the same

# --- notebook cell 277 ---
np.allclose(coeffs.dot(solution), depvars)

# --- notebook cell 279 ---
import math
data = np.empty((768, 1024))
for y in range(768):
    for x in range(1024):
        data[y, x] = math.sin(x*y/40.5)  # BAD! Very inefficient.

# --- notebook cell 281 ---
x_coords = np.arange(0, 1024)  # [0, 1, 2, ..., 1023]
y_coords = np.arange(0, 768)   # [0, 1, 2, ..., 767]
X, Y = np.meshgrid(x_coords, y_coords)
X

# --- notebook cell 282 ---
Y

# --- notebook cell 284 ---
data = np.sin(X*Y/40.5)

# --- notebook cell 286 ---
import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig = plt.figure(1, figsize=(7, 6))
plt.imshow(data, cmap=cm.hot, interpolation="bicubic")
plt.show()

# --- notebook cell 288 ---
a = np.random.rand(2,3)
a

# --- notebook cell 289 ---
np.save("my_array", a)

# --- notebook cell 291 ---
with open("my_array.npy", "rb") as f:
    content = f.read()

content

# --- notebook cell 293 ---
a_loaded = np.load("my_array.npy")
a_loaded

# --- notebook cell 295 ---
np.savetxt("my_array.csv", a)

# --- notebook cell 297 ---
with open("my_array.csv", "rt") as f:
    print(f.read())

# --- notebook cell 299 ---
np.savetxt("my_array.csv", a, delimiter=",")

# --- notebook cell 301 ---
a_loaded = np.loadtxt("my_array.csv", delimiter=",")
a_loaded

# --- notebook cell 303 ---
b = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
b

# --- notebook cell 304 ---
np.savez("my_arrays", my_a=a, my_b=b)

# --- notebook cell 306 ---
with open("my_arrays.npz", "rb") as f:
    content = f.read()

repr(content)[:180] + "[...]"

# --- notebook cell 308 ---
my_arrays = np.load("my_arrays.npz")
my_arrays

# --- notebook cell 310 ---
my_arrays.keys()

# --- notebook cell 311 ---
my_arrays["my_a"]