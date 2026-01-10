# --- notebook cell 4 ---
from __future__ import division, print_function, unicode_literals

# --- notebook cell 7 ---
[10.5, 5.2, 3.25, 7.0]

# --- notebook cell 9 ---
import numpy as np
video = np.array([10.5, 5.2, 3.25, 7.0])
video

# --- notebook cell 11 ---
video.size

# --- notebook cell 13 ---
video[2]  # 3rd element

# --- notebook cell 15 ---
import matplotlib.pyplot as plt

# --- notebook cell 17 ---
u = np.array([2, 5])
v = np.array([3, 1])

# --- notebook cell 19 ---
x_coords, y_coords = zip(u, v)
plt.scatter(x_coords, y_coords, color=["r","b"])
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

# --- notebook cell 21 ---
def plot_vector2d(vector2d, origin=[0, 0], **options):
    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],
              head_width=0.2, head_length=0.3, length_includes_head=True,
              **options)

# --- notebook cell 23 ---
plot_vector2d(u, color="r")
plot_vector2d(v, color="b")
plt.axis([0, 9, 0, 6])
plt.grid()
plt.show()

# --- notebook cell 25 ---
a = np.array([1, 2, 8])
b = np.array([5, 6, 3])

# --- notebook cell 27 ---
from mpl_toolkits.mplot3d import Axes3D

subplot3d = plt.subplot(111, projection='3d')
x_coords, y_coords, z_coords = zip(a,b)
subplot3d.scatter(x_coords, y_coords, z_coords)
subplot3d.set_zlim3d([0, 9])
plt.show()

# --- notebook cell 29 ---
def plot_vectors3d(ax, vectors3d, z0, **options):
    for v in vectors3d:
        x, y, z = v
        ax.plot([x,x], [y,y], [z0, z], color="gray", linestyle='dotted', marker=".")
    x_coords, y_coords, z_coords = zip(*vectors3d)
    ax.scatter(x_coords, y_coords, z_coords, **options)

subplot3d = plt.subplot(111, projection='3d')
subplot3d.set_zlim([0, 9])
plot_vectors3d(subplot3d, [a,b], 0, color=("r","b"))
plt.show()

# --- notebook cell 31 ---
def vector_norm(vector):
    squares = [element**2 for element in vector]
    return sum(squares)**0.5

print("||", u, "|| =")
vector_norm(u)

# --- notebook cell 33 ---
import numpy.linalg as LA
LA.norm(u)

# --- notebook cell 35 ---
radius = LA.norm(u)
plt.gca().add_artist(plt.Circle((0,0), radius, color="#DDDDDD"))
plot_vector2d(u, color="red")
plt.axis([0, 8.7, 0, 6])
plt.grid()
plt.show()

# --- notebook cell 38 ---
print(" ", u)
print("+", v)
print("-"*10)
u + v

# --- notebook cell 40 ---
plot_vector2d(u, color="r")
plot_vector2d(v, color="b")
plot_vector2d(v, origin=u, color="b", linestyle="dotted")
plot_vector2d(u, origin=v, color="r", linestyle="dotted")
plot_vector2d(u+v, color="g")
plt.axis([0, 9, 0, 7])
plt.text(0.7, 3, "u", color="r", fontsize=18)
plt.text(4, 3, "u", color="r", fontsize=18)
plt.text(1.8, 0.2, "v", color="b", fontsize=18)
plt.text(3.1, 5.6, "v", color="b", fontsize=18)
plt.text(2.4, 2.5, "u+v", color="g", fontsize=18)
plt.grid()
plt.show()

# --- notebook cell 43 ---
t1 = np.array([2, 0.25])
t2 = np.array([2.5, 3.5])
t3 = np.array([1, 2])

x_coords, y_coords = zip(t1, t2, t3, t1)
plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")

plot_vector2d(v, t1, color="r", linestyle=":")
plot_vector2d(v, t2, color="r", linestyle=":")
plot_vector2d(v, t3, color="r", linestyle=":")

t1b = t1 + v
t2b = t2 + v
t3b = t3 + v

x_coords_b, y_coords_b = zip(t1b, t2b, t3b, t1b)
plt.plot(x_coords_b, y_coords_b, "b-", x_coords_b, y_coords_b, "bo")

plt.text(4, 4.2, "v", color="r", fontsize=18)
plt.text(3, 2.3, "v", color="r", fontsize=18)
plt.text(3.5, 0.4, "v", color="r", fontsize=18)

plt.axis([0, 6, 0, 5])
plt.grid()
plt.show()

# --- notebook cell 46 ---
print("1.5 *", u, "=")

1.5 * u

# --- notebook cell 48 ---
k = 2.5
t1c = k * t1
t2c = k * t2
t3c = k * t3

plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")

plot_vector2d(t1, color="r")
plot_vector2d(t2, color="r")
plot_vector2d(t3, color="r")

x_coords_c, y_coords_c = zip(t1c, t2c, t3c, t1c)
plt.plot(x_coords_c, y_coords_c, "b-", x_coords_c, y_coords_c, "bo")

plot_vector2d(k * t1, color="b", linestyle=":")
plot_vector2d(k * t2, color="b", linestyle=":")
plot_vector2d(k * t3, color="b", linestyle=":")

plt.axis([0, 9, 0, 9])
plt.grid()
plt.show()

# --- notebook cell 52 ---
plt.gca().add_artist(plt.Circle((0,0),1,color='c'))
plt.plot(0, 0, "ko")
plot_vector2d(v / LA.norm(v), color="k")
plot_vector2d(v, color="b", linestyle=":")
plt.text(0.3, 0.3, "$\hat{u}$", color="k", fontsize=18)
plt.text(1.5, 0.7, "$u$", color="b", fontsize=18)
plt.axis([-1.5, 5.5, -1.5, 3.5])
plt.grid()
plt.show()

# --- notebook cell 54 ---
def dot_product(v1, v2):
    return sum(v1i * v2i for v1i, v2i in zip(v1, v2))

dot_product(u, v)

# --- notebook cell 56 ---
np.dot(u,v)

# --- notebook cell 58 ---
u.dot(v)

# --- notebook cell 60 ---
print("  ",u)
print("* ",v, "(NOT a dot product)")
print("-"*10)

u * v

# --- notebook cell 63 ---
def vector_angle(u, v):
    cos_theta = u.dot(v) / LA.norm(u) / LA.norm(v)
    return np.arccos(np.clip(cos_theta, -1, 1))

theta = vector_angle(u, v)
print("Angle =", theta, "radians")
print("      =", theta * 180 / np.pi, "degrees")

# --- notebook cell 66 ---
u_normalized = u / LA.norm(u)
proj = v.dot(u_normalized) * u_normalized

plot_vector2d(u, color="r")
plot_vector2d(v, color="b")

plot_vector2d(proj, color="k", linestyle=":")
plt.plot(proj[0], proj[1], "ko")

plt.plot([proj[0], v[0]], [proj[1], v[1]], "b:")

plt.text(1, 2, "$proj_u v$", color="k", fontsize=18)
plt.text(1.8, 0.2, "$v$", color="b", fontsize=18)
plt.text(0.8, 3, "$u$", color="r", fontsize=18)

plt.axis([0, 8, 0, 5.5])
plt.grid()
plt.show()

# --- notebook cell 69 ---
[
    [10, 20, 30],
    [40, 50, 60]
]

# --- notebook cell 71 ---
A = np.array([
    [10,20,30],
    [40,50,60]
])
A

# --- notebook cell 74 ---
A.shape

# --- notebook cell 76 ---
A.size

# --- notebook cell 78 ---
A[1,2]  # 2nd row, 3rd column

# --- notebook cell 80 ---
A[1, :]  # 2nd row vector (as a 1D array)

# --- notebook cell 82 ---
A[:, 2]  # 3rd column vector (as a 1D array)

# --- notebook cell 84 ---
A[1:2, :]  # rows 2 to 3 (excluded): this returns row 2 as a one-row matrix

# --- notebook cell 85 ---
A[:, 2:3]  # columns 3 to 4 (excluded): this returns column 3 as a one-column matrix

# --- notebook cell 91 ---
np.diag([4, 5, 6])

# --- notebook cell 93 ---
D = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
np.diag(D)

# --- notebook cell 95 ---
np.eye(3)

# --- notebook cell 98 ---
B = np.array([[1,2,3], [4, 5, 6]])
B

# --- notebook cell 99 ---
A

# --- notebook cell 100 ---
A + B

# --- notebook cell 102 ---
B + A

# --- notebook cell 104 ---
C = np.array([[100,200,300], [400, 500, 600]])

A + (B + C)

# --- notebook cell 105 ---
(A + B) + C

# --- notebook cell 107 ---
2 * A

# --- notebook cell 109 ---
A * 2

# --- notebook cell 111 ---
2 * (3 * A)

# --- notebook cell 112 ---
(2 * 3) * A

# --- notebook cell 114 ---
2 * (A + B)

# --- notebook cell 115 ---
2 * A + 2 * B

# --- notebook cell 118 ---
D = np.array([
        [ 2,  3,  5,  7],
        [11, 13, 17, 19],
        [23, 29, 31, 37]
    ])
E = A.dot(D)
E

# --- notebook cell 120 ---
40*5 + 50*17 + 60*31

# --- notebook cell 121 ---
E[1,2]  # row 2, column 3

# --- notebook cell 123 ---
try:
    D.dot(A)
except ValueError as e:
    print("ValueError:", e)

# --- notebook cell 125 ---
F = np.array([
        [5,2],
        [4,1],
        [9,3]
    ])
A.dot(F)

# --- notebook cell 126 ---
F.dot(A)

# --- notebook cell 128 ---
G = np.array([
        [8,  7,  4,  2,  5],
        [2,  5,  1,  0,  5],
        [9, 11, 17, 21,  0],
        [0,  1,  0,  1,  2]])
A.dot(D).dot(G)     # (AB)G

# --- notebook cell 129 ---
A.dot(D.dot(G))     # A(BG)

# --- notebook cell 131 ---
(A + B).dot(D)

# --- notebook cell 132 ---
A.dot(D) + B.dot(D)

# --- notebook cell 134 ---
A.dot(np.eye(3))

# --- notebook cell 135 ---
np.eye(2).dot(A)

# --- notebook cell 137 ---
A * B   # NOT a matrix multiplication

# --- notebook cell 139 ---
import sys
print("Python version: {}.{}.{}".format(*sys.version_info))
print("Numpy version:", np.version.version)

# Uncomment the following line if your Python version is ≥3.5
# and your NumPy version is ≥1.10:

#A @ D

# --- notebook cell 142 ---
A

# --- notebook cell 143 ---
A.T

# --- notebook cell 145 ---
A.T.T

# --- notebook cell 147 ---
(A + B).T

# --- notebook cell 148 ---
A.T + B.T

# --- notebook cell 150 ---
(A.dot(D)).T

# --- notebook cell 151 ---
D.T.dot(A.T)

# --- notebook cell 153 ---
D.dot(D.T)

# --- notebook cell 155 ---
u

# --- notebook cell 156 ---
u.T

# --- notebook cell 158 ---
u_row = np.array([u])
u_row

# --- notebook cell 160 ---
u[np.newaxis, :]

# --- notebook cell 162 ---
u[np.newaxis]

# --- notebook cell 164 ---
u[None]

# --- notebook cell 166 ---
u_row.T

# --- notebook cell 168 ---
u[:, np.newaxis]

# --- notebook cell 170 ---
P = np.array([
        [3.0, 4.0, 1.0, 4.6],
        [0.2, 3.5, 2.0, 0.5]
    ])
x_coords_P, y_coords_P = P
plt.scatter(x_coords_P, y_coords_P)
plt.axis([0, 5, 0, 4])
plt.show()

# --- notebook cell 172 ---
plt.plot(x_coords_P, y_coords_P, "bo")
plt.plot(x_coords_P, y_coords_P, "b--")
plt.axis([0, 5, 0, 4])
plt.grid()
plt.show()

# --- notebook cell 174 ---
from matplotlib.patches import Polygon
plt.gca().add_artist(Polygon(P.T))
plt.axis([0, 5, 0, 4])
plt.grid()
plt.show()

# --- notebook cell 177 ---
H = np.array([
        [ 0.5, -0.2, 0.2, -0.1],
        [ 0.4,  0.4, 1.5, 0.6]
    ])
P_moved = P + H

plt.gca().add_artist(Polygon(P.T, alpha=0.2))
plt.gca().add_artist(Polygon(P_moved.T, alpha=0.3, color="r"))
for vector, origin in zip(H.T, P.T):
    plot_vector2d(vector, origin=origin)

plt.text(2.2, 1.8, "$P$", color="b", fontsize=18)
plt.text(2.0, 3.2, "$P+H$", color="r", fontsize=18)
plt.text(2.5, 0.5, "$H_{*,1}$", color="k", fontsize=18)
plt.text(4.1, 3.5, "$H_{*,2}$", color="k", fontsize=18)
plt.text(0.4, 2.6, "$H_{*,3}$", color="k", fontsize=18)
plt.text(4.4, 0.2, "$H_{*,4}$", color="k", fontsize=18)

plt.axis([0, 5, 0, 4])
plt.grid()
plt.show()

# --- notebook cell 179 ---
H2 = np.array([
        [-0.5, -0.5, -0.5, -0.5],
        [ 0.4,  0.4,  0.4,  0.4]
    ])
P_translated = P + H2

plt.gca().add_artist(Polygon(P.T, alpha=0.2))
plt.gca().add_artist(Polygon(P_translated.T, alpha=0.3, color="r"))
for vector, origin in zip(H2.T, P.T):
    plot_vector2d(vector, origin=origin)

plt.axis([0, 5, 0, 4])
plt.grid()
plt.show()

# --- notebook cell 181 ---
P + [[-0.5], [0.4]]  # same as P + H2, thanks to NumPy broadcasting

# --- notebook cell 183 ---
def plot_transformation(P_before, P_after, text_before, text_after, axis = [0, 5, 0, 4], arrows=False):
    if arrows:
        for vector_before, vector_after in zip(P_before.T, P_after.T):
            plot_vector2d(vector_before, color="blue", linestyle="--")
            plot_vector2d(vector_after, color="red", linestyle="-")
    plt.gca().add_artist(Polygon(P_before.T, alpha=0.2))
    plt.gca().add_artist(Polygon(P_after.T, alpha=0.3, color="r"))
    plt.text(P_before[0].mean(), P_before[1].mean(), text_before, fontsize=18, color="blue")
    plt.text(P_after[0].mean(), P_after[1].mean(), text_after, fontsize=18, color="red")
    plt.axis(axis)
    plt.grid()

P_rescaled = 0.60 * P
plot_transformation(P, P_rescaled, "$P$", "$0.6 P$", arrows=True)
plt.show()

# --- notebook cell 185 ---
U = np.array([[1, 0]])

# --- notebook cell 187 ---
U.dot(P)

# --- notebook cell 189 ---
def plot_projection(U, P):
    U_P = U.dot(P)
    
    axis_end = 100 * U
    plot_vector2d(axis_end[0], color="black")

    plt.gca().add_artist(Polygon(P.T, alpha=0.2))
    for vector, proj_coordinate in zip(P.T, U_P.T):
        proj_point = proj_coordinate * U
        plt.plot(proj_point[0][0], proj_point[0][1], "ro")
        plt.plot([vector[0], proj_point[0][0]], [vector[1], proj_point[0][1]], "r--")

    plt.axis([0, 5, 0, 4])
    plt.grid()
    plt.show()

plot_projection(U, P)

# --- notebook cell 191 ---
angle30 = 30 * np.pi / 180  # angle in radians
U_30 = np.array([[np.cos(angle30), np.sin(angle30)]])

plot_projection(U_30, P)

# --- notebook cell 194 ---
angle120 = 120 * np.pi / 180
V = np.array([
        [np.cos(angle30), np.sin(angle30)],
        [np.cos(angle120), np.sin(angle120)]
    ])
V

# --- notebook cell 196 ---
V.dot(P)

# --- notebook cell 198 ---
P_rotated = V.dot(P)
plot_transformation(P, P_rotated, "$P$", "$VP$", [-2, 6, -2, 4], arrows=True)
plt.show()

# --- notebook cell 201 ---
F_shear = np.array([
        [1, 1.5],
        [0, 1]
    ])
plot_transformation(P, F_shear.dot(P), "$P$", "$F_{shear} P$",
                    axis=[0, 10, 0, 7])
plt.show()

# --- notebook cell 203 ---
Square = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0]
    ])
plot_transformation(Square, F_shear.dot(Square), "$Square$", "$F_{shear} Square$",
                    axis=[0, 2.6, 0, 1.8])
plt.show()

# --- notebook cell 205 ---
F_squeeze = np.array([
        [1.4, 0],
        [0, 1/1.4]
    ])
plot_transformation(P, F_squeeze.dot(P), "$P$", "$F_{squeeze} P$",
                    axis=[0, 7, 0, 5])
plt.show()

# --- notebook cell 207 ---
plot_transformation(Square, F_squeeze.dot(Square), "$Square$", "$F_{squeeze} Square$",
                    axis=[0, 1.8, 0, 1.2])
plt.show()

# --- notebook cell 209 ---
F_reflect = np.array([
        [1, 0],
        [0, -1]
    ])
plot_transformation(P, F_reflect.dot(P), "$P$", "$F_{reflect} P$",
                    axis=[-2, 9, -4.5, 4.5])
plt.show()

# --- notebook cell 211 ---
F_inv_shear = np.array([
    [1, -1.5],
    [0, 1]
])
P_sheared = F_shear.dot(P)
P_unsheared = F_inv_shear.dot(P_sheared)
plot_transformation(P_sheared, P_unsheared, "$P_{sheared}$", "$P_{unsheared}$",
                    axis=[0, 10, 0, 7])
plt.plot(P[0], P[1], "b--")
plt.show()

# --- notebook cell 213 ---
F_inv_shear = LA.inv(F_shear)
F_inv_shear

# --- notebook cell 215 ---
plt.plot([0, 0, 1, 1, 0, 0.1, 0.1, 0, 0.1, 1.1, 1.0, 1.1, 1.1, 1.0, 1.1, 0.1],
         [0, 1, 1, 0, 0, 0.1, 1.1, 1.0, 1.1, 1.1, 1.0, 1.1, 0.1, 0, 0.1, 0.1],
         "r-")
plt.axis([-0.5, 2.1, -0.5, 1.5])
plt.show()

# --- notebook cell 217 ---
F_project = np.array([
        [1, 0],
        [0, 0]
    ])
plot_transformation(P, F_project.dot(P), "$P$", "$F_{project} \cdot P$",
                    axis=[0, 6, -1, 4])
plt.show()

# --- notebook cell 219 ---
try:
    LA.inv(F_project)
except LA.LinAlgError as e:
    print("LinAlgError:", e)

# --- notebook cell 221 ---
angle30 = 30 * np.pi / 180
F_project_30 = np.array([
               [np.cos(angle30)**2, np.sin(2*angle30)/2],
               [np.sin(2*angle30)/2, np.sin(angle30)**2]
         ])
plot_transformation(P, F_project_30.dot(P), "$P$", "$F_{project\_30} \cdot P$",
                    axis=[0, 6, -1, 4])
plt.show()

# --- notebook cell 223 ---
LA.inv(F_project_30)

# --- notebook cell 225 ---
F_shear.dot(LA.inv(F_shear))

# --- notebook cell 227 ---
LA.inv(LA.inv(F_shear))

# --- notebook cell 229 ---
F_involution  = np.array([
        [0, -2],
        [-1/2, 0]
    ])
plot_transformation(P, F_involution.dot(P), "$P$", "$F_{involution} \cdot P$",
                    axis=[-8, 5, -4, 4])
plt.show()

# --- notebook cell 231 ---
F_reflect.dot(F_reflect.T)

# --- notebook cell 234 ---
M = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ])
LA.det(M)

# --- notebook cell 236 ---
LA.det(F_project)

# --- notebook cell 238 ---
LA.det(F_project_30)

# --- notebook cell 240 ---
LA.det(F_shear)

# --- notebook cell 243 ---
F_scale = np.array([
        [0.5, 0],
        [0, 0.5]
    ])
plot_transformation(P, F_scale.dot(P), "$P$", "$F_{scale} \cdot P$",
                    axis=[0, 6, -1, 4])
plt.show()

# --- notebook cell 245 ---
LA.det(F_scale)

# --- notebook cell 247 ---
LA.det(F_reflect)

# --- notebook cell 249 ---
P_squeezed_then_sheared = F_shear.dot(F_squeeze.dot(P))

# --- notebook cell 251 ---
P_squeezed_then_sheared = (F_shear.dot(F_squeeze)).dot(P)

# --- notebook cell 253 ---
F_squeeze_then_shear = F_shear.dot(F_squeeze)
P_squeezed_then_sheared = F_squeeze_then_shear.dot(P)

# --- notebook cell 256 ---
LA.inv(F_shear.dot(F_squeeze)) == LA.inv(F_squeeze).dot(LA.inv(F_shear))

# --- notebook cell 258 ---
U, S_diag, V_T = LA.svd(F_shear) # note: in python 3 you can rename S_diag to Σ_diag
U

# --- notebook cell 259 ---
S_diag

# --- notebook cell 261 ---
S = np.diag(S_diag)
S

# --- notebook cell 263 ---
U.dot(np.diag(S_diag)).dot(V_T)

# --- notebook cell 264 ---
F_shear

# --- notebook cell 266 ---
plot_transformation(Square, V_T.dot(Square), "$Square$", "$V^T \cdot Square$",
                    axis=[-0.5, 3.5 , -1.5, 1.5])
plt.show()

# --- notebook cell 268 ---
plot_transformation(V_T.dot(Square), S.dot(V_T).dot(Square), "$V^T \cdot Square$", "$\Sigma \cdot V^T \cdot Square$",
                    axis=[-0.5, 3.5 , -1.5, 1.5])
plt.show()

# --- notebook cell 270 ---
plot_transformation(S.dot(V_T).dot(Square), U.dot(S).dot(V_T).dot(Square),"$\Sigma \cdot V^T \cdot Square$", "$U \cdot \Sigma \cdot V^T \cdot Square$",
                    axis=[-0.5, 3.5 , -1.5, 1.5])
plt.show()

# --- notebook cell 273 ---
eigenvalues, eigenvectors = LA.eig(F_squeeze)
eigenvalues # [λ0, λ1, …]

# --- notebook cell 274 ---
eigenvectors # [v0, v1, …]

# --- notebook cell 276 ---
eigenvalues2, eigenvectors2 = LA.eig(F_shear)
eigenvalues2 # [λ0, λ1, …]

# --- notebook cell 277 ---
eigenvectors2 # [v0, v1, …]

# --- notebook cell 280 ---
D = np.array([
        [100, 200, 300],
        [ 10,  20,  30],
        [  1,   2,   3],
    ])
np.trace(D)

# --- notebook cell 282 ---
np.trace(F_project)