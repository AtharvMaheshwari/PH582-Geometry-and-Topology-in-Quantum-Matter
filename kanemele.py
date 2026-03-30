
# Assignment 2 : Kane-Mele Model
# Name: Atharv Maheshwari
# Roll No. 23b1805
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.widgets import Slider

#defining parameters for kane-mele model

# nearest neighbour vectors
a1 = np.array([0, -1])
a2 = np.array([np.sqrt(3)/2, 1/2])
a3 = np.array([-np.sqrt(3)/2, 1/2])

avec = [a1, a2, a3]

# next nearest neighbour vectors
b1 = a2 - a3
b2 = a3 - a1
b3 = a1 - a2

bvec = [b1, b2, b3]

# hopping parameters
lambda_so = 0.1   # S-O coupling term
lambda_r_0 = 0.3    # Rashba Term # lambda_r
lambda_v = 0.2    # Staggered Sublattice Potential

t1 = 1
t2_0 = lambda_so
delta_0 = lambda_v

# I am doing the following to index the BZ with non orthogonal vectors so that I do not have to use the mask

a1_modified = np.array([0, -1,0])
a2_modified = np.array([np.sqrt(3)/2, 1/2,0])
a3_modified = np.array([-np.sqrt(3)/2, 1/2,0])

avec = [a1_modified, a2_modified, a3_modified]

# next nearest neighbour vectors
b1_modified = a2_modified - a3_modified
b2_modified = a3_modified - a1_modified
b3_modified = a1_modified - a2_modified

bvec = [b1_modified, b2_modified, b3_modified]

z_unit=np.array([0,0,1])
k1_vec=2*np.pi*(np.cross(-b2_modified,z_unit)/np.abs(np.dot(b1_modified,np.cross(-b2_modified,z_unit))))
k2_vec=2*np.pi*(np.cross(z_unit,b1_modified)/np.abs(np.dot(b1_modified,np.cross(-b2_modified,z_unit))))

def k_vec(Nk, n1, n2):
    return (n1 * k1_vec + n2 * k2_vec) / Nk

def H_noRashba(Nk, n1, n2, delta, t2):
    k=k_vec(Nk,n1,n2)
    kx=k[0]
    ky=k[1]

    S1,S2,S3=0.0,0.0,0.0

    for a in avec:
         ka=kx*a[0]+ky*a[1]
         S1+=(np.cos(ka))
         S2+=(np.sin(ka))
         
    for b in bvec:
        S3+=(np.sin(kx*b[0]+ky*b[1]))
    
    dx=t1*S1
    dy=t1*S2
    dz_up=delta-2*t2*(S3)
    dz_down=delta+2*t2*(S3)

    return np.array([
        [dz_up , dx-1j*dy, 0, 0 ],
        [dx+1j*dy , -dz_up, 0, 0 ],
        [0, 0, dz_down, dx-1j*dy ],
        [0, 0, dx+1j*dy, -dz_down ]
        ]
        , dtype=complex)

def H_upleft(H_noRashba_0):
    return np.array([
        [H_noRashba_0[0,0],H_noRashba_0[0,1]],
        [H_noRashba_0[1,0],H_noRashba_0[1,1]]
    ])

def H_downright(H_noRashba_0):
    return np.array([
        [H_noRashba_0[2,2],H_noRashba_0[2,3]],
        [H_noRashba_0[3,2],H_noRashba_0[3,3]]
    ])


Nk=200
N1,N2=np.linspace(0,Nk-1,Nk),np.linspace(0,Nk-1,Nk)

k_grid = np.zeros((Nk, Nk, 3))
kx_vals = np.zeros((Nk, Nk))
ky_vals = np.zeros((Nk, Nk))

for i in range(Nk):
    for j in range(Nk):
        k_grid[i, j, :] = k_vec(Nk, N1[i], N2[j])
        kx_vals[i, j] = k_grid[i, j, 0]
        ky_vals[i, j] = k_grid[i, j, 1]

E1_up = np.zeros((Nk,Nk))
E2_up = np.zeros((Nk,Nk))
E1_down = np.zeros((Nk,Nk))
E2_down = np.zeros((Nk,Nk))


plt.scatter(kx_vals,ky_vals, s=0.1)


for i in range(Nk):
    for j in range(Nk):
        Hij=H_noRashba(Nk, N1[i], N2[j], delta_0, t2_0)
        w_up,_ = np.linalg.eigh(H_upleft(Hij))
        w_down,_ = np.linalg.eigh(H_downright(Hij))
        E1_up[i,j]=w_up[0]
        E2_up[i,j]=w_up[1]
        E1_down[i,j]=w_down[0]
        E2_down[i,j]=w_down[1]


fig = plt.figure(figsize=(20, 12))

vmin = min(E1_up.min(), E2_up.min())
vmax = max(E1_up.max(), E2_up.max())

# ---- Surface for E1 ----
ax1 = fig.add_subplot(121, projection='3d')
surf1 = ax1.plot_surface(kx_vals, ky_vals, E1_up, cmap='viridis', vmin=vmin, vmax=vmax)
surf2 = ax1.plot_surface(kx_vals, ky_vals, E2_up, cmap='viridis', vmin=vmin, vmax=vmax)
ax1.set_title('Energy bands for up-spin')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel("E")

ax1.view_init(elev=15, azim=40) # VIEWING ANGLE
fig.colorbar(surf1, ax=ax1, shrink=0.5)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(20,12))

vmin = min(E1_down.min(), E2_down.min())
vmax = max(E1_down.max(), E2_down.max())

# ---- Surface for E1 ----
ax1 = fig.add_subplot(121, projection='3d')
surf3 = ax1.plot_surface(kx_vals, ky_vals, E1_down, cmap='magma', vmin=vmin, vmax=vmax)
surf4 = ax1.plot_surface(kx_vals, ky_vals, E2_down, cmap='magma', vmin=vmin, vmax=vmax)
ax1.set_title('Energy bands for down-spin')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('E')

ax1.view_init(elev=15, azim=40) # VIEWING ANGLE
fig.colorbar(surf3, ax=ax1, shrink=0.5)
plt.tight_layout()
plt.show()

%matplotlib widget

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')

# ---- SPIN UP (purple) ----
ax.plot_surface(kx_vals, ky_vals, E1_up, cmap='Purples',
                edgecolor='k', linewidth=0.1, alpha=0.9)

ax.plot_surface(kx_vals, ky_vals, E2_up, cmap='Purples_r',
                edgecolor='k', linewidth=0.1, alpha=0.9)

# ---- SPIN DOWN (red) ----
ax.plot_surface(kx_vals, ky_vals, E1_down, cmap='Reds',
                edgecolor='k', linewidth=0.1, alpha=0.5)

ax.plot_surface(kx_vals, ky_vals, E2_down, cmap='Reds_r',
                edgecolor='k', linewidth=0.1, alpha=0.5)

# labels
ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_ylabel(r'$k_y$', fontsize=18)
ax.set_zlabel(r'$E$', fontsize=18)

# initial view
init_elev = 15
init_azim = 40
ax.view_init(elev=init_elev, azim=init_azim)

plt.title(r"Kane-Mele: Spin-resolved bands", fontsize=24)

# ---- SLIDER AXES ----
ax_elev = plt.axes([0.2, 0.05, 0.6, 0.02])
ax_azim = plt.axes([0.2, 0.01, 0.6, 0.02])

slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=init_elev)
slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=init_azim)

# ---- UPDATE FUNCTION ----
def update(val):
    ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
    fig.canvas.draw_idle()

slider_elev.on_changed(update)
slider_azim.on_changed(update)

plt.tight_layout()
plt.show()

%matplotlib inline

# As the k vectors are not orthogonal, we will have to do hamiltonian derivative along indices n1 and n2(non orthogonal directions) and not kx and ky, and then use appropriate areas evaluate using cross products 


def berry_curvature_2band_n(H_func, Nk, n1, n2, delta, t2, spin="up"):
    
    # periodic wrapping
    def wrap(x):
        return x % Nk

    # select block directly
    if spin == "up":
        H = H_upleft(H_func(Nk, n1, n2, delta, t2))  # 2x2 block
        E, U = np.linalg.eigh(H)

        # finite differences in (n1,n2) lattice directions
        H1 = (H_upleft(H_func(Nk, wrap(n1+1), n2, delta, t2)) -
            H_upleft(H_func(Nk, wrap(n1-1), n2, delta, t2))) / 2

        H2 = (H_upleft(H_func(Nk, n1, wrap(n2+1), delta, t2)) -
            H_upleft(H_func(Nk, n1, wrap(n2-1), delta, t2))) / 2

    elif spin == "down":
        H = H_downright(H_func(Nk, n1, n2, delta, t2))  # 2x2 block
        E, U = np.linalg.eigh(H)

        # finite differences in (n1,n2) lattice directions
        H1 = (H_downright(H_func(Nk, wrap(n1+1), n2, delta, t2)) -
            H_downright(H_func(Nk, wrap(n1-1), n2, delta, t2))) / 2

        H2 = (H_downright(H_func(Nk, n1, wrap(n2+1), delta, t2)) -
            H_downright(H_func(Nk, n1, wrap(n2-1), delta, t2))) / 2


    F = np.zeros(2)
    for n in range(2):
        for m in range(2):
            if m != n:
                num = np.vdot(U[:, n], H1 @ U[:, m]) * \
                      np.vdot(U[:, m], H2 @ U[:, n])
                F[n] += -2 * np.imag(num) / (E[m] - E[n])**2

    # scalar Jacobian, to make it according to kx ky space and vectors are non orthogonal (remultiplied later when integration is done)
    # The problem is that k vectors k1 and k2 are not unit vectors and the area occupied by one grid point is not unity, so we need to take that fact into accout while taking derivative and when we remultiple in kx ky space we take unit vectors, this problem is solved by the Jacobian
    J = np.linalg.norm(np.cross(k1_vec, k2_vec))

    return F / J


def compute_chern_by_sum(H_func, Nk, delta, t2, spin="up"):

    chern = np.zeros(2)

    # area element
    J = np.linalg.norm(np.cross(k1_vec, k2_vec))

    #the berry curvature integration area element is J/(Nk**2) which is contribution of element Fij, so we have to multiply each element by J to get total sum to be J.  
    for i in range(Nk):
        for j in range(Nk):

            F = berry_curvature_2band_n(H_func, Nk, i, j, delta, t2, spin)

            chern += F
    chern *= J
    chern /= (2 * np.pi)

    return chern


chern_down=compute_chern_by_sum(H_noRashba, Nk, delta_0,t2_0,"down")
chern_up=compute_chern_by_sum(H_noRashba, Nk, delta_0,t2_0,"up")


print(f"For Down Spin:")
labels = ["Lower band", "Upper band"]
for i, val in enumerate(chern_down):
    print(f"{labels[i]} Chern number: {val:.4f} (~ {int(np.rint(val))})")

print(f" ")
print(f"For Up Spin:")
labels = ["Lower band", "Upper band"]
for i, val in enumerate(chern_up):
    print(f"{labels[i]} Chern number: {val:.4f} (~ {int(np.rint(val))})")

# Just for Visualization, plotting with kx ky basis as done in Haldane Model Project

def H_noRashba_plot(kx, ky, delta, t2):
    
    S1,S2,S3=0.0,0.0,0.0

    for a in avec:
         ka=kx*a[0]+ky*a[1]
         S1+=(np.cos(ka))
         S2+=(np.sin(ka))
         
    for b in bvec:
        S3+=(np.sin(kx*b[0]+ky*b[1]))
    
    dx=t1*S1
    dy=t1*S2
    dz_up=delta-2*t2*(S3)
    dz_down=delta+2*t2*(S3)

    return np.array([
        [dz_up , dx-1j*dy, 0, 0 ],
        [dx+1j*dy , -dz_up, 0, 0 ],
        [0, 0, dz_down, dx-1j*dy ],
        [0, 0, dx+1j*dy, -dz_down ]
        ]
        , dtype=complex)

def H_upleft(H_noRashba_0):
    return np.array([
        [H_noRashba_0[0,0],H_noRashba_0[0,1]],
        [H_noRashba_0[1,0],H_noRashba_0[1,1]]
    ])

def H_downright(H_noRashba_0):
    return np.array([
        [H_noRashba_0[2,2],H_noRashba_0[2,3]],
        [H_noRashba_0[3,2],H_noRashba_0[3,3]]
    ])


H_noRashba_plot(0.5, 0.5, delta_0, t2_0)


kx_vals_plot=np.linspace(-np.pi,np.pi,201)
ky_vals_plot=np.linspace(-np.pi,np.pi,201)

X,Y=np.meshgrid(kx_vals_plot, ky_vals_plot)

E1_up_plot=np.zeros_like(X)
E2_up_plot=np.zeros_like(X)
E1_down_plot=np.zeros_like(X)
E2_down_plot=np.zeros_like(X)


for i in range(len(kx_vals_plot)):
    for j in range(len(ky_vals_plot)):
        Hij=H_noRashba_plot(X[i,j],Y[i,j], delta_0, t2_0)
        w_up,_ = np.linalg.eigh(H_upleft(Hij))
        w_down,_ = np.linalg.eigh(H_downright(Hij))
        E1_up_plot[i,j]=w_up[0]
        E2_up_plot[i,j]=w_up[1]
        E1_down_plot[i,j]=w_down[0]
        E2_down_plot[i,j]=w_down[1]


fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, projection='3d')

vmin = min(E1_up_plot.min(), E2_up_plot.min())
vmax = max(E1_up_plot.max(), E2_up_plot.max())

# ---- SPIN UP (purple) ----
# upper band
surf1 = ax.plot_surface(X, Y, E1_up_plot, cmap='viridis', edgecolor='k', linewidth=0.15, rstride=2, cstride=2, alpha=0.9, vmin=vmin, vmax=vmax)

# lower band
surf2 = ax.plot_surface(X, Y, E2_up_plot, cmap='viridis', edgecolor='k', linewidth=0.15, rstride=2, cstride=2, alpha=0.9, vmin=vmin, vmax=vmax)

# labels
ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_ylabel(r'$k_y$', fontsize=18)
ax.set_zlabel(r'$E$', fontsize=18)
fig.colorbar(surf1, ax=ax, shrink=0.5)
ax.view_init(elev=10, azim=35)

plt.title('Energy bands for up-spin', fontsize=24)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111, projection='3d')

vmin = min(E1_down_plot.min(), E2_down_plot.min())
vmax = max(E1_down_plot.max(), E2_down_plot.max())

# ---- SPIN down (purple) ----
# upper band
surf1 = ax.plot_surface(X, Y, E1_down_plot, cmap='magma', edgecolor='k', linewidth=0.15, rstride=2, cstride=2, alpha=0.9, vmin=vmin, vmax=vmax)

# lower band
surf2 = ax.plot_surface(X, Y, E2_down_plot, cmap='magma', edgecolor='k', linewidth=0.15, rstride=2, cstride=2, alpha=0.9, vmin=vmin, vmax=vmax)

# labels
ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_ylabel(r'$k_y$', fontsize=18)
ax.set_zlabel(r'$E$', fontsize=18)

ax.view_init(elev=10, azim=35)
fig.colorbar(surf1, ax=ax, shrink=0.5)

plt.title('Energy bands for down-spin', fontsize=24)

plt.tight_layout()
plt.show()

def berry_curvature_2band(Hk_func, kx, ky, dk):

    H = Hk_func(kx, ky)
    E, U = np.linalg.eigh(H)

    # derivatives
    Hx = (Hk_func(kx+dk, ky) - Hk_func(kx-dk, ky)) / (2*dk)
    Hy = (Hk_func(kx, ky+dk) - Hk_func(kx, ky-dk)) / (2*dk)

    F = np.zeros(2)

    for n in range(2):
        for m in range(2):
            if m != n:
                num = np.vdot(U[:,n], Hx @ U[:,m]) * \
                      np.vdot(U[:,m], Hy @ U[:,n])
                F[n] += -2*np.imag(num) / (E[m]-E[n])**2

    return F

def berry_curvature_spin(kx, ky, dk, spin="up"):

    # select block directly
    if spin == "up":
        H  = H_upleft(H_noRashba_plot(kx, ky, delta_0, t2_0))
        Hx = (H_upleft(H_noRashba_plot(kx+dk, ky, delta_0, t2_0)) -
              H_upleft(H_noRashba_plot(kx-dk, ky, delta_0, t2_0))) / (2*dk)
        Hy = (H_upleft(H_noRashba_plot(kx, ky+dk, delta_0, t2_0)) -
              H_upleft(H_noRashba_plot(kx, ky-dk, delta_0, t2_0))) / (2*dk)

    elif spin == "down":
        H  = H_downright(H_noRashba_plot(kx, ky, delta_0, t2_0))
        Hx = (H_downright(H_noRashba_plot(kx+dk, ky, delta_0, t2_0)) -
              H_downright(H_noRashba_plot(kx-dk, ky, delta_0, t2_0))) / (2*dk)
        Hy = (H_downright(H_noRashba_plot(kx, ky+dk, delta_0, t2_0)) -
              H_downright(H_noRashba_plot(kx, ky-dk, delta_0, t2_0))) / (2*dk)

    E, U = np.linalg.eigh(H)

    F = np.zeros(2)
    
    for n in range(2):
        for m in range(2):
            if m != n:
                num = np.vdot(U[:,n], Hx @ U[:,m]) * \
                      np.vdot(U[:,m], Hy @ U[:,n])
                F[n] += -2*np.imag(num) / (E[m]-E[n])**2

    return F


Nk=201
dk=2*np.pi/(Nk-1)
F_up_lower = np.zeros((Nk, Nk))
F_up_upper = np.zeros((Nk, Nk))

F_down_lower = np.zeros((Nk, Nk))
F_down_upper = np.zeros((Nk, Nk))

X, Y = np.meshgrid(kx_vals_plot, ky_vals_plot)

for i in range(Nk):
    for j in range(Nk):

        F_up = berry_curvature_spin(X[i,j], Y[i,j], dk, "up")
        F_down = berry_curvature_spin(X[i,j], Y[i,j], dk, "down")

        F_up_lower[i,j] = F_up[0]
        F_up_upper[i,j] = F_up[1]

        F_down_lower[i,j] = F_down[0]
        F_down_upper[i,j] = F_down[1]


maxF = max(
    np.abs(F_up_lower).max(), np.abs(F_up_upper).max(),
    np.abs(F_down_lower).max(), np.abs(F_down_upper).max()
)

norm = colors.TwoSlopeNorm(vmin=-maxF, vcenter=0, vmax=maxF)

plt.figure(figsize=(8,5))

# SPIN UP
plt.subplot(2,2,1)
plt.imshow(F_up_lower, origin="lower",
           extent=[-np.pi,np.pi,-np.pi,np.pi],
           cmap="Purples", norm=norm)
plt.title("Spin ↑ (lower)")
plt.colorbar()

plt.subplot(2,2,2)
plt.imshow(F_up_upper, origin="lower",
           extent=[-np.pi,np.pi,-np.pi,np.pi],
           cmap="Purples", norm=norm)
plt.title("Spin ↑ (upper)")
plt.colorbar()

# SPIN DOWN
plt.subplot(2,2,3)
plt.imshow(F_down_lower, origin="lower",
           extent=[-np.pi,np.pi,-np.pi,np.pi],
           cmap="Reds", norm=norm)
plt.title("Spin ↓ (lower)")
plt.colorbar()

plt.subplot(2,2,4)
plt.imshow(F_down_upper, origin="lower",
           extent=[-np.pi,np.pi,-np.pi,np.pi],
           cmap="Reds", norm=norm)
plt.title("Spin ↓ (upper)")
plt.colorbar()

plt.tight_layout()

plt.show()


fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection='3d')

scale = 2   # amplify curvature contrast

maxF = max(abs(F_up_lower.min()), abs(F_up_lower.max()))

norm = colors.TwoSlopeNorm(
    vmin=-maxF/scale,
    vcenter=0,
    vmax=maxF/scale
)

surf1 = ax.plot_surface(
    X, Y, E1_up_plot,
    facecolors=plt.cm.coolwarm(norm(F_up_lower)),
    edgecolor='none',
    antialiased=True,
    rstride=1,
    cstride=1
)

surf2 = ax.plot_surface(
    X, Y, E2_up_plot,
    facecolors=plt.cm.coolwarm(norm(F_up_upper)),
    edgecolor='none',
    antialiased=True,
    rstride=1,
    cstride=1
)

ax.set_xlabel("$k_x$")
ax.set_ylabel("$k_y$")
ax.set_zlabel("Energy")

# VIEWING ANGLE
ax.view_init(elev=10, azim=40)

# better aspect ratio
ax.set_box_aspect((1,1,0.6))

plt.title("Band structure colored by Berry curvature")

plt.show()



def H_withRashba(Nk, n1, n2, delta, t2, lambda_r):
    k=k_vec(Nk,n1,n2)
    kx=k[0]
    ky=k[1]

    S1,S2,S3=0.0,0.0,0.0

    for a in avec:
         ka=kx*a[0]+ky*a[1]
         S1+=(np.cos(ka))
         S2+=(np.sin(ka))
         
    for b in bvec:
        S3+=(np.sin(kx*b[0]+ky*b[1]))
    
    dx=t1*S1
    dy=t1*S2
    dz_up=delta-2*t2*(S3)
    dz_down=delta+2*t2*(S3)

    star1= (1j*lambda_r/2)*(-np.exp(1j*ky)+(np.exp(1j*ky/2))*(np.cos(np.sqrt(3)*kx/2)+np.sqrt(3)*np.sin(np.sqrt(3)*kx/2)))
    star4=np.conj(star1)
    star3= (1j*lambda_r/2)*(-np.exp(1j*ky)+(np.exp(1j*ky/2))*(np.cos(np.sqrt(3)*kx/2)-np.sqrt(3)*np.sin(np.sqrt(3)*kx/2)))
    star2=np.conj(star3)
    

    return np.array([
        [dz_up , dx-1j*dy, 0, star1 ],
        [dx+1j*dy , -dz_up, star2, 0 ],
        [0, star3, dz_down, dx-1j*dy ],
        [star4, 0, dx+1j*dy, -dz_down ]
        ]
        , dtype=complex)


H=H_withRashba(100, 30, 30, delta_0, t2_0, lambda_r_0)


np.allclose(H, H.conj().T)


Nk=200
N1,N2=np.linspace(0,Nk-1,Nk),np.linspace(0,Nk-1,Nk)
# N1,N2=np.linspace(-200,200,401),np.linspace(-200,200,401)

k_grid = np.zeros((Nk, Nk, 3))
kx_vals = np.zeros((Nk, Nk))
ky_vals = np.zeros((Nk, Nk))

for i in range(Nk):
    for j in range(Nk):
        k_grid[i, j, :] = k_vec(Nk, N1[i], N2[j])
        kx_vals[i, j] = k_grid[i, j, 0]
        ky_vals[i, j] = k_grid[i, j, 1]

E1 = np.zeros((Nk,Nk))
E2 = np.zeros((Nk,Nk))
E3 = np.zeros((Nk,Nk))
E4 = np.zeros((Nk,Nk))


for i in range(Nk):
    for j in range(Nk):
        Hij=H_withRashba(Nk, N1[i], N2[j], delta_0, t2_0, lambda_r_0)
        w,_ = np.linalg.eigh(Hij)
        E1[i,j]=w[0]
        E2[i,j]=w[1]
        E3[i,j]=w[2]
        E4[i,j]=w[3]

%matplotlib widget

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(kx_vals, ky_vals, E1, cmap='Purples',
                edgecolor='k', linewidth=0.1, alpha=0.9)

ax.plot_surface(kx_vals, ky_vals, E2, cmap='Greens',
                edgecolor='k', linewidth=0.1, alpha=0.5)

ax.plot_surface(kx_vals, ky_vals, E3, cmap='Reds',
                edgecolor='k', linewidth=0.1, alpha=0.5)

ax.plot_surface(kx_vals, ky_vals, E4, cmap='Blues',
                edgecolor='k', linewidth=0.1, alpha=0.5)

# labels
ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_ylabel(r'$k_y$', fontsize=18)
ax.set_zlabel(r'$E$', fontsize=18)

# initial view
init_elev = 15
init_azim = 40
ax.view_init(elev=init_elev, azim=init_azim)

plt.title(r"Kane-Mele: Spin-resolved bands", fontsize=24)

# ---- SLIDER AXES ----
ax_elev = plt.axes([0.2, 0.05, 0.6, 0.02])
ax_azim = plt.axes([0.2, 0.01, 0.6, 0.02])

slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=init_elev)
slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=init_azim)

# ---- UPDATE FUNCTION ----
def update(val):
    ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
    fig.canvas.draw_idle()

slider_elev.on_changed(update)
slider_azim.on_changed(update)

plt.tight_layout()
plt.show()

%matplotlib inline

hbar=1.0#6.626*1e-34/(2*np.pi)

def spin_berry_curvature_2band(H_func, Nk, n1, n2, delta, t2, lambda_r, epsilon):

    def wrap(x):
        return x % Nk

    H = H_func(Nk, n1, n2, delta, t2, lambda_r)
    E, U = np.linalg.eigh(H)

    # velocity operators (finite difference)
    H1 = (H_func(Nk, wrap(n1+1), n2, delta, t2, lambda_r) -
          H_func(Nk, wrap(n1-1), n2, delta, t2, lambda_r)) / 2

    H2 = (H_func(Nk, n1, wrap(n2+1), delta, t2, lambda_r) -
          H_func(Nk, n1, wrap(n2-1), delta, t2, lambda_r)) / 2

    # --- define spin operator sz (modify basis if needed) ---
    sz = np.diag([1, -1, 1, -1])  # (A↑, A↓, B↑, B↓)

    # spin current operator: J_x^{sz} = 1/2 {sz, v_x}
    Jx_sz = 0.5 * (sz @ H1 + H1 @ sz)

    F = np.zeros(4)

    for n in range(4):
        for m in range(4):
            if m != n:
                num = np.vdot(U[:, n], Jx_sz @ U[:, m]) * \
                      np.vdot(U[:, m], H2 @ U[:, n])

                F[n] += 2 * np.imag(num) / (((E[n] - E[m])**2) + epsilon)

    # Jacobian correction
    J = np.linalg.norm(np.cross(k1_vec, k2_vec))

    return F / J


epsilon_0 = 1e-16
F_map = np.zeros((Nk, Nk, 4))
Ef=0
e_charge=1.0#1.6*1e-19

J = np.linalg.norm(np.cross(k1_vec, k2_vec))
dk1 = np.linalg.norm(k1_vec) / Nk
dk2 = np.linalg.norm(k2_vec) / Nk


def compute_spin_hall_conductivity(H_func, Nk, delta, t2, lambda_r, epsilon, return_Fmap=False):

    sigma_n = np.zeros(4)

    if return_Fmap:
        F_map = np.zeros((Nk, Nk, 4))

    # Jacobian (constant over grid)
    J = np.linalg.norm(np.cross(k1_vec, k2_vec))

    for i in range(Nk):
        for j in range(Nk):

            H = H_func(Nk, N1[i], N2[j], delta, t2, lambda_r)
            E, _ = np.linalg.eigh(H)

            F = spin_berry_curvature_2band(
                H_func, Nk, N1[i], N2[j],
                delta, t2, lambda_r, epsilon
            )

            if return_Fmap:
                F_map[i, j, :] = F

            for n in range(4):
                if E[n] < Ef:
                    sigma_n[n] += F[n]

    # --- prefactors ---
    sigma_n = sigma_n * (e_charge / hbar)

    # integration over BZ
    sigma_n *= (J / Nk**2) / (2*np.pi)**2

    sigma_total = np.sum(sigma_n)

    if return_Fmap:
        return sigma_n, sigma_total, F_map
    else:
        return sigma_n, sigma_total


sigma_n, sigma_total = compute_spin_hall_conductivity(H_withRashba, Nk, delta_0, t2_0, lambda_r_0,epsilon=1e-10)

print("Band SHC:", sigma_n)
print("Total SHC:", sigma_total)


eps_list = np.logspace(-2, -22, 11)  # from 1e-2 → 1e-12


sigma_total_list = []
sigma_band_list = []

for eps in eps_list:
    sigma_n, sigma_total = compute_spin_hall_conductivity(
        H_withRashba, Nk, delta_0, t2_0+0.1, lambda_r_0, epsilon=eps
    )
    
    sigma_total_list.append(sigma_total)
    sigma_band_list.append(sigma_n.copy())

sigma_total_list = np.array(sigma_total_list)
sigma_band_list = np.array(sigma_band_list)


plt.figure(figsize=(6,5))

plt.plot(eps_list, sigma_band_list[:,0], marker='o')

plt.xscale('log')   # log scale on X-axis
plt.xlabel(r'$\epsilon$')
plt.ylabel(r'$\sigma^{s_z}_{xy}$ (Band 0)')
plt.title('SHC vs epsilon (Band 0)')

plt.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()


epsilon_ideal=1e-8
t2_vals = np.linspace(0, 0.5, 11)
lambda_r_vals = np.linspace(0, 0.5, 11)

phase_map = np.zeros((len(t2_vals), len(lambda_r_vals)))
phase_map_1 = np.zeros((len(t2_vals), len(lambda_r_vals)))
phase_map_2 = np.zeros((len(t2_vals), len(lambda_r_vals)))

for i, t2 in enumerate(t2_vals):
    for j, lr in enumerate(lambda_r_vals):

        sigma_band, sigma_total = compute_spin_hall_conductivity(
            H_withRashba, Nk, delta_0, t2, lr, epsilon=epsilon_ideal
        )
        phase_map[i, j] = sigma_total
        phase_map_1[i, j] = sigma_band[0]
        phase_map_2[i, j] = sigma_band[1]


plt.figure(figsize=(8, 6))

# Create heatmap
plt.imshow(
    phase_map,
    aspect='auto',
    origin='lower',
    extent=[lambda_r_vals.min(), lambda_r_vals.max(),
            t2_vals.min(), t2_vals.max()],
    cmap='viridis'
)

# Colorbar
plt.colorbar(label='Phase')

# Labels and title
plt.xlabel('lambda_r')
plt.ylabel('lambda_so')
plt.title('Phase Map')

plt.tight_layout()
plt.show()


print("Band-resolved SHC: (scaled by h/e)")
for n in range(4):
    print(f"Band {n}: {sigma_n[n]}")


sigma_total = np.sum(sigma_n)
print("Total SHC:", sigma_total)


epsilon_ideal=1e-8
t2_vals = np.linspace(0, 0.3, 10)
sigma_total_plot=np.zeros_like(t2_vals)

for i, t2 in enumerate(t2_vals):

        sigma_band, sigma_total = compute_spin_hall_conductivity(
            H_withRashba, Nk, delta_0, t2, lambda_r_0, epsilon=epsilon_ideal
        )
        sigma_total_plot[i] = sigma_total


# Theoretical transition t2 value
t2_transition = lambda_r_0 / (2 * np.sqrt(3))

plt.figure(figsize=(7,5))

# Computed spin Hall conductivity
plt.plot(t2_vals, sigma_total_plot, marker='o', linestyle='-', color='b', label='Computed sigma_total')

# Vertical line at theoretical t2
plt.axvline(x=t2_transition, color='r', linestyle='--', label=r'Theoretical $t_2 = \lambda_r / 2\sqrt{3}$')

plt.xlabel('t2')
plt.ylabel('Total Spin Hall Conductivity sigma_total')
plt.title('Spin Hall Conductivity vs t2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


def spin_berry_curvature_2band_no(H_func, Nk, n1, n2, delta, t2, epsilon):

    def wrap(x):
        return x % Nk

    H = H_func(Nk, n1, n2, delta, t2)
    E, U = np.linalg.eigh(H)

    # velocity operators (finite difference)
    H1 = (H_func(Nk, wrap(n1+1), n2, delta, t2) -
          H_func(Nk, wrap(n1-1), n2, delta, t2)) / 2

    H2 = (H_func(Nk, n1, wrap(n2+1), delta, t2) -
          H_func(Nk, n1, wrap(n2-1), delta, t2)) / 2

    # --- define spin operator sz (modify basis if needed) ---
    sz = np.diag([1, -1, 1, -1])  # (A↑, A↓, B↑, B↓)

    # spin current operator: J_x^{sz} = 1/2 {sz, v_x}
    Jx_sz = 0.5 * (sz @ H1 + H1 @ sz)

    F = np.zeros(4)

    for n in range(4):
        for m in range(4):
            if m != n:
                num = np.vdot(U[:, n], Jx_sz @ U[:, m]) * \
                      np.vdot(U[:, m], H2 @ U[:, n])

                F[n] += 2 * np.imag(num) / (((E[n] - E[m])**2) + epsilon)

    # Jacobian correction (same as yours)
    J = np.linalg.norm(np.cross(k1_vec, k2_vec))

    return F / J


def compute_spin_hall_conductivity_no(H_func, Nk, delta, t2, lambda_r, epsilon, return_Fmap=False):

    sigma_n = np.zeros(4)

    if return_Fmap:
        F_map = np.zeros((Nk, Nk, 4))

    # Jacobian (constant over grid)
    J = np.linalg.norm(np.cross(k1_vec, k2_vec))

    for i in range(Nk):
        for j in range(Nk):

            H = H_func(Nk, N1[i], N2[j], delta, t2)
            E, _ = np.linalg.eigh(H)

            F = spin_berry_curvature_2band_no(
                H_func, Nk, N1[i], N2[j],
                delta, t2, epsilon
            )

            if return_Fmap:
                F_map[i, j, :] = F

            for n in range(4):
                if E[n] < Ef:
                    sigma_n[n] += F[n]

    # --- prefactors ---
    sigma_n = sigma_n * (e_charge / hbar)

    # integration over BZ
    sigma_n *= (J / Nk**2) / (2*np.pi)**2

    sigma_total = np.sum(sigma_n)

    if return_Fmap:
        return sigma_n, sigma_total, F_map
    else:
        return sigma_n, sigma_total


epsilon_ideal=1e-8
t2_vals = np.linspace(0, 1.2, 13)
sigma_total_plot=np.zeros_like(t2_vals)

for i, t2 in enumerate(t2_vals):

        sigma_band, sigma_total = compute_spin_hall_conductivity_no(
            H_noRashba, Nk, delta_0, t2, lambda_r_0, epsilon=epsilon_ideal
        )
        sigma_total_plot[i] = sigma_total


# Theoretical transition t2 value
t2_transition = lambda_r_0 / (2 * np.sqrt(3))

plt.figure(figsize=(7,5))

# Computed spin Hall conductivity
plt.plot(t2_vals, sigma_total_plot, marker='o', linestyle='-', color='b', label='Computed sigma_total')

# Vertical line at theoretical t2
plt.axvline(x=t2_transition, color='r', linestyle='--', label=r'Theoretical $t_2 = \lambda_r / 2\sqrt{3}$')

plt.xlabel('t2')
plt.ylabel('Total Spin Hall Conductivity sigma_total')
plt.title('Spin Hall Conductivity vs t2')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

def H_withRashba_plot(kx, ky, delta, t2, lambda_r):

    S1,S2,S3=0.0,0.0,0.0

    for a in avec:
         ka=kx*a[0]+ky*a[1]
         S1+=(np.cos(ka))
         S2+=(np.sin(ka))
         
    for b in bvec:
        S3+=(np.sin(kx*b[0]+ky*b[1]))
    
    dx=t1*S1
    dy=t1*S2
    dz_up=delta-2*t2*(S3)
    dz_down=delta+2*t2*(S3)

    star1= (1j*lambda_r/2)*(-np.exp(1j*ky)+(np.exp(1j*ky/2))*(np.cos(np.sqrt(3)*kx/2)+np.sqrt(3)*np.sin(np.sqrt(3)*kx/2)))
    star4=np.conj(star1)
    star3= (1j*lambda_r/2)*(-np.exp(1j*ky)+(np.exp(1j*ky/2))*(np.cos(np.sqrt(3)*kx/2)-np.sqrt(3)*np.sin(np.sqrt(3)*kx/2)))
    star2=np.conj(star3)
    

    return np.array([
        [dz_up , dx-1j*dy, 0, star1 ],
        [dx+1j*dy , -dz_up, star2, 0 ],
        [0, star3, dz_down, dx-1j*dy ],
        [star4, 0, dx+1j*dy, -dz_down ]
        ]
        , dtype=complex)


kx_vals_plot=np.linspace(-np.pi,np.pi,201)
ky_vals_plot=np.linspace(-np.pi,np.pi,201)

X,Y=np.meshgrid(kx_vals_plot, ky_vals_plot)

E1_plot=np.zeros_like(X)
E2_plot=np.zeros_like(X)
E3_plot=np.zeros_like(X)
E4_plot=np.zeros_like(X)


for i in range(len(kx_vals_plot)):
    for j in range(len(ky_vals_plot)):
        
        kx, ky = kx_vals_plot[i], ky_vals_plot[j]
        
        Hij=H_withRashba_plot(kx,ky, delta_0, t2_0, lambda_r_0)
        w,_ = np.linalg.eigh(Hij)
        E1_plot[j,i]=w[0]
        E2_plot[j,i]=w[1]
        E3_plot[j,i]=w[2]
        E4_plot[j,i]=w[3]


%matplotlib widget


fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, E2_plot, cmap='Greens', edgecolor='k', linewidth=0.1, alpha=0.5)
ax.plot_surface(X, Y, E1_plot, cmap='Purples', edgecolor='k', linewidth=0.1, alpha=0.9)
ax.plot_surface(X, Y, E3_plot, cmap='Reds', edgecolor='k', linewidth=0.1, alpha=0.9)
ax.plot_surface(X, Y, E4_plot, cmap='Blues', edgecolor='k', linewidth=0.1, alpha=0.5)

# labels
ax.set_xlabel(r'$k_x$', fontsize=18)
ax.set_ylabel(r'$k_y$', fontsize=18)
ax.set_zlabel(r'$E$', fontsize=18)

# initial view
init_elev = 0
init_azim = 30
ax.view_init(elev=init_elev, azim=init_azim)

plt.title(r"Kane-Mele: Spin-resolved bands", fontsize=24)

# ---- SLIDER AXES ----
ax_elev = plt.axes([0.2, 0.05, 0.6, 0.02])
ax_azim = plt.axes([0.2, 0.01, 0.6, 0.02])

slider_elev = Slider(ax_elev, 'Elevation', -90, 90, valinit=init_elev)
slider_azim = Slider(ax_azim, 'Azimuth', 0, 360, valinit=init_azim)

# ---- UPDATE FUNCTION ----
def update(val):
    ax.view_init(elev=slider_elev.val, azim=slider_azim.val)
    fig.canvas.draw_idle()

slider_elev.on_changed(update)
slider_azim.on_changed(update)

plt.tight_layout()
plt.show()

# plotting in 2D K' to Gamma to K

N = 400

K = np.array([4*np.pi/(3*np.sqrt(3)), 0])
Kp = -K
Gamma = np.array([0, 0])

# Path: K' -> Gamma -> K
path1 = np.linspace(Kp, Gamma, N//2)
path2 = np.linspace(Gamma, K, N//2)

path = np.vstack((path1, path2))


E = np.zeros((len(path), 4))

for i, (kx, ky) in enumerate(path):
    H = H_withRashba_plot(kx, ky, delta_0, t2_0, lambda_r_0)
    w, _ = np.linalg.eigh(H)
    E[i] = w


plt.figure(figsize=(8,6))

for n in range(4):
    plt.plot(E[:, n], lw=2)

# Mark symmetry points
plt.axvline(x=N//2, color='k', linestyle='--')  # Gamma

plt.xticks([0, N//2, N-1], [r"$K'$", r"$\Gamma$", r"$K$"])
plt.ylabel("Energy")
plt.xlabel("k-path")
plt.title("Band structure (K' → Gamma → K)")

plt.grid(alpha=0.3)
plt.show()


lambda_list = [
    0.0,
    t2_0,                     # small Rashba
    2*np.sqrt(3)*t2_0,        # critical value
    4*np.sqrt(3)*t2_0         # strong Rashba
]


N = 400

K = np.array([4*np.pi/(3*np.sqrt(3)), 0])
Kp = -K
Gamma = np.array([0, 0])

path1 = np.linspace(Kp, Gamma, N//2)
path2 = np.linspace(Gamma, K, N//2)
path = np.vstack((path1, path2))


fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)

axes = axes.flatten()

for idx, lambda_r_val in enumerate(lambda_list):

    E = np.zeros((len(path), 4))

    for i, (kx, ky) in enumerate(path):
        H = H_withRashba_plot(kx, ky, delta_0, t2_0, lambda_r_val)
        w, _ = np.linalg.eigh(H)
        E[i] = w

    ax = axes[idx]

    for n in range(4):
        ax.plot(E[:, n], lw=2)

    # symmetry point line
    ax.axvline(x=N//2, color='k', linestyle='--', alpha=0.5)

    ax.set_xticks([0, N//2, N-1])
    ax.set_xticklabels([r"$K'$", r"$\Gamma$", r"$K$"])

    titles = [
        r"$\lambda_R = 0$",
        r"$\lambda_R = \lambda_{SO}$",
        r"$\lambda_R = 2\sqrt{3}\,\lambda_{SO}$",
        r"$\lambda_R = 4\sqrt{3}\,\lambda_{SO}$"
    ]

    ax.set_title(titles[idx], fontsize=14)

# shared labels
fig.supylabel("Energy", fontsize=16)
fig.supxlabel("k-path", fontsize=16)

plt.tight_layout()
plt.show()
