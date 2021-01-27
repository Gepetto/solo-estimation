from example_robot_data import load
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed
import time
import math
import sklearn.preprocessing, sklearn.decomposition, sklearn.svm, sklearn.model_selection, sklearn.pipeline, sklearn.model_selection
from mpl_toolkits.mplot3d import Axes3D

def EulerToQuaternion(roll_pitch_yaw):
    """Roll Pitch Yaw to Quaternion"""

    roll, pitch, yaw = roll_pitch_yaw
    sr = math.sin(roll/2.)
    cr = math.cos(roll/2.)
    sp = math.sin(pitch/2.)
    cp = math.cos(pitch/2.)
    sy = math.sin(yaw/2.)
    cy = math.cos(yaw/2.)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return [qx, qy, qz, qw]

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


##
# Load motion capture data
##

tsv_name = "Measurement_19_01_2020_17_11.tsv"
markers = np.loadtxt(fname=tsv_name, dtype=str,
                     delimiter="\t", skiprows=9, max_rows=1)[1:]
data = np.genfromtxt(fname=tsv_name, delimiter="\t", skip_header=10)
data = data[3400:8400, :]
t_mocap = np.linspace(0, data.shape[0], data.shape[0] + 1)[:-1] * 0.005

##
# Load data logged by robot
##

meas_name = "data_2021_01_19_17_11.npz"
meas_ctrl_name = "data_control_2021_01_19_17_11.npz"
data_meas = np.load(meas_name)
data_ctrl = np.load(meas_ctrl_name)

torques = data_meas['torquesFromCurrentMeasurment']
pos = data_ctrl['log_feet_pos']
pos_target = data_ctrl['log_feet_pos_target']
q_des = data_ctrl['log_qdes']
q_mes = data_meas["q_mes"]

N = torques.shape[0]
t_meas = np.linspace(
    0, torques.shape[0], torques.shape[0] + 1)[:-1] * 0.002 * 0.993
t_meas += 0.43

###
# Plot 3D position of feet measured by mocap
###

"""index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
lgd1 = ["Pos X", "Pos Y", "Pos Z"]
lgd2 = ["FL", "FR", "HL", "HR"]
plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index[i])
    else:
        plt.subplot(3, 4, index[i], sharex=ax0)

    h1, = plt.plot(t_mocap, data[:, 3*11+i], "r", linewidth=3)

    plt.xlabel("Time [s]")
    plt.ylabel(lgd1[i % 3]+" "+lgd2[int(i/3)])
plt.suptitle("Feet positions (world frame)")"""

###
# Plot Z position of FL foot + Leg torques + Height target + Height pos
###

plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[tmp > 18.38] = np.nan
plt.plot(t_mocap, tmp, "g", linewidth=3)

# plt.plot(t_meas, torques[:, 1]+18.3, linewidth=3)
# plt.plot(t_meas, torques[:, 2]+18.3, linewidth=3)
# plt.plot(t_meas, pos_target[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="-.")
# plt.plot(t_meas, pos[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="--")

plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")

###
# Plot torques with color marking for contact state
###

"""in_contact = data[:, 3*11+2].copy()
in_contact = in_contact < 18.38

match = np.zeros(t_meas.shape[0], dtype=int)
for i in range(t_meas.shape[0]):
    match[i] = np.argmin(np.abs(t_meas[i] - t_mocap))
contact_match = in_contact[match]

c_contact = np.array(["r"] * t_meas.shape[0])
c_contact[contact_match] = "g"

plt.figure()
plt.scatter(torques[:, 1], torques[:, 2], marker='x', color=c_contact)
plt.plot(np.linspace(-0.4, 1.0, 100), 0.365 +
         np.linspace(-0.4, 1.0, 100) * 0.7, marker='o', color="k")"""

###
# Plot foot Z position and state detection with linear regression of previous graph
###

"""contact_condition = torques[:, 2] > (0.365 + torques[:, 1] * 0.7)
match2 = np.zeros(t_mocap.shape[0], dtype=int)
for i in range(t_mocap.shape[0]):
    match2[i] = np.argmin(np.abs(t_meas - t_mocap[i]))
contact_match2 = contact_condition[match2]

plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[np.logical_not(contact_match2)] = np.nan
plt.plot(t_mocap, tmp, "g", linewidth=3)
plt.plot(t_meas, torques[:, 1]+18.3, linewidth=3)
plt.plot(t_meas, torques[:, 2]+18.3, linewidth=3)
plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")"""

###
# Comparison 3D feet position / desired 3D feet position
###

"""plt.figure()
plt.plot(pos_target[2, 0, :])
plt.plot(pos[2, 0, :])
plt.plot(pos_target[2, 1, :])
plt.plot(pos[2, 1, :])
plt.figure()
plt.plot(pos_target[2, 3, :])
plt.plot(pos[2, 3, :])
plt.plot(pos_target[2, 2, :])
plt.plot(pos[2, 2, :])"""

###
# Comparison measured actuator positions / positions desired by WBC
###

"""plt.figure()
for i in range(12):
    if i == 0:
        ax0 = plt.subplot(3, 4, index[i])
    else:
        plt.subplot(3, 4, index[i], sharex=ax0)

    plt.plot(q_des[i, :-5], "r", linewidth=3)
    plt.plot(q_mes[:-5, i], "b", linewidth=3)
"""

###
# Principal component analysis for FL leg
###

"""N = torques.shape[0] - 5
match = np.zeros(N, dtype=int)
for i in range(N):
    match[i] = np.argmin(np.abs(t_meas[i] - t_mocap))

X = np.zeros((N, 6))
X[:, 0:3] = torques[:N, 0:3]
X[:, 3:6] = q_mes[:N, 0:3]
X = sklearn.preprocessing.scale(X)

y = (data[match, 3*11+2] < 18.38).astype(int)

n_components = 2
ipca = sklearn.decomposition.IncrementalPCA(
    n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

colors = ['navy', 'darkorange']

for X_transformed, title in [(X_ipca, "Incremental PCA")]:
    fig = plt.figure(figsize=(8, 8))
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    for color, i, target_name in zip(colors, [0, 1], ["No contact", "Contact"]):
        if n_components == 2:
            plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                        color=color, lw=2, label=target_name)
        elif n_components == 3:
            ax.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                        X_transformed[y == i, 2], color=color, lw=2, label=target_name)

    plt.legend(loc="best", shadow=False, scatterpoints=1)
    #plt.axis([-4, 4, -1.5, 1.5])
"""
###
# Support Vector Machine classification
###

"""# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (sklearn.svm.SVC(kernel='linear', C=C),
          sklearn.svm.LinearSVC(C=C, max_iter=10000),
          sklearn.svm.SVC(kernel='rbf', gamma=0.7, C=C),
          sklearn.svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
models = (clf.fit(X_transformed, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X_transformed[:, 0], X_transformed[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

# Scores
C = 1.0  # SVM regularization parameter
models = (sklearn.svm.SVC(kernel='linear', C=C),
          sklearn.svm.LinearSVC(C=C, max_iter=10000),
          sklearn.svm.SVC(kernel='rbf', gamma=0.7, C=C),
          sklearn.svm.SVC(kernel='poly', degree=3, gamma='auto', C=C))
print("-- Computing scores --")
for clf, title in zip(models, titles):
    scores = sklearn.model_selection.cross_val_score(clf, X_transformed, y, cv=5)
    print(title + ": ")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

"""

"""from IPython import embed
embed()"""

###
# Test Pipeline
###

N = torques.shape[0] - 5
match = np.zeros(N, dtype=int)
for i in range(N):
    match[i] = np.argmin(np.abs(t_meas[i] - t_mocap))

X = np.zeros((N, 6))
X[:, 0:3] = torques[:N, 0:3]
X[:, 3:6] = q_mes[:N, 0:3]
y = (data[match, 3*11+2] < 18.38).astype(int)

t_start = time.time()

estimators_grid = [('scaler', sklearn.preprocessing.StandardScaler()),
              ('reduce_dim', sklearn.decomposition.IncrementalPCA()),
              ('clf', sklearn.svm.SVC())]

# Parameters of pipelines can be set using ‘__’ separated parameter names:
"""param_grid = {
    'reduce_dim__n_components': [1, 2],
    'reduce_dim__batch_size': [100],
    'clf__C': np.logspace(-2, 2, 5),
    'clf__kernel': ['linear'],
}
pipe_grid = sklearn.pipeline.Pipeline(estimators_grid)
search = sklearn.model_selection.GridSearchCV(pipe_grid, param_grid, n_jobs=-1, verbose=3)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)"""
# Best parameter (CV score=0.952):
# {'clf__C': 0.01, 'clf__kernel': 'linear', 'reduce_dim__batch_size': 100, 'reduce_dim__n_components': 3}

"""param_grid = {
    'reduce_dim__n_components': [2, 3, 4],
    'reduce_dim__batch_size': [100],
    'clf__C': np.logspace(-2, 2, 5),
    'clf__kernel': ['poly'],
    'clf__gamma': ['auto'],
    'clf__degree': [2, 3],

}
pipe_grid = sklearn.pipeline.Pipeline(estimators_grid)
search = sklearn.model_selection.GridSearchCV(pipe_grid, param_grid, n_jobs=-1, verbose=3)
search.fit(X, y)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)"""
# Best parameter (CV score=0.954):
# {'clf__C': 0.1, 'clf__degree': 3, 'clf__gamma': 'auto', 'clf__kernel': 'poly', 'reduce_dim__batch_size': 100, 'reduce_dim__n_components': 3}

print("Computation time: ", time.time() - t_start)

C = 0.01 # SVM regularization parameter
n_components = 2
estimators = [('scaler', sklearn.preprocessing.StandardScaler()),
              ('reduce_dim', sklearn.decomposition.IncrementalPCA(n_components=n_components, batch_size=100)),
              ('clf', sklearn.svm.SVC(kernel='linear', C=C))]
pipe = sklearn.pipeline.Pipeline(estimators)
pipe.fit(X, y)

X_scaled = pipe['scaler'].transform(X)
X_transformed = pipe['reduce_dim'].transform(X_scaled)

X0, X1 = X_transformed[:, 0], X_transformed[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()
ax = plt.gca()
plot_contours(ax, pipe['clf'], xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("SVC with linear kernel")

###
# Testing detection on other datasets (Measurement_19_01_2020_16_58)
##


tsv_name = "Measurement_19_01_2020_16_58.tsv"
markers = np.loadtxt(fname=tsv_name, dtype=str,
                     delimiter="\t", skiprows=9, max_rows=1)[1:]
data = np.genfromtxt(fname=tsv_name, delimiter="\t", skip_header=10)
data = data[3500:8200, :]
t_mocap = np.linspace(0, data.shape[0], data.shape[0] + 1)[:-1] * 0.005

meas_name = "data_2021_01_19_16_58.npz"
meas_ctrl_name = "data_control_2021_01_19_16_58.npz"
data_meas = np.load(meas_name)
data_ctrl = np.load(meas_ctrl_name)

torques = data_meas['torquesFromCurrentMeasurment']
pos = data_ctrl['log_feet_pos']
pos_target = data_ctrl['log_feet_pos_target']
q_des = data_ctrl['log_qdes']
q_mes = data_meas["q_mes"]

N = torques.shape[0] - 5 
t_meas = np.linspace(
    0, N, N + 1)[:-1] * 0.002 * 0.973
t_meas += 0.1 - 0.058

"""index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[tmp > 18.38] = np.nan
# plt.plot(t_mocap, tmp, "g", linewidth=3)

plt.plot(t_meas, torques[:, 1]+18.3, linewidth=3)
plt.plot(t_meas, torques[:, 2]+18.3, linewidth=3)
plt.plot(t_meas, pos_target[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="-.")
plt.plot(t_meas, pos[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="--")

plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")"""



X_test = np.zeros((N, 6))
X_test[:, 0:3] = torques[:N, 0:3]
X_test[:, 3:6] = q_mes[:N, 0:3]

y_test = pipe.predict(X_test)

N = data.shape[0]
match = np.zeros(N, dtype=int)
for i in range(N):
    match[i] = np.argmin(np.abs(t_meas - t_mocap[i]))

y_mocap = y_test[match]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[y_mocap == 0] = np.nan
plt.plot(t_mocap, tmp, "g", linewidth=3)
plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")

X_scaled = pipe['scaler'].transform(X_test)
X_transformed = pipe['reduce_dim'].transform(X_scaled)

X0, X1 = X_transformed[:, 0], X_transformed[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()
ax = plt.gca()
plot_contours(ax, pipe['clf'], xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("SVC with linear kernel")

###
# Testing detection on other datasets (Measurement_19_01_2020_16_58)
##


tsv_name = "Measurement_19_01_2020_16_58.tsv"
markers = np.loadtxt(fname=tsv_name, dtype=str,
                     delimiter="\t", skiprows=9, max_rows=1)[1:]
data = np.genfromtxt(fname=tsv_name, delimiter="\t", skip_header=10)
data = data[3500:8200, :]
t_mocap = np.linspace(0, data.shape[0], data.shape[0] + 1)[:-1] * 0.005

meas_name = "data_2021_01_19_16_58.npz"
meas_ctrl_name = "data_control_2021_01_19_16_58.npz"
data_meas = np.load(meas_name)
data_ctrl = np.load(meas_ctrl_name)

torques = data_meas['torquesFromCurrentMeasurment']
pos = data_ctrl['log_feet_pos']
pos_target = data_ctrl['log_feet_pos_target']
q_des = data_ctrl['log_qdes']
q_mes = data_meas["q_mes"]

N = torques.shape[0] - 5 
t_meas = np.linspace(
    0, N, N + 1)[:-1] * 0.002 * 0.973
t_meas += 0.1 - 0.058

"""index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[tmp > 18.38] = np.nan
# plt.plot(t_mocap, tmp, "g", linewidth=3)

plt.plot(t_meas, torques[:, 1]+18.3, linewidth=3)
plt.plot(t_meas, torques[:, 2]+18.3, linewidth=3)
plt.plot(t_meas, pos_target[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="-.")
plt.plot(t_meas, pos[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="--")

plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")"""



X_test = np.zeros((N, 6))
X_test[:, 0:3] = torques[:N, 0:3]
X_test[:, 3:6] = q_mes[:N, 0:3]

y_test = pipe.predict(X_test)

N = data.shape[0]
match = np.zeros(N, dtype=int)
for i in range(N):
    match[i] = np.argmin(np.abs(t_meas - t_mocap[i]))

y_mocap = y_test[match]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[y_mocap == 0] = np.nan
plt.plot(t_mocap, tmp, "g", linewidth=3)
plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")

X_scaled = pipe['scaler'].transform(X_test)
X_transformed = pipe['reduce_dim'].transform(X_scaled)

X0, X1 = X_transformed[:, 0], X_transformed[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()
ax = plt.gca()
plot_contours(ax, pipe['clf'], xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("SVC with linear kernel")

###
# Testing detection on other datasets (Measurement_19_01_2020_16_33)
##


tsv_name = "Measurement_19_01_2020_16_33.tsv"
markers = np.loadtxt(fname=tsv_name, dtype=str,
                     delimiter="\t", skiprows=9, max_rows=1)[1:]
data = np.genfromtxt(fname=tsv_name, delimiter="\t", skip_header=10)
data = data[2560:7250, :]
t_mocap = np.linspace(0, data.shape[0], data.shape[0] + 1)[:-1] * 0.005

meas_name = "data_2021_01_19_16_33.npz"
meas_ctrl_name = "data_control_2021_01_19_16_33.npz"
data_meas = np.load(meas_name)
data_ctrl = np.load(meas_ctrl_name)

torques = data_meas['torquesFromCurrentMeasurment']
pos = data_ctrl['log_feet_pos']
pos_target = data_ctrl['log_feet_pos_target']
q_des = data_ctrl['log_qdes']
q_mes = data_meas["q_mes"]
v = data_meas["estimatorVelocity"]

N = torques.shape[0] - 5
t_meas = np.linspace(
    0, N, N + 1)[:-1] * 0.002 * 0.9727
t_meas += - 0.047

index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
"""plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[tmp > 18.38] = np.nan
# plt.plot(t_mocap, tmp, "g", linewidth=3)

plt.plot(t_meas, v[:, 0])
plt.plot(t_meas, torques[:, 1]+18.3, linewidth=3)
plt.plot(t_meas, torques[:, 2]+18.3, linewidth=3)
plt.plot(t_meas, pos_target[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="-.")
plt.plot(t_meas, pos[2, 0, :] * 1000 + 18.3, linewidth=4, linestyle="--")

plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")"""



X_test = np.zeros((N, 6))
X_test[:, 0:3] = torques[:N, 0:3]
X_test[:, 3:6] = q_mes[:N, 0:3]

y_test = pipe.predict(X_test)

N = data.shape[0]
match = np.zeros(N, dtype=int)
for i in range(N):
    match[i] = np.argmin(np.abs(t_meas - t_mocap[i]))

y_mocap = y_test[match]
index = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]
plt.figure()
plt.plot(t_mocap, data[:, 3*11+2], "r", linewidth=3)
tmp = data[:, 3*11+2].copy()
tmp[y_mocap == 0] = np.nan
plt.plot(t_mocap, tmp, "g", linewidth=3)
plt.xlabel("Time [s]")
plt.ylabel("Mocap foot pos Z [mm]")

X_scaled = pipe['scaler'].transform(X_test)
X_transformed = pipe['reduce_dim'].transform(X_scaled)

X0, X1 = X_transformed[:, 0], X_transformed[:, 1]
xx, yy = make_meshgrid(X0, X1)

plt.figure()
ax = plt.gca()
plot_contours(ax, pipe['clf'], xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title("SVC with linear kernel")
###
# Display figures
###

plt.show(block=True)

###
# Replay in Gepetto Viewer for 3D visualization
###

"""# Load robot model and data
solo = load('solo12')
q = solo.q0.reshape((-1, 1))

# Initialisation of the Gepetto viewer
solo.initDisplay(loadModel=True)
if ('viewer' in solo.viz.__dict__):
    solo.viewer.gui.addFloor('world/floor')
    solo.viewer.gui.setRefreshIsSynchronous(False)
solo.display(q)

log_q = data_ctrl["log_q"]
q_mes = data_meas["q_mes"]
for i in range(log_q.shape[1]):

    # Display desired 3D position of feet with magenta spheres (gepetto gui)
    rgbt = [1.0, 0.0, 1.0, 0.5]
    for k in range(4):
        if i == 0:
            solo.viewer.gui.addSphere(
                "world/sphere"+str(k)+"_des", .02, rgbt)  # .1 is the radius
        solo.viewer.gui.applyConfiguration("world/sphere"+str(k)+"_des",
                                           (pos_target[0, k, i], pos_target[1, k, i],
                                            pos_target[2, k, i], 1., 0., 0., 0.))

    rgbt = [0.0, 0.0, 1.0, 0.5]
    for k in range(4):
        if i == 0:
            solo.viewer.gui.addSphere(
                "world/sphere"+str(k)+"_pos", .02, rgbt)  # .1 is the radius
        solo.viewer.gui.applyConfiguration("world/sphere"+str(k)+"_pos",
                                           (pos[0, k, i], pos[1, k, i],
                                            pos[2, k, i], 1., 0., 0., 0.))

    print("==")
    print("(x, y, z): ", (pos_target[0, 0, i],
                          pos_target[1, 0, i], pos_target[2, 0, i]))
    print("(x, y, z): ", (pos[0, 0, i], pos[1, 0, i], pos[2, 0, i]))

    q[0:3, 0] = log_q[0:3, i]
    q[3:7, 0] = EulerToQuaternion(log_q[3:6, i])
    q[7:, 0] = q_mes[i, :]
    solo.display(q)
    time.sleep(0.2)"""
