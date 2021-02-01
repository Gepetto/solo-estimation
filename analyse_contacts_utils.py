import numpy as np
from scipy import interpolate
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
from sklearn import preprocessing, decomposition, svm, model_selection, pipeline 
from sklearn.model_selection import train_test_split

from data_readers import read_data_file_laas, read_data_file_laas_ctrl, shortened_arr_dic



DT = 0.002
DT_MOCAP = 0.005
DATA_FOLDER = 'data/Mesures_Contact/'
##########

def read_mocap(path_tsv, dt, S=None, E=None):
    """
    Fields names:
    Front_1	Front_2	Front_4	Front_5	Front_6	Rear_1	Rear_2	Forward_1	Forward_2	Backward_1	Backward_2	
    FL	FR	HL	HR
    11:15 -> 
    """

    markers = np.loadtxt(fname=path_tsv, dtype=str,
                     delimiter="\t", skiprows=9, max_rows=1)[1:]
    data = np.genfromtxt(fname=path_tsv, delimiter="\t", skip_header=10)
    if S is None:
        S = 0
    if E is None:
        E = len(data)
    data = data[S:E, :]
    t_mocap = np.arange(data.shape[0]) * dt

    fpos_dic = {
        'FL': data[:,-12:-9],
        'FR': data[:,-9:-6],
        'HL': data[:,-6:-3],
        'HR': data[:,-3:]
    }

    return t_mocap, fpos_dic

def warp(t_arr, a, b):
    return t_arr * a + b

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

def interpolate_mocap(t_new, t_mocap, mocapd):
    new_mocapd = {k: np.zeros((len(t_new),3)) for k in mocapd}
    for key in mocapd:
        for i in range(3):
            new_mocapd[key][:,i] = np.interp(t_new, t_mocap, mocapd[key][:,i])

    return t_new, new_mocapd

def read_expe1():
    tsv_name = "Measurement_19_01_2020_17_11.tsv"
    meas_name = "data_2021_01_19_17_11.npz"
    t_mocap, mocapd = read_mocap(DATA_FOLDER+tsv_name, DT_MOCAP, 3400, 8400)
    measd = read_data_file_laas(DATA_FOLDER+meas_name, DT)
    t_meas = warp(measd['t'], 0.993, 0.43)

    t_arr, mocapd = interpolate_mocap(t_meas, t_mocap, mocapd)

    return t_arr, mocapd, measd

def read_expe2():
    tsv_name = "Measurement_19_01_2020_16_58.tsv"
    meas_name = "data_2021_01_19_16_58.npz"
    t_mocap, mocapd = read_mocap(DATA_FOLDER+tsv_name, DT_MOCAP, 3500, 8200)
    measd = read_data_file_laas(DATA_FOLDER+meas_name, DT)
    t_meas = warp(measd['t'], 0.973, 0.1 - 0.058)

    t_arr, mocapd = interpolate_mocap(t_meas, t_mocap, mocapd)

    return t_arr, mocapd, measd

def read_expe3():
    tsv_name = "Measurement_19_01_2020_16_33.tsv"
    meas_name = "data_2021_01_19_16_33.npz"
    t_mocap, mocapd = read_mocap(DATA_FOLDER+tsv_name, DT_MOCAP, 2560, 7250)
    measd = read_data_file_laas(DATA_FOLDER+meas_name, DT)
    t_meas = warp(measd['t'], 0.9727, -0.047)

    t_arr, mocapd = interpolate_mocap(t_meas, t_mocap, mocapd)

    return t_arr, mocapd, measd


def ground_truth(z_arr):
    # get contact ground truth from mocap foot height
    peaks, _ = find_peaks(-z_arr, prominence=2)
    peaks = np.hstack([[0], peaks, [len(z_arr)-1]])

    # stepwise function extenting found minima midway to the next one
    f = interpolate.interp1d(peaks, z_arr[peaks], kind='nearest')
    minima_arr = f(np.arange(len(z_arr)))
    thresh_arr = minima_arr + 2  # from the stepwise minima curve, add a constant threshold

    return (z_arr > thresh_arr).astype(int)


########################################

if __name__ == '__main__':
    t_arr, mocapd, measd = read_expe1()

    ###
    # TRAIN Pipeline on FL contact classification
    ###

    # input data consist in 
    X = np.zeros((len(t_arr), 6))
    X[:, 0:3] = measd['tau'][:, 0:3]  # FL .. .. ..
    X[:, 3:6] = measd['dqa'][:, 0:3]

    # label ground truth based on threshold on mocap foot height
    Z_thresh = 18.38
    y = (mocapd['FL'][:,2] < Z_thresh).astype(int)

    C = 0.01 # SVM regularization parameter
    n_components = 2
    estimators = [('scaler', preprocessing.StandardScaler()),
                ('reduce_dim', decomposition.IncrementalPCA(n_components=n_components, batch_size=100)),
                ('clf', svm.SVC(kernel='linear', C=C))]
    pipe = pipeline.Pipeline(estimators)
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


    ### TEST data

    t_arr, mocapd, measd = read_expe2()

    X_test = np.zeros((len(t_arr), 6))
    X_test[:, 0:3] = measd['tau'][:, 0:3]
    X_test[:, 3:6] = measd['dqa'][:, 0:3]

    y_test = pipe.predict(X_test)

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


    plt.show()