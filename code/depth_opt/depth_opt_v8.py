import sys
sys.path.append('../')
from create_expressions import LogicTreeTrainer, LogicTree, LogicNode, TrueNode
from create_expressions import FalseNode, PNode, QNode, RNode, AndNode, OrNode
from create_expressions import ImplicationNode, NotNode
from datetime import datetime
import itertools
import matplotlib.pyplot as plt
import random
import math
from random import sample
import pickle as pkl
# import numpy as np
import scipy.io
import scipy.linalg as linalg
from sklearn.model_selection import train_test_split

import autograd.numpy as np
from autograd.numpy import linalg as la
from autograd import grad
from autograd.misc.optimizers import adam

from bow_representation import BOW_exprs


"""
Got the following code on projecting to nearest
positive semidefinite matrix at this link

https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
"""

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False


# LOAD/PREP DATA

v = 8
path = '../../data/metrics/metric' + str(v) + '/'


prep = False
if prep:
    print("Prepping data")

    bigram_dataset = pkl.load(open('../../data/bigram_datasets/T_bigram_dataset.pkl', 'rb'))
    bigs = []
    labs = []
    for treetup in bigram_dataset.values():
        bigs.append(np.array(treetup[0][1]))
        labs.append(len(treetup[1]) - 1)


    bigs = np.array(bigs)
    labs = np.array(labs)


    bigs_by_lab = []
    for i in np.unique(labs):
        bigs_by_lab.append((bigs[labs==i], i))


    trains = []
    tests = []
    for tup in bigs_by_lab:
        if tup[1] > 2:
            x = tup[0]
            y = labs[labs==tup[1]]
            X_train, X_test, y_train, y_test = train_test_split(\
                            x, y, test_size=0.3, \
                            random_state = int(datetime.now().strftime('%f')))
            trains.append((X_train, y_train))
            tests.append((X_test, y_test))

        else:
            trains.append((tup[0], labs[labs==tup[1]]))
            tests.append((tup[0], labs[labs==tup[1]]))


    train_bigs = []
    train_labs = []
    for tup in trains:
        train_bigs.extend(tup[0])
        train_labs.extend(tup[1])

    train_bigs = np.array(train_bigs)
    train_labs = np.array(train_labs)


    X_test = []
    y_test = []
    for tup in tests:
        X_test.extend(tup[0])
        y_test.extend(tup[1])

    X_test = np.array(X_test)
    y_test = np.array(y_test)




    samps = 30000
    include = [(0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (2,5), (3,4), (3,5), (4,5)]
    diff_tups = []
    for D1 in range(max(train_labs)+1):
        for D2 in range(D1, max(train_labs)+1): #labs[-1]+1 ensures depth=5 is used
            # if random.randint(0,1) == 0:

            if (D1,D2) in include:

                D1vs = (train_bigs[train_labs==D1]).copy()
                D2vs = (train_bigs[train_labs==D2]).copy()

                if D1 >= 2:

                    for i in range(samps):
                       # print("lens", len(D1vs), len(D2vs))
                       # print(i%len(D1vs), i%len(D2vs))
                        d1v = D1vs[i%len(D1vs)]
                        d2v = D2vs[i%len(D2vs)]
                        diff_tups.append((d2v-d1v, D2-D1))

                else:

                    D1samps = sample(list(D1vs), min(samps, len(D1vs)))
                    D2samps = sample(list(D2vs), min(samps, len(D2vs)))


                    for i in range(len(D2samps)):
                        if i >= len(D1samps):
                            idx = random.randint(0, len(D1samps)-1)
                            d1v = D1samps[idx]
                            d2v = D2samps[i]
                            diff_tups.append((d2v-d1v, D2-D1))

                        else:
                            d1v = D1samps[i]
                            d2v = D2samps[i]
                            diff_tups.append((d2v-d1v, D2-D1))


    X = np.array([tup[0] for tup in diff_tups])
    Y = np.array([tup[1] for tup in diff_tups])






def fit_metric(X, Y, iters=1000, rate=1e-7, smart_steps=False, seed=None, verbose=True):

    print("Xshape", X.shape)

    def objective(M):
        obj = 0
        for idx in range(len(X)):
            x = X[idx]
            y = Y[idx]
            obj += (np.dot(np.dot(x,M),x) - (y)**2)**2
        return obj


    if type(seed) != np.ndarray:
        M = np.identity(X.shape[1])
    else:
        M = seed.copy()


    rate_lb = 1e-20
    obj_grad = grad(objective)


    for i in range(1, iters):

        while True:

            if verbose:
                print("Computing gradient")


            m_grad = obj_grad(M)
            delta_grad = m_grad*rate

            if verbose:
                print("Restricting M")
            newM = nearestPD(M-delta_grad)

            init_obj = objective(M)
            new_obj = objective(newM)

            if new_obj < init_obj:
                break
            else:
                # print(rate, rate_lb, rate<rate_lb)
                if rate < rate_lb:
                    break
                if verbose:
                    print("Decreasing rate", rate, rate/2)
                rate /= 2


        if smart_steps and i <=100:
            if verbose:
                print("Doing smart steps")

            while (new_obj < init_obj) and (abs(new_obj - init_obj) > 10):

                if verbose:
                    if abs(new_obj - init_obj) < 2:
                        print(i, init_obj, new_obj, new_obj - init_obj, rate)
                    else:
                        print(i, int(init_obj), int(new_obj), int(new_obj - init_obj), rate)


                M = newM
                init_obj = objective(M)
                if verbose:
                    print("Restricting M")
                delta_grad = m_grad*rate
                newM = nearestPD(M-delta_grad)
                new_obj = objective(newM)
                rate *= 1.05

        else:

            if verbose:
                if abs(new_obj - init_obj) < 2:
                    print(i, init_obj, new_obj, new_obj - init_obj, rate)
                else:
                    print(i, int(init_obj), int(new_obj), int(new_obj - init_obj), rate)


            M = newM
            rate *= 1.05




        if (abs(new_obj - init_obj) < 0.1) or rate < rate_lb:
            if verbose:
                if abs(new_obj - init_obj) < 2:
                    print(i, init_obj, new_obj, new_obj - init_obj, rate)
                else:
                    print(i, int(init_obj), int(new_obj), int(new_obj - init_obj), rate)

                print("Metric converged with objective", int(new_obj))
            break
    else:
        if verbose:
            print("LMNN didn't converge in %d steps." % iters)

    return M


iters = 500

filename = 'metric' + str(v) + '_full_i' + str(iters)

test_dump_filename = path+'m' + str(v) + '_i' + str(iters) + '_test_data.pkl'
training_dump_filename = path+'m' + str(v) + '_i' + str(iters) + '_train_data.pkl'



fit = False
if fit:

    print("Dumping data")
    pkl.dump((X_test, y_test), open(test_dump_filename, 'wb'))
    pkl.dump((train_bigs, train_labs), open(training_dump_filename, 'wb'))


    print("Fitting metric")
    metric = fit_metric(X, Y, iters=iters, smart_steps=True, rate=5e-7)

    print("Dumping metric")
    pkl.dump(metric, open(path+filename + '.pkl', 'wb'))






# TEST METRIC

eval = True
if eval:

    print("Evaluating")

    metric = pkl.load(open(path+filename + '.pkl', 'rb'))

    # FOR SOME REASON A HAS IMAGINARY COMPONENTS BUT THEY'RE ALL 0 SO JUST TAKE THE REAL
    A = np.real(linalg.sqrtm(metric))

    test_data = pkl.load(open(test_dump_filename, 'rb'))
    train_data = pkl.load(open(training_dump_filename, 'rb'))


    test_vecs = test_data[0]
    test_depths = test_data[1]
    test_trans_vecs = np.matmul(test_vecs, A.T)

    train_vecs = train_data[0]
    train_depths = train_data[1]
    train_trans_vecs = np.matmul(train_vecs, A.T)


    def d_dists(vecs, trans_vecs, depths):

        odepth_vecs = []
        for depth in np.unique(depths):
            dvecs = vecs[depths == depth]
            odepth_vecs.append(dvecs)
        odepth_vecs = odepth_vecs[1:]

        tdepth_vecs = []
        for depth in np.unique(depths):
            dvecs = trans_vecs[depths == depth]
            tdepth_vecs.append(dvecs)
        tdepth_vecs = tdepth_vecs[1:]


        ot = vecs[0]
        tt = trans_vecs[0]


        odepth_dists = []
        tdepth_dists = []

        for depth in range(len(odepth_vecs)):

            ovecs = odepth_vecs[depth]
            tvecs = tdepth_vecs[depth]

            odists = []
            tdists = []

            for ovec in ovecs:
                odists.append(np.linalg.norm(ovec-ot))

            for tvec in tvecs:
                tdists.append(np.linalg.norm(tvec-tt))

            odepth_dists.append(np.array(odists))
            tdepth_dists.append(np.array(tdists))

        return (odepth_dists, tdepth_dists)


    print("Computing test dists")
    res = d_dists(test_vecs, test_trans_vecs, test_depths)
    test_odepth_dists, test_tdepth_dists = res[0], res[1]


    print("Computing train dists")
    res = d_dists(train_vecs, train_trans_vecs, train_depths)
    train_odepth_dists, train_tdepth_dists = res[0], res[1]


    print("Original Average Dists:")
    for depth in range(len(test_odepth_dists)):
        avg_d_dist = np.average(test_odepth_dists[depth])
        print(depth+1,':',avg_d_dist)

    print("Transform Average Dists:\n")
    for depth in range(len(test_tdepth_dists)):
        avg_d_dist = np.average(test_tdepth_dists[depth])
        print(depth+1,':',avg_d_dist)



    def dists_overlap(depth_dists):
        for d1 in range(1, len(depth_dists)):
            for d2 in range(d1+1, len(depth_dists)+1):

                d1_dists = depth_dists[d1-1]
                d2_dists = depth_dists[d2-1]

                num_overlap = 0
                num_total = len(d1_dists)*len(d2_dists)

                for d1_d in d1_dists:
                    for d2_d in d2_dists:
                        if d1_d >= d2_d:
                            num_overlap += 1

                percent_overlap = round((num_overlap/num_total)*100, 2)

                print("d", d1, ": d", d2, "percent_overlap:", percent_overlap)


    print('Original dists overlap')
    dists_overlap(test_odepth_dists)
    print('\nTransformed dists overlap')
    dists_overlap(test_tdepth_dists)


    show_test = True
    if show_test:
        odepth_dists = test_odepth_dists
        tdepth_dists = test_tdepth_dists
    else:
        odepth_dists = train_odepth_dists
        tdepth_dists = train_tdepth_dists

    cols = ['c','k','g','r','b']

    print("Plotting original distances")
    x = 1
    for depth in range(len(odepth_dists)):
        odists = odepth_dists[depth]
        for y in odists:
            plt.scatter(x, y, s=1, c=cols[depth])
            x += 1


    plt.xlabel("x_i")
    plt.ylabel("distance to base expression")

    # plt.ylim(0.0001,0.0006)
    plt.grid(b=True)

    plt.show()

    print("Plotting transformed distances")
    x = 1
    for depth in range(len(tdepth_dists)):
        tdists = tdepth_dists[depth]
        xs = []
        ys = []
        for y in tdists:
            xs.append(x)
            ys.append(y)
            x += 1
        plt.scatter(xs, ys, s=0.1, c=cols[depth])

    plt.xlabel("x_i")
    plt.ylabel("distance to base expression")

    # plt.ylim(0.0001,0.0005)
    plt.grid(b=True)

    plt.show()






# comment
