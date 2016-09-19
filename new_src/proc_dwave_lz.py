#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from matplotlib.pyplot import cm

import sys, os, json
from collections import defaultdict

PAR = False
FS = 14

EQ_FILE = './data/models/d_wave_lz.txt'
EQ_TO_FILE = False


def model_to_file(popts):
    '''Save models to file'''

    try:
        fp = open(EQ_FILE, 'w')
    except:
        print('Failed to open eq file...')
        return

    for rt in sorted(popts):
        s = '{0} {1} {2}\n'.format(rt, popts[rt][0], popts[rt][1])
        fp.write(s)

    fp.close()


def load_data(fn):
    '''parse the data file'''

    try:
        fp = open(fn, 'r')
    except IOError:
        print('Failed to open file: {0}...'.format(fn))
        raise IOError

    data = []

    rts = defaultdict(int)
    i = 0
    for line in fp:
        if '#' in line:
            continue
        dat = line.strip().split()
        rt, p, d0, w0, root = dat
        rt = int(rt)
        rts[rt] += 1
        p, d0, w0 = map(float, [p, d0, w0])
        # if p < .5:
        #     print(rt, i, d0, root)
        x = {'rt':rt, 'p':p, 'gap':d0, 'w':w0, 'root':root, 'x':d0*w0}
        data.append(x)
        i += 1
    fp.close()

    print(rts)

    return data

def rt_slice(data, rt):
    ''' slice all the gaps and widths for a given runtime'''

    slc = []
    for x in data:
        if x['rt']==rt:
            slc.append([x['gap'], x['w'], x['p']])

    slc = np.array(slc)
    return slc

def gap_slice(data, gap):
    ''' slice all the runtimes and widths for a given gap'''

    slc = []
    for x in data:
        if abs(x['gap']-gap) < 1e-6*abs(gap):
            slc.append([x['rt'], x['p']])

    slc = np.array(slc)
    return slc

def par_slice(data, rt):
    ''' '''

    slc = []
    for x in data:
        if x['rt']==rt:
            slc.append([x['x'], x['p']])
    return np.array(slc)

def fit_gap(x,y,z):
    ''' '''

    fit_func = lambda x, a: np.exp(-2*np.pi*a*x)
    fit_func = lambda x, a, b: 1./(1+np.exp((x-b)/abs(a)))

    X = x
    popt, pcov = curve_fit(fit_func, X, z)
    perr = np.sqrt(np.diag(pcov))

    fitted = lambda x: fit_func(x, *popt)

    print(popt, perr)
    return popt, perr, fitted

def fit_data(x,y,z):
    ''' '''

    fit_func = lambda X, a, b: np.exp(-2*np.pi*(a*X[0]+b*X[1]))

    X = np.vstack([x,y])
    popt, pcov = curve_fit(fit_func, X, z)
    perr = np.sqrt(np.diag(pcov))
    print(popt, perr)

def main(fname, opts):
    ''' '''

    data = load_data(fname)

    if PAR:
        if opts:
            rts = [opts[0]]
        else:
            rts = [5, 10, 20, 40, 100, 200, 400, 1000, 2000]
        color=iter(cm.rainbow(np.linspace(0,1,len(rts))))
        for rt in rts:
            slc = par_slice(data, rt)
            x = slc[:,0]
            y = slc[:,1]
            c = next(color)
            plt.plot(np.log(x),y,'x', c=c)
        plt.show()

    elif True:
        if opts:
            rts = [opts[0]]
        else:
            rts = [5, 10, 20, 40, 100, 200, 400, 1000, 2000]
        print(rts)
        color = iter(cm.rainbow(np.linspace(0,1,len(rts))))

        fig1 = plt.figure('RT slice')
        ax = fig1.add_subplot(111, projection='3d')
        funcs = {}
        opts = {}
        for rt in rts:
            slc = rt_slice(data, rt)
            x = np.abs(slc[:,0])
            y = np.abs(slc[:,1])
            z = 1-slc[:,2]
            popt, perr, pfunc = fit_gap(x, y, z)
            funcs[rt] = pfunc
            opts[rt] = popt
            print('{0} data points'.format(slc.shape[0]))

            c = next(color)
            ax.scatter(x, y, z, c=c)

        ax.set_xlabel('Gap', fontsize=FS)
        ax.set_ylabel('Width', fontsize=FS)
        ax.set_zlabel('Prob', fontsize=FS)
        plt.show(block=False)

        # show fitting
        plt.figure('Gap fitting')
        color = iter(cm.rainbow(np.linspace(0,1,len(rts))))
        leg = []
        handles = []
        legend = []
        for rt in rts:
            slc = rt_slice(data, rt)
            x = np.abs(slc[:,0])
            z = 1-slc[:,2]

            c = next(color)
            h, = plt.plot(x, z, 'x', c=c, markersize=8, markeredgewidth=2)
            handles.append(h)
            legend.append('{0} us'.format(str(rt)))

            X = np.linspace(0, np.max(x), 100)
            plt.plot(X, funcs[rt](X), c=c, linewidth=2)

            leg += [None, str(rt)]

        plt.xlabel('Energy Gap (GHz)', fontsize=FS)
        plt.ylabel('Transition Probability', fontsize=FS)
        plt.ylim([0,1])
        plt.legend(handles, legend, numpoints=1, fontsize=FS)
        plt.savefig('./img/lz-fits.eps', bbox_inches='tight')
        plt.show()

        if EQ_TO_FILE:
            model_to_file(opts)


    else:
        gaps = [0.0882979361176, 0.06882066, 0.0496605069, 1.0530989, 1.55221505]
        for E in gaps:
            slc = gap_slice(data, E)

            plt.plot(np.log(slc[:,0]), slc[:, 1], 'x')
            plt.xlabel('Runtime', fontsize=FS)
            plt.ylabel('Prob', fontsize=FS)
            plt.show()

if __name__ == '__main__':

    try:
        fn = sys.argv[1]
    except KeyError:
        print('No data file given...')
        sys.exit()

    opts = [int(x) for x in sys.argv[2:]]

    main(fn, opts)
