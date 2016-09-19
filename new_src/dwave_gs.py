import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os, sys, json, re

from collections import defaultdict

FS = 18

RT_MIN = 0
RT_MAX = 2000

SAVE = True
IMG_DIR = './img/'

def get_files(direc):
    '''get all the files nested under a directory'''

    regex = re.compile('.*\.json')

    fnames = []
    for root, dirs, files in os.walk(direc):
        for fn in files:
            if regex.match(fn):
                fname = os.path.join(root, fn)
                fnames.append(fname)

    return fnames

def load_json_file(fname):
    ''' Pull data from a json formatted D-Wave result file '''

    try:
        fp = open(fname, 'r')
    except IOError:
        print('Failed to load file: {0}'.format(fname))
        raise IOError

    data = json.load(fp)
    fp.close()

    qbits = data['qbits']
    energies = data['energies']
    spins = data['spins']
    occ = data['occ']

    # pull runtime
    rt = int(os.path.basename(os.path.dirname(fname)))

    return qbits, energies, spins, occ, rt


def process(fname):

    qbits, energies, spins, occ, rt = load_json_file(fname)

    # number of qbits
    N = len(qbits)

    occ = np.array(occ)
    energies = np.array(energies)
    # pull ground state probability
    gsp = np.sum(occ[energies==energies[0]])*1./np.sum(occ)

    if gsp < 0.8:
        print(fname)
        print('{0:2f}: {1:.2f}\n'.format(energies[0], gsp))

    return N, gsp, rt

def main(direc, max_count=-1):

    fnames = get_files(direc)
    #np.random.shuffle(fnames)

    Ns = defaultdict(list)
    GSPs = defaultdict(list)
    for count, fn in enumerate(fnames):
        if count == max_count:
            break
        try:
            N, gsp, rt = process(fn)
            if rt > RT_MAX or rt < RT_MIN:
                continue
            Ns[rt].append(N)
            GSPs[rt].append(gsp)
            if gsp < .5 and N < 20:
                print(rt, N, round(gsp,3), fn)
        except KeyboardInterrupt:
            continue
        except Exception as e:
            print(e.message)
            continue

    if True:
        colors = cm.rainbow(np.linspace(0,1,len(Ns)))
        rts = sorted(Ns)
        for rt, c in zip(rts, colors):
            plt.plot(Ns[rt], GSPs[rt], 'x', color=c, markersize=6, markeredgewidth=1.5)
            plt.xlabel('Number of qubits', fontsize=FS)
            plt.ylabel('Ground state probability', fontsize=FS)
        plt.legend(['{0} us'.format(x) for x in rts], fontsize=FS, numpoints=1, loc='best')
        if SAVE:
            plt.savefig('./img/dwave_gs.eps', bboxinches='tight')
        plt.show()

    if False:
        colors = cm.rainbox(np.linspace(0,1, len(Ns)))


if __name__ == '__main__':

    try:
        direc = sys.argv[1]
    except:
        print('No directory path given...')
        sys.exit()

    try:
        max_count = int(sys.argv[2])
    except:
        max_count = -1

    main(direc, max_count=max_count)
