import numpy as np
import matplotlib.pyplot as plt

FS = 16

def main():

    if False:
        X = np.linspace(.2,10, 100)

        Y1 = 1e-3/X**2
        Y2 = 1e-3*np.exp(.5/X**2)

        plt.plot(X, Y1, 'r')
        plt.plot(X, Y2, 'b')
        plt.xlabel('quantum dot separation', fontsize=FS)
        plt.ylabel('Energy')

        plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

        plt.savefig('./img/gk.eps', bbox_inches='tight')
        plt.show()


    X = np.linspace(0.1, 2, 100)
    Y = (1./(2*X))/np.sqrt(1+(1./(2*X))**2)

    plt.plot(X, Y, linewidth=2)
    plt.xlabel('$\gamma$/$E_k$', fontsize=24)
    plt.ylabel('Maximum Polarization', fontsize=20)
    plt.savefig('./img/ekg.eps', bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()
