#!/usr/bin/env python3

'''
    class to handle updating of parameters
'''

class Updater:
    '''
        base class

        declaring methods needed for different kinds of updaters
    '''
    def __init__(self, pars_init, n_iter):
        '''
            Parameter:
                pars_init: initial parameter

                n_iter: int
                    max number of updating iterations
        '''
        self.pars=np.array(pars_init)

        self.max_iter=n_iter
        self.n_iter=0

    def update(self, dpars):
        '''
            one update to parameters

            Parameter
                dpars:
                    modify of parameters
        '''
        if self.n_iter>=self.max_iter:
            print('iterations ended')
            return

        # simple updating is just add `dpars`
        # complex implement might implement, e.g., momentum mechanism
        self._update(dpars)

        self.n_iter+=1

    def _update(self, dpars):
        '''
            real implement of update
        '''
        self.pars+=dpars