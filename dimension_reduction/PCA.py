from common_head import *

class PCA(object):
    '''
    PCA: principle component analysis
    '''

    def __init__(self,X):

        self.__X = tk.normalize4mat(X)


