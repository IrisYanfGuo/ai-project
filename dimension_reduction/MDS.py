
class MDS(object):
    '''
    MDS: multiple Dimensional Scaling
    input: Distance matrix, _d: dimension after reduction
    '''

    def __init__(self,dist):
        self.__dist = dist

