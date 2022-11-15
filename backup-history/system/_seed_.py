
import torch
import random
import numpy

class seed:

    def set(number=123):

        random.seed(number)
        numpy.random.seed(number)
        torch.cuda.manual_seed(number)
        torch.manual_seed(number)
        return

    pass
