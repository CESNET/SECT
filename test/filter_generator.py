import pandas as pd
import numpy as np
import datetime
from utils import *

if __name__=='__main__':

    x = np.array([
        0.5, 0., 0.5, 0., 0., 0.,
        0., 0.00613497, 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0.49693252, 0., 0.06748466, 0., 0., 0.,
        0.05521472, 0.03680982, 0., 0.03680982, 0.05521472, 0.03680982,
        0., 0.12883436, 0.05521472, 0.03680982, 0., 0.03680982,
        0.05521472, 0.03680982, 0., 0.03680982, 0.03680982, 0.03680982,
        0., 0.03680982, 0.03680982, 0.03680982, 0., 0.03680982,
        0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0.,
        0.03680982, 0., 0., 0., 0.03680982, 0.,
        0., 0., 0.03680982, 0., 0., 0.])

    first = datetime.datetime.fromisoformat('2019-09-25').timestamp()
    agg = 900
    offset = 1
    thr = 0.033


    x = get_beginning(x, first, agg, offset)
    print(x)