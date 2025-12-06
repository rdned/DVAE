import math
import pandas as pd
pd.options.mode.chained_assignment = None

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)