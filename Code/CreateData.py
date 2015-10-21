# -*- coding: utf-8 -*-
"""
This python code downloads and unzipps the flight data
and converts the data ...
"""

import pandas as pd
f = pd.read_csv("/Users/dirkkalmbach/data/2008.csv")
f.head()
f.tail()
