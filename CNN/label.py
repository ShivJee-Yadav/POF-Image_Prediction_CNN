import numpy as np
import pandas as pd
import os

img_dir = "../Final_Sweep_reading"
files = sorted(os.listdir(img_dir))

files = [ f for f in files if f.endswith((".Bmp"))]

N = len(files)
T= 1000.0 # total Times 

indices = np.arange(N)
t_approx = indices / (N - 1) * T

freq_approx = 1 + np.floor(t_approx)

df = pd.DataFrame({
    "filename":files,
    "frequency_hz" : freq_approx.astype(int)
})
# print(df.info)
# input("Next")
# print(df.head(10))
# input("Next")
# print(df.tail(10))
df.to_csv("sweep_labels_approx.csv" , index=False)