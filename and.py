from utils.model import Perceptron
from utils.all_utils import prefare_data, save_model,save_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib  # for saving model as binary fiel
from matplotlib.colors import ListedColormap
import os
plt.style.use('fivethirtyeight')  # for grapg style



AND={'x1':[0,1,0,1],
    'x2':[0,0,1,1],
    'y':[0,0,0,1]}
df=pd.DataFrame(AND)

x,y=prefare_data(df)


eta=0.1
epochs=10
model=Perceptron(eta,epochs)
model.fit(x,y)
_=model.total_loss()