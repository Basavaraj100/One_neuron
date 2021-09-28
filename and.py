from utils.model import Perceptron
from utils.all_utils import prefare_data




AND={'x1':[0,1,0,1],
    'x2':[0,0,1,1],
    'y':[0,0,0,1]}
df=pd.DataFrame(AND)
df
x,y=prefare_data(df)
x