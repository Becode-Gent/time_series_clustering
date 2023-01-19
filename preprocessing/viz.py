import plotly.express as px
import matplotlib.pyplot as plt

def viz_data(data,features):
    ax= plt.gca()
    for feature in features:
        data.plot(kind='line',x='ts_r',y= feature, ax=ax)
