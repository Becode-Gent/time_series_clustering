import plotly.express as px
import matplotlib.pyplot as plt

def viz_data(data,features,title):
   
    plt.figure(1)
    data.plot(kind='line',x='ts_n',y= features,title = title)
    plt.show()
    


def viz_output(data, color,title,y):

    fig = px.scatter(data, x='ts_n', y= y, color=color)
    fig.update_layout(title_text=title, title_x=0.5)
    fig.show()


def viz_bar(data,title):
    
    fig = px.bar(data.groupby(['label', 'cluster']).size().unstack(level=1),width=600, height=400)
    fig.update_layout(title_text=title, title_x=0.5)
    fig.show()

def viz_score(data, title):
    
    plt.figure(1)
    data.plot(kind='line',x='k',y= 'Silhouette Score',title = title)
    plt.show()
    