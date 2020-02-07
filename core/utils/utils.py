__author__ = "Steve Ataucuri"
__copyright__ = "Sprace.org.br"
__version__ = "1.0.0"

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from pylab import *


class Timer():
 
    def __init__(self):
        self.start_dt = None
 
    def start(self):
        self.start_dt = dt.datetime.now()
 
    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))


def position_3D_approximationn(y_true, y_predicted, cilyndrical=False):
    '''
        Return the closest points for every hit predicted with y_true and just the true values
    '''

    #this dataframe receives all X,Y,Z predicted considering a set of hists
    df3d = 0
    y_truecopy = np.copy(y_true)
    print(y_truecopy.shape)
    if not cilyndrical:
        df3d = pd.DataFrame({'x':y_predicted[:,0],'y':y_predicted[:,1],'z':y_predicted[:,2]})
        df3d['x-shortest'] = 0
        df3d['y-shortest'] = 0
        df3d['z-shortest'] = 0
        #df3d['dist'] = 0
    else:
        df3d = pd.DataFrame({'rho':y_predicted[:,0],'eta':y_predicted[:,1],'phi':y_predicted[:,2]})
        df3d['rho-shortest'] = 0
        df3d['eta-shortest'] = 0
        df3d['phi-shortest'] = 0
        #df3d['dist'] = 0

    #for each predicted hit, we will approximate to the closest hit considering gemoetric distance
    for index, row in df3d.iterrows():
        #obtain the row with least geometric distance between predicted row and original rows (in yclone)
        # se sao coordenadas cilindricas
        if not cilyndrical:        
            Xpred=df3d.loc[index, 'x']
            Ypred=df3d.loc[index, 'y']
            Zpred=df3d.loc[index, 'z']
        else:
            Xpred=df3d.loc[index, 'rho']
            Ypred=df3d.loc[index, 'eta']
            Zpred=df3d.loc[index, 'phi']
            
        #in this column we will create the geometric distance from all available hits and the current hit
        y_truecopy['dist'] = ( 
                                ((y_truecopy.iloc[:,0] - Xpred) **2) + 
                                ((y_truecopy.iloc[:,1] - Ypred) **2) + 
                                ((y_truecopy.iloc[:,2] - Zpred) **2) ).pow(0.5)

        y_truecopy=y_truecopy.sort_values(by=['dist'])

     
        # se sao coordenadas cilindricas
        if not cilyndrical:
            df3d.loc[index, 'x-shortest'] = y_truecopy.iloc[index,0]
            df3d.loc[index, 'y-shortest'] = y_truecopy.iloc[index,1]
            df3d.loc[index, 'z-shortest'] = y_truecopy.iloc[index,2]
        else:
            df3d.loc[index, 'rho-shortest'] = y_truecopy.iloc[index,0]
            df3d.loc[index, 'eta-shortest'] = y_truecopy.iloc[index,1]
            df3d.loc[index, 'phi-shortest'] = y_truecopy.iloc[index,2]            

    #df3d['dist'] = y_truecopy.iloc[0:,3]
    
    if not cilyndrical:
        
        df3d.drop('x', axis=1, inplace=True)
        df3d.drop('y', axis=1, inplace=True)
        df3d.drop('z', axis=1, inplace=True)
    else:
        df3d.drop('rho', axis=1, inplace=True)
        df3d.drop('eta', axis=1, inplace=True)
        df3d.drop('phi', axis=1, inplace=True) 
        
    #return the fourth hit of all tracks
    return(df3d)

def calculate_distances_vec(e1, e2):
    '''
        Calculate distances between two vectors
    '''
    # others ways:
    #d1 = np.linalg.norm(y_test.values[0]-predicted[0])
    #d2 = distance.euclidean(y_test.values[0], predicted[0])
    #d3 = calculate_distances_vec(y_test.values[0], predicted[0])
    # 
    return pow(
            (e1[0] - e2[0])**2 +
            (e1[1] - e2[1])**2 +
            (e1[2] - e2[2])**2 , 0.5)    

def get_shortest_points(y_true, y_predicted):
    '''
        Return the shortest points for every hit predicted with y_true 
    '''
    new_points = []
    tmp = 0
    neartest = 0
    br = np.array(y_predicted)
    ar = np.array(y_true) 
    
    for i in range(0, len(br)):
        tmp = 10000
        for j in range(0, len(ar)):
            d = calculate_distances_vec(br[i], ar[j])
            if(d<=tmp):             
                tmp = d
                neartest = ar[j]
            
        new_points.append(neartest)
  
    return new_points
        
def calculate_distances(y_true, y_predicted, y_hit_shortest, save_to):
    '''
        This function calculates distances between hits
        
        input:
            y_predicted : predicted hit
            y_hit_shortest: hit near of every predicted hit
            y_true: y_true or y_test
        
        return:
            d1 : distances between y_true and y_predicted
            d2 : distances between y_true and y_hit_shortest
        
    '''
    y_predicted = pd.DataFrame(y_predicted)
    
    dftemp = pd.DataFrame(index=range(len(y_true)),columns=range(12))
    dftemp[0]=y_true.iloc[:,0] # x
    dftemp[1]=y_true.iloc[:,1] # y 
    dftemp[2]=y_true.iloc[:,2] # z

    dftemp[3]=y_predicted.iloc[:,0]
    dftemp[4]=y_predicted.iloc[:,1]
    dftemp[5]=y_predicted.iloc[:,2]

    dftemp[6]=y_hit_shortest.iloc[:,0]
    dftemp[7]=y_hit_shortest.iloc[:,1]
    dftemp[8]=y_hit_shortest.iloc[:,2]

    
    #  y_true - y_predicted
    dftemp[9] =(((dftemp[0]-dftemp[3])**2)+((dftemp[1]-dftemp[4])**2)+((dftemp[2]-dftemp[5])**2)).pow(1./2)
    # y_true - y_hit_shortest
    dftemp[10]=(((dftemp[0]-dftemp[6])**2)+((dftemp[1]-dftemp[7])**2)+((dftemp[2]-dftemp[8])**2)).pow(1./2)
    # y_pred - y_shortest
    dftemp[11]=(((dftemp[3]-dftemp[6])**2)+((dftemp[4]-dftemp[7])**2)+((dftemp[5]-dftemp[8])**2)).pow(1./2)

    #print (dftemp.iloc[0:10,:])
    dftemp=dftemp.sort_values(by=[10], ascending=False)
    

    print ("Average Distance prediction (y_true - y_pred)" , dftemp[9].mean())
    print ("Average Distance Approximation (y_true - y_shortest)" , dftemp[10].mean())
    print ("Average Distance Approximation (y_pred - y_shortest)" , dftemp[11].mean())
    
    #predicted
    dist_pred = dftemp.iloc[:,9:10]
    # Approximated
    dist_approx = dftemp.iloc[:,10:11]
  

    return dist_pred, dist_approx

def calculate_distances_matrix(y_true, y_predicted):
    '''
        Distance calculation between two matrix
    '''
    dist = pow(
            (y_true[:,0] - y_predicted[:,0])**2 +
            (y_true[:,1] - y_predicted[:,1])**2 +
            (y_true[:,2] - y_predicted[:,2])**2 , 0.5)
    #dist.sort_values(ascending=False)

    return dist

def plot_distances(d1, d2, save_to):
    
    sns.set(rc={"figure.figsize": (14, 10)})

    subplot(2,2,1)
    ax = sns.distplot(d1)

    subplot(2,2,2)
    ax = sns.distplot(d1, rug=True, hist=True, color="r")
    plt.savefig(save_to)
    plt.show()
    '''
    fig,axes=plt.subplots(1,2)
    sns.distplot(d1,ax=axes[0])
    plt.grid(True)
    sns.distplot(d2,rug=True,ax=axes[1], color="r")
    plt.show()
    '''

def plot_distances_plotly(d1, d2, save_to):
    '''
        it seens does not save offline
    '''
    hist_data = [d1, d2]

    group_labels = ['Distances predicted', 'Distances Approx']

    # Create distplot with custom bin_size
    #fig = ff.create_distplot(hist_data, group_labels, bin_size=.25, curve_type='normal')
    fig = ff.create_distplot(hist_data, group_labels, show_curve=False, bin_size=.2)

    fig['layout'].update(title='Distances')

    fig.show()


