import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from torch import nn, optim
import sklearn
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import model_selection,metrics
import matplotlib.pyplot as plt
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

import random
random.seed(10761594)

ori_data = pd.read_csv('musicData.csv')

raw_data = copy.deepcopy(ori_data)

raw_data.drop("artist_name",axis = 1 ,inplace = True)
raw_data.drop("track_name",axis = 1 ,inplace = True)
raw_data.drop("obtained_date",axis = 1 ,inplace = True)
raw_data.drop("instance_id",axis = 1 ,inplace = True)
'''energy hignly correlated with loudness'''
raw_data.drop("energy",axis = 1 ,inplace = True)
#raw_data = raw_data[raw_data.tempo != '?']
#raw_data = raw_data[raw_data.duration_ms != -1]
raw_data = raw_data.dropna()
raw_data.info()


corr = raw_data.corr()

Key_list = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]


'''normalize all the numerical column to 0-1'''



def process_one_genre(genre_pd,n):
    genre_pd_copy = copy.deepcopy(genre_pd)
    current = genre_pd_copy[genre_pd_copy.duration_ms != -1]
    genre_pd_copy['duration_ms'].replace(-1, current['duration_ms'].mean(), inplace=True)
    current = genre_pd_copy[genre_pd_copy.tempo != '?']
    current['tempo'] = current['tempo'].astype(float)
    genre_pd_copy['tempo'].replace('?', current['tempo'].mean(), inplace=True)
    genre_pd_copy['tempo'] = genre_pd_copy['tempo'].astype(float)
    genre_pd_copy['music_genre'] = n
    
    genre_pd_copy['key'].replace("A", 1, inplace=True)
    genre_pd_copy['key'].replace("A#", 2, inplace=True)
    genre_pd_copy['key'].replace("B", 3, inplace=True)
    genre_pd_copy['key'].replace("C", 4, inplace=True)
    genre_pd_copy['key'].replace("C#", 5, inplace=True)
    genre_pd_copy['key'].replace("D", 6, inplace=True)
    genre_pd_copy['key'].replace("D#", 7, inplace=True)
    genre_pd_copy['key'].replace("E", 8, inplace=True)
    genre_pd_copy['key'].replace("F", 9, inplace=True)
    genre_pd_copy['key'].replace("F#", 10, inplace=True)
    genre_pd_copy['key'].replace("G", 11, inplace=True)
    genre_pd_copy['key'].replace("G#", 12, inplace=True)
    
    genre_pd_copy['mode'].replace("Major", 0, inplace=True)
    genre_pd_copy['mode'].replace("Minor", 1, inplace=True)
    

    return genre_pd_copy
''' replace all non_sense data with mean of that category'''

all_Electronic = raw_data[raw_data.music_genre == 'Electronic']
all_Electronic = process_one_genre(all_Electronic,0)


all_Anime = raw_data[raw_data.music_genre == 'Anime']
all_Anime = process_one_genre(all_Anime,1)

all_Jazz = raw_data[raw_data.music_genre == 'Jazz']
all_Jazz = process_one_genre(all_Jazz,2)

all_Alternative = raw_data[raw_data.music_genre == 'Alternative']
all_Alternative = process_one_genre(all_Alternative,3)

all_Country = raw_data[raw_data.music_genre == 'Country']
all_Country = process_one_genre(all_Country,4)

all_Rap = raw_data[raw_data.music_genre == 'Rap']
all_Rap = process_one_genre(all_Rap,5)

all_Blues = raw_data[raw_data.music_genre == 'Blues']
all_Blues = process_one_genre(all_Blues,6)

all_Rock = raw_data[raw_data.music_genre == 'Rock']
all_Rock = process_one_genre(all_Rock,7)

all_Classical = raw_data[raw_data.music_genre == 'Classical']
all_Classical = process_one_genre(all_Classical,8)

all_Hip_Hop = raw_data[raw_data.music_genre == 'Hip-Hop']
all_Hip_Hop = process_one_genre(all_Hip_Hop,9)

new_data = copy.deepcopy(all_Electronic)
new_data = new_data.merge(all_Anime,how = 'outer')
new_data = new_data.merge(all_Jazz,how = 'outer')
new_data = new_data.merge(all_Alternative,how = 'outer')
new_data = new_data.merge(all_Country,how = 'outer')
new_data = new_data.merge(all_Rap,how = 'outer')
new_data = new_data.merge(all_Blues,how = 'outer')
new_data = new_data.merge(all_Rock,how = 'outer')
new_data = new_data.merge(all_Classical,how = 'outer')
new_data = new_data.merge(all_Hip_Hop,how = 'outer')


temp = new_data[['key','mode','music_genre']]
new_data.drop('key',axis = 1 ,inplace = True)
new_data.drop('mode',axis = 1 ,inplace = True)
new_data.drop('music_genre',axis = 1 ,inplace = True)
new_data = stats.zscore(new_data)
new_data = new_data.join(temp)


from umap import UMAP
umap_data = copy.deepcopy(new_data)
umap_data.drop("music_genre",axis = 1 ,inplace = True)
umap_model = UMAP(n_neighbors=15,min_dist=0.001,random_state=42)
umap_model.fit(umap_data)
X_umap = umap_model.transform(umap_data)


Genre_list = ['Electronic','Anime','Jazz','Alternative','Country','Rap','Blues','Rock','Classical','Hip-Hop']


cluster0 = X_umap[0:5000]
cluster1 = X_umap[5000:10000]
cluster2 = X_umap[10000:15000]
cluster3 = X_umap[15000:20000]
cluster4 = X_umap[20000:25000]
cluster5 = X_umap[25000:30000]
cluster6 = X_umap[30000:35000]
cluster7 = X_umap[35000:40000]
cluster8 = X_umap[40000:45000]
cluster9 = X_umap[45000:]


plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Electronic UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster0 = np.array(cluster0)
x = cluster0[:,0]
y = cluster0[:,1]
plt.scatter(x, y,color='red')
plt.show()







plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Anime UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster1 = np.array(cluster1)
x = cluster1[:,0]
y = cluster1[:,1]
plt.scatter(x, y,color='red')
plt.show()




plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Jazz UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster2 = np.array(cluster2)
x = cluster2[:,0]
y = cluster2[:,1]
plt.scatter(x, y,color='red')
plt.show()






plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Alternative UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster3 = np.array(cluster3)
x = cluster3[:,0]
y = cluster3[:,1]
plt.scatter(x, y,color='red')
plt.show()





plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Country UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster4 = np.array(cluster4)
x = cluster4[:,0]
y = cluster4[:,1]
plt.scatter(x, y,color='red')
plt.show()






plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Rap UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster5 = np.array(cluster5)
x = cluster5[:,0]
y = cluster5[:,1]
plt.scatter(x, y,color='red')
plt.show()






plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Blues UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster6 = np.array(cluster6)
x = cluster6[:,0]
y = cluster6[:,1]
plt.scatter(x, y,color='red')
plt.show()



plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Rock UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster7 = np.array(cluster7)
x = cluster7[:,0]
y = cluster7[:,1]
plt.scatter(x, y,color='red')
plt.show()



plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Classical UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster8 = np.array(cluster8)
x = cluster8[:,0]
y = cluster8[:,1]
plt.scatter(x, y,color='red')

plt.show()


plt.scatter(X_umap[:, 0], X_umap[:, 1])
plt.title("Hip-Hop UMAP")
plt.xlabel('Component 1')
plt.ylabel('Component 2')
cluster9 = np.array(cluster9)
x = cluster9[:,0]
y = cluster9[:,1]
plt.scatter(x, y,color='red')
plt.show()



pd.plotting.scatter_matrix(raw_data[['loudness']])
plt.show()


from sklearn.cluster import KMeans

kmeans = KMeans(10, n_init='auto')
labels = kmeans.fit_predict(X_umap)
k_mean_centers = kmeans.cluster_centers_

cluster0 = []
cluster1 = []
cluster2 = []
cluster3 = []
cluster4 = []
cluster5 = []
cluster6 = []
cluster7 = []
cluster8 = []
cluster9 = []


for index in range(len(labels)):
    if labels[index] == 0:
        cluster0.append(X_umap[index])
    elif labels[index] == 1:
        cluster1.append(X_umap[index])
    elif labels[index] == 2:
        cluster2.append(X_umap[index])
    elif labels[index] == 3:
        cluster3.append(X_umap[index])
    elif labels[index] == 4:
        cluster4.append(X_umap[index])
    elif labels[index] == 5:
        cluster5.append(X_umap[index])
    elif labels[index] == 6:
        cluster6.append(X_umap[index])
    elif labels[index] == 7:
        cluster7.append(X_umap[index])
    elif labels[index] == 8:
        cluster8.append(X_umap[index])
    elif labels[index] == 9:
        cluster9.append(X_umap[index])
    


plt.title("K Means")

cluster0 = np.array(cluster0)
x = cluster0[:,0]
y = cluster0[:,1]
plt.scatter(x, y,color='blue')

cluster1 = np.array(cluster1)
x = cluster1[:,0]
y = cluster1[:,1]
plt.scatter(x, y,color='orange')

cluster2 = np.array(cluster2)
x = cluster2[:,0]
y = cluster2[:,1]
plt.scatter(x, y,color='green')



cluster3 = np.array(cluster3)
x = cluster3[:,0]
y = cluster3[:,1]
plt.scatter(x, y,color='red')

cluster4 = np.array(cluster4)
x = cluster4[:,0]
y = cluster4[:,1]
plt.scatter(x, y,color='purple')

cluster5 = np.array(cluster5)
x = cluster5[:,0]
y = cluster5[:,1]
plt.scatter(x, y,color='brown')


cluster6 = np.array(cluster6)
x = cluster6[:,0]
y = cluster6[:,1]
plt.scatter(x, y,color='pink')

cluster7 = np.array(cluster7)
x = cluster7[:,0]
y = cluster7[:,1]
plt.scatter(x, y,color='gray')

cluster8 = np.array(cluster8)
x = cluster8[:,0]
y = cluster8[:,1]
plt.scatter(x, y,color='olive')


cluster9 = np.array(cluster9)
x = cluster9[:,0]
y = cluster9[:,1]
plt.scatter(x, y,color='cyan')
plt.show()





for Genre_int in range(10):
    current_genre = raw_data[raw_data.music_genre == Genre_list[Genre_int]]
    loudness = current_genre['loudness'].mean()
    print("Genre:",Genre_list[Genre_int],"  Mean loudness =", loudness)
    









def train_test_helper(genre_pd):
    label = genre_pd['music_genre']
    predictor = copy.deepcopy(genre_pd)
    predictor.drop("music_genre",axis = 1 ,inplace = True)
    X_train, X_test, Y_train, Y_test = train_test_split(predictor, label, test_size = 0.1, random_state=42) 
    training = copy.deepcopy(X_train)
    training = training.join(Y_train)
    testing = copy.deepcopy(X_test)
    testing = testing.join(Y_test)
    return  training, testing



new_data['labels'] = labels



Electronic_training, Electronic_testing = train_test_helper(new_data[new_data.music_genre == 0])
Anime_training, Anime_testing = train_test_helper(new_data[new_data.music_genre == 1])
Jazz_training, Jazz_testing = train_test_helper(new_data[new_data.music_genre == 2])
Alternative_training, Alternative_testing = train_test_helper(new_data[new_data.music_genre == 3])
Country_training, Country_testing = train_test_helper(new_data[new_data.music_genre == 4])
Rap_training, Rap_testing = train_test_helper(new_data[new_data.music_genre == 5])
Blues_training, Blues_testing = train_test_helper(new_data[new_data.music_genre == 6])
Rock_training, Rock_testing = train_test_helper(new_data[new_data.music_genre == 7])
Classical_training, Classical_testing = train_test_helper(new_data[new_data.music_genre == 8])
Hip_Hop_training, Hip_Hop_testing = train_test_helper(new_data[new_data.music_genre == 9])




training = copy.deepcopy(Electronic_training)
training = training.merge(Anime_training,how = 'outer')
training = training.merge(Jazz_training,how = 'outer')
training = training.merge(Alternative_training,how = 'outer')
training = training.merge(Country_training,how = 'outer')
training = training.merge(Rap_training,how = 'outer')
training = training.merge(Blues_training,how = 'outer')
training = training.merge(Rock_training,how = 'outer')
training = training.merge(Classical_training,how = 'outer')
training = training.merge(Hip_Hop_training,how = 'outer')

testing = copy.deepcopy(Electronic_testing)
testing = testing.merge(Anime_testing,how = 'outer')
testing = testing.merge(Jazz_testing,how = 'outer')
testing = testing.merge(Alternative_testing,how = 'outer')
testing = testing.merge(Country_testing,how = 'outer')
testing = testing.merge(Rap_testing,how = 'outer')
testing = testing.merge(Blues_testing,how = 'outer')
testing = testing.merge(Rock_testing,how = 'outer')
testing = testing.merge(Classical_testing,how = 'outer')
testing = testing.merge(Hip_Hop_testing,how = 'outer')

training = training.sample(frac = 1)

X_training = copy.deepcopy(training)
Y_training = training["music_genre"]
X_training.drop("music_genre",axis = 1 ,inplace = True)

X_testing = copy.deepcopy(testing)
Y_testing = testing["music_genre"]
X_testing.drop("music_genre",axis = 1 ,inplace = True)

'''Question 1'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")






learning_rate = 1e-1
lambda_l2 = 1e-6


X_train_tensor = torch.tensor(X_training.values, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_training.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_testing.values, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_testing.values, dtype=torch.long)

X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)

D = 13
H1 = 100
H2 = 80
H3 = 60
H4 = 40
H5 = 20
C = 10

model = nn.Sequential(
    nn.Linear(D, H1),
    nn.ReLU(),
    nn.Linear(H1, H2),
    nn.ReLU(),
    nn.Linear(H2, H3),
    nn.ReLU(),
    nn.Linear(H3, H4),
    nn.ReLU(),
    nn.Linear(H4, H5),
    nn.ReLU(),
    nn.Linear(H5,C)
)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=lambda_l2)


for t in range(10000):
    
    # Forward pass over the model to get the logits
    y_pred = model(X_train_tensor)
    
    # Compute the loss and accuracy
    loss = criterion(y_pred, Y_train_tensor)
    
    
    #auc = metrics.roc_auc_score(Y_train_diabetes, y_pred[:,1])
    '''
    print("[EPOCH]: %i, [LOSS]: %.6f" % (t, loss.item()))
    '''
    
    # reset (zero) the gradients before running the backward pass over the model
    # we need to do this because the gradients get accumulated at the same place across iterations
    optimizer.zero_grad()
    # Backward pass to compute the gradient of loss w.r.t our learnable params (weights and biases)
    loss.backward()
    # Update params
    optimizer.step()




y_pred = model(X_test_tensor)

y_pred = y_pred.detach().cpu().numpy()






plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')



def plot_auc_curve(Genre_int,Genre_list,y_pred):
    Genre_testing_for_roc = copy.deepcopy(Y_testing)
    if Genre_int!=0:
        for k in range(0,10):
            if k!= Genre_int:
                Genre_testing_for_roc.replace(k, 0, inplace=True)
        Genre_testing_for_roc.replace(Genre_int, 1, inplace=True)
    else:
        Genre_testing_for_roc.replace(0, -1, inplace=True)
        for k in range(0,10):
            if k!= Genre_int:
                Genre_testing_for_roc.replace(k, 0, inplace=True)
        Genre_testing_for_roc.replace(-1, 1, inplace=True)
    
    fpr, tpr, thresholds = metrics.roc_curve(Genre_testing_for_roc, y_pred[:,Genre_int])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label="%s(AUC = %.3f)"% (Genre_list[Genre_int],roc_auc))

plot_auc_curve(0,Genre_list,y_pred)
plot_auc_curve(1,Genre_list,y_pred)
plot_auc_curve(2,Genre_list,y_pred)
plot_auc_curve(3,Genre_list,y_pred)
plot_auc_curve(4,Genre_list,y_pred)
plot_auc_curve(5,Genre_list,y_pred)
plot_auc_curve(6,Genre_list,y_pred)
plot_auc_curve(7,Genre_list,y_pred)
plot_auc_curve(8,Genre_list,y_pred)
plot_auc_curve(9,Genre_list,y_pred)


plt.legend()
plt.show()


#auc = metrics.roc_auc_score(Y_test_diabetes, y_pred[:,1])
#print("one hidden layer ReLU():",auc)





