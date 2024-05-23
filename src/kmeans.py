import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, normalized_mutual_info_score, rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from collections import Counter
import os

dataset_name = 'elec'

#visual
data_path = '../data/'+dataset_name+'/image_feat.npy'
mm_feat = np.load(data_path)
k=100
model = KMeans(n_clusters=k,n_init=5)
model.fit(mm_feat)
label = list(model.predict(mm_feat))
np.savetxt('./visual_feat_label_'+dataset_name+'.txt',np.array(label))


label = Counter(label)
label = sorted(label.items(), key=lambda x: x[1], reverse=True)
for i in range(len(label)):
    label[i] = (i+1,label[i][1])
label = dict(label)
fig, ax = plt.subplots(figsize=(8, 5))
plt.bar(label.keys(),label.values(), color='#81B8DF',label='visual modality')
# plt.bar_label(y1.containers[0],fmt='%.1f')
# plt.title('Node'+str(top_node),fontsize=18)
# plt.xlim(0,0.01)
plt.xlabel('Class',fontsize=16)
plt.ylabel('Frequency',fontsize=16)
plt.savefig('./visual_feat_distribution_'+dataset_name+'.jpg',dpi=400)
plt.show()

#textual
data_path = '../data/'+dataset_name+'/text_feat.npy'
mm_feat = np.load(data_path)
model = KMeans(n_clusters=k,n_init=10)
model.fit(mm_feat)
label = list(model.predict(mm_feat))
np.savetxt('./texual_feat_label_'+dataset_name+'.txt',np.array(label))
label = Counter(label)
label = sorted(label.items(), key=lambda x: x[1], reverse=True)
for i in range(len(label)):
    label[i] = (i+1,label[i][1])
label = dict(label)
fig, ax = plt.subplots(figsize=(8, 5))
plt.bar(label.keys(),label.values(), color='#FE817D',label='textual modality')
# plt.bar_label(y1.containers[0],fmt='%.1f')
# plt.title('Node'+str(top_node),fontsize=18)
# plt.xlim(0,0.01)
plt.xlabel('Class',fontsize=16)
plt.ylabel('Frequency',fontsize=16)

plt.savefig('./textual_feat_distribution_'+dataset_name+'.jpg',dpi=400)
plt.show()
