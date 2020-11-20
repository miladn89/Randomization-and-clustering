# Randomization-and-clustering
A design of experiment to test the significant differences between clusters are due to randomization or not


import os
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import random
import glob
# milad pca 12 features, randomize effect for small group clusters 1:6 2:19 3:30




final_clustering_sentiment = pd.read_csv(r'C:\Users\Owner\OneDrive - Mississippi State University\00. SNA\00. Second paper\000. SNA2_final analysis\000. more features_final\PCA and sentimental results.csv', index_col=0)

pca_data = final_clustering_sentiment.iloc[:, 1:-8]
df_final_2 = pd.DataFrame(pca_data, columns=final_clustering_sentiment.columns.values.tolist()[1:-8], index=final_clustering_sentiment.index)
df_final_1 = final_clustering_sentiment.iloc[:, -7:].join(df_final_2, on=final_clustering_sentiment.index, how='inner')
df_final_1.drop(['key_0'], axis=1, inplace=True)
# for i in df_final_1.columns.values[:12]:
#     df_final_1.loc[:,i] = df_final_1.loc[:,i]*(100.0/df_final_1.loc[:,i].sum(axis=0))

# pca = PCA(n_components=30)
# df_pca = pca.fit_transform(df_final_2)
df_pca = df_final_2

one = np.ones(6)
two = np.ones(19) *2
three = np.ones(30) *3
one = one.astype('int')
two = two.astype('int')
three = three.astype('int')
c_final = one.tolist() + two.tolist() + three.tolist()

for i in range(100):
    random.shuffle(c_final)
    df_final_1[f"Cluster_labels_random_{i}"] = c_final
    
    
    
for i in range(100):    
    fig, ax_new = plt.subplots(1, 7, sharey=False)
    df_final_1.boxplot(df_final_1.columns.values.tolist()[:7], by=f'Cluster_labels_random_{i}', ax=ax_new)
    # [ax_tmp.set_xlabel('') for ax_tmp in ax_new.reshape(-1)]
    # [ax_tmp.set_ylim(0.5, 3) for ax_tmp in ax_new.reshape(-1)]
    # ax_new[-1][-1].set_ylim(-4, -8)
    # ax_new[-1][-2].set_ylim(20,40)
    fig.suptitle(f'Box Plot of 55 Twitter_users features group by random clusters number {i}')
    plt.savefig(os.path.join(r'C:\Users\Admin\Desktop\nlp_twitter_clustering\random_result', f'random cluster number {i}.jpg'))
    plt.show()
    
# dict_final = defaultdict(int)
# for i in c_final:
#     dict_final[i] +=1
# print(dict_final)
df_final_1.to_csv(os.path.join(r'C:\Users\Admin\Desktop\nlp_twitter_clustering\random_result' ,'random_labling_100.csv'))


col = df_final_1.columns[:7].values.tolist()
c1 = [i+"_mean" for i in col]
c2 = [i+"_variance" for i in col]
cols = c1 + c2
for i in range(100):    
    temp = pd.DataFrame(np.zeros((3,14)), columns=cols)
    # df_sum = df_final_1.iloc[:,:7].groupby(f"Cluster_labels_random_{i}").sum()
    df_mean = df_final_1.sort_values([f"Cluster_labels_random_{i}"], ascending=True).groupby(f"Cluster_labels_random_{i}").mean()
    df_var = df_final_1.sort_values([f"Cluster_labels_random_{i}"], ascending=True).groupby(f"Cluster_labels_random_{i}").var()
    
    temp.iloc[:,:7] = df_mean.iloc[:,:7].values
    temp.iloc[:,7:14] = df_var.iloc[:,:7].values

    temp.to_csv(os.path.join(r'C:\Users\Admin\Desktop\nlp_twitter_clustering\random_result' ,f'mean_and_var_random_number_{i}.csv'))
 
    
path =r'C:\Users\Owner\OneDrive - Mississippi State University\00. SNA\00. Second paper\000. SNA2_final analysis\000. more features_final\random_result\New folder'
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df= pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    
# Concatenate all data into one DataFrame
myframe = pd.concat(li, axis=0, ignore_index=True)    
myframe.to_csv(os.path.join(r'C:\Users\Owner' ,'all.csv'))
