import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os

'''
split full csv dataset into train, val, and test csv datasets, and mapp categories  to integer
'''

if __name__ == '__main__':
    # Loads the complete file name list with annotated labels for tasks (site, histology)
    df=pd.read_csv("data/histo_metadata.csv",delimiter='\t')
    df=df.dropna()
    
    
    df['site']=df['site'].apply(lambda x: str(x).replace("\"",""))
    unique_sites =  df['site'].unique()
    unique_sites = [site.strip() for site in unique_sites]
    unique_site_ids = range(len(unique_sites))
    site_dict = dict(zip(unique_sites, unique_site_ids))


    unique_histologies =  df['histology'].unique()
    unique_hist_ids = range(len(unique_histologies))
    histology_dict = dict(zip(unique_histologies, unique_hist_ids))
    #print(histology_dict)


    mapper_dir = "data/mapper"
    if not os.path.isdir(mapper_dir):
        os.mkdir(mapper_dir)

    site_mapper = os.path.join(mapper_dir,"site_class_mapper.json")
    with open(site_mapper, "w") as outfile:  
        json.dump(site_dict, outfile) 

    histology_mapper = os.path.join(mapper_dir,"histology_class_mapper.json")
    with open(histology_mapper, "w") as outfile:  
        json.dump(histology_dict, outfile) 

    # Train, Val, Test Split has 0.7, 0.1 and 0.2 of the entire data respectively
    train2, test = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
    train, val = train_test_split(train2,test_size=0.1,random_state=42,shuffle=True)

    split_dir = "data/split"
    if not os.path.isdir(split_dir):
        os.mkdir(split_dir)

    train.to_csv("data/split/train_labels.csv",index=False)
    val.to_csv("data/split/val_labels.csv", index=False)
    test.to_csv("data/split/test_labels.csv", index=False)

    print(len(df))
