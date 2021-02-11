import pandas as pd
from sklearn.model_selection import train_test_split

'''
split full csv dataset into train, val, and test csv datasets
'''

if __name__ == '__main__':
    # Loads the complete file name list with annotated labels for tasks (site, histology)
    df=pd.read_csv("data/histo_metadata.csv",delimiter='\t')
    df=df.dropna()
    df['site']=df['site'].apply(lambda x: str(x).replace("\"",""))
    # Train, Val, Test Split has 0.7, 0.1 and 0.2 of the entire data respectively
    train2, test = train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
    train, val = train_test_split(train2,test_size=0.1,random_state=42,shuffle=True)
    
    train.to_csv("data/split/train_labels.csv",index=False)
    val.to_csv("data/split/val_labels.csv", index=False)
    test.to_csv("data/split/test_labels.csv", index=False)

    print(len(df))