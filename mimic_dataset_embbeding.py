import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer

from embedding_based_medical_processor import *
import cv2

from tqdm import tqdm
import re


#configure this path according to the download of this file
csv_jpg_chest='/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'

dataset=pd.read_csv(csv_jpg_chest)
chexnet_targets = [
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices']

u_one_features = ['Atelectasis', 'Edema','No Finding']
u_zero_features = ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']
otros = [
       'Enlarged Cardiomediastinum',  'Lung Opacity',
       'Lung Lesion', 'Pneumonia', 
       'Pneumothorax', 'Pleural Other', 'Fracture',
       'Support Devices']
def feature_string(row):
    feature_list = []
    for feature in u_one_features:
        if row[feature] in [-1,1]:
            feature_list.append(feature)
            
    for feature in u_zero_features:
        if row[feature] == 1:
            feature_list.append(feature)
    return ';'.join(feature_list)

df_balanced = dataset
df_balanced = df_balanced[ ['subject_id',	'study_id',	'Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis','Pleural Effusion','No Finding']]
df_balanced['label'] = df_balanced.apply(feature_string,axis = 1,).fillna('')
df_balanced['label'] =df_balanced['label'] .apply(lambda x:x.split(";"))


df_balanced=df_balanced[df_balanced["label"].apply(lambda x: '' not in x)]

one_hot = MultiLabelBinarizer()
encoder=one_hot.fit_transform(df_balanced["label"])
df_balanced['encoder'] = encoder.tolist()


df_balanced=df_balanced[df_balanced["label"].apply(lambda x: '' not in x)]


def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center')



def generate_images_emmbedings(img_path):
    img_cxr_shape = [224, 224]
    img_cxr = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (img_cxr_shape[0], img_cxr_shape[1]))
    densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img_cxr)
    return densefeature_embeddings, prediction_embeddings

def extract_reason_exam(report):
    regular_expresion = r'(INDICATION:(.+?)\n \n|REASON FOR EXAM:(.+?)\n \n|REASON FOR EXAMINATION:(.+?)\n \n|CLINICAL INFORMATION:(.+?)\n \n|HISTORY:(.+?)\n \n|CLINICAL HISTORY(.+?)\n \n)'
    text = re.search(regular_expresion,report, re.DOTALL)
    if text:
                text=text.group(1)
    else:
                text="Not specified in the report"
    return text


#configure this path according to the download of this file
csv_metadata='/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
csv_jpg_chest='/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'
root_dir='/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/'
split='/physionet/files/mimiciv/1.0/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv'

df_merge=pd.merge(pd.read_csv(csv_metadata),df_balanced,on=['subject_id', 'study_id'])
df_merge=pd.merge(pd.read_csv(split),df_merge,on=['subject_id', 'study_id'])


pd.set_option('display.max_columns', None)
df_merge


image_embedding=[]
text_embedding=[]   

for i in tqdm(range(len(df_merge))):
    image_path=root_dir+df_merge.iloc[i, 6] +"/"+df_merge.iloc[i, 5] 
    densefeature_embeddings, prediction_embeddings= generate_images_emmbedings(image_path)
    image_embedding.append(densefeature_embeddings)
    text = df_merge.iloc[i, 7]
    text=extract_reason_exam(text)
    embedding, hidden_embedding = get_biobert_embeddings(text)
    text_embedding.append(embedding)


df_merge['image_embedding']=image_embedding
df_merge['text_embedding']=text_embedding

df_merge= df.drop_duplicates(subset=['Img_Filename'])



all=df_merge[["subject_id" ,"text_embedding","image_embedding","label"]]   
all = all.reset_index(drop=True)

num_splits = 5

train_test_splits = []

for i in range(5):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=i)
    all[f"Split_{i}"] = "Train"
    for train_idx, test_idx in gss.split(all, groups=all["subject_id"]):
        all.loc[test_idx, f"Split_{i}"] = "Test"  


# Save the multimodal dataset to a Python pickle file
all.to_pickle('/Hollistic_Dataset.pk')
