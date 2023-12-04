import torch
import pandas as pd
import numpy as np


categories = ['Edema', 'Consolidation', 'Pleural Effusion','Atelectasis','Cardiomegaly','No Finding']

def one_hot_encode(data):
    # Initialize the encoded tensor with zeros
    categories = ['Edema', 'Consolidation', 'Pleural Effusion','Atelectasis','Cardiomegaly','No Finding']
    encoded_tensor = np.zeros((len(data), len(categories)))

    # Iterate over the data and encode each observation
    for i, obs in enumerate(data):
        for cat in obs:
            j = categories.index(cat)
            encoded_tensor[i, j] = 1

    return encoded_tensor

# Generate the encoded tensor
#encoded_tensor = one_hot_encode(categories, dataset)

class MIMIC_MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, mimic_pk_path, split_nunber,split_interaction,embedding_interactions ,transform=None):
        self.df = pd.read_pickle(mimic_pk_path)
        self.df=self.df[self.df[split_nunber]==split_interaction]
        
        self.df_labels=self.df['label']
        self.transform = transform
        self.embedding_interactions=embedding_interactions
        self.labels=one_hot_encode(self.df_labels)
        
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        label= self.labels[idx]
        densefeature_embeddings= self.df.iloc[idx]["image_embedding"]
        full_embedding_text = self.df.iloc[idx]["text_embedding"]
        try:
            aggregated_embedding = np.average(full_embedding_text, axis=0)
        except:
            aggregated_embedding = np.zeros(768)
        
        if(self.embedding_interactions=='summation'):
            aggregated_embedding= np.append(aggregated_embedding, np.zeros((1, 256),dtype=np.float32))
            embeding_summation= aggregated_embedding+densefeature_embeddings
            embeding_summation=np.reshape(embeding_summation, (1, 1024))
            sample = {'embedding': embeding_summation, 'label': label}
        
        if(self.embedding_interactions=='concatenation'):
                concatenate_embedding=np.concatenate((densefeature_embeddings,aggregated_embedding), axis=0)
                concatenate_embedding=np.reshape(concatenate_embedding, (1, 1792))
                sample = {'embedding': concatenate_embedding, 'label': label}
        
        if(self.embedding_interactions=='image_embedding'):
                embeding_summation=np.reshape(densefeature_embeddings, (1, 1024))
                sample = {'embedding': embeding_summation, 'label': label}

                
        if(self.embedding_interactions=='CoAttentionFusion'):
            concatenate_embedding=np.concatenate((densefeature_embeddings,aggregated_embedding), axis=0)
            sample = {'embedding': concatenate_embedding, 'label': label}        
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        
        embbeding,label = sample['embedding'], sample['label']

        return {'embedding': torch.from_numpy(embbeding),
                'label': torch.from_numpy(label).float()
                }
