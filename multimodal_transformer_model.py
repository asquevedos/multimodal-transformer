import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np
from torch.utils.data import  DataLoader,random_split
from torchvision import transforms
import torch.nn as nn
import optuna

import pytorch_lightning as pl
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification import MultilabelAUROC
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from pytorch_lightning.loggers import CometLogger
from multimodal_data_processor import *



categories = ['Edema', 'Consolidation', 'Pleural Effusion','Atelectasis','Cardiomegaly','No Finding']        
composed = transforms.Compose([ToTensor()])

#Choose between three different embedding interactions
#embedding_interactions='summation'
embedding_interactions='concatenation'
#embedding_interactions='image_embedding'


# We can Choice between three different embedding interactions and five different splits
mimic_train_dataset= MIMIC_MultiModalDataset(mimic_pk_path='/home/sebastian/data/Hollistic_Dataset.pk',
                                    split_nunber="Split_1",split_interaction="Train",         
                                    embedding_interactions=embedding_interactions,
                                    transform=composed 
                                    )

mimic_test_dataset = MIMIC_MultiModalDataset(mimic_pk_path='/home/sebastian/data/Hollistic_Dataset.pk',
                                    split_nunber="Split_1",split_interaction="Test",         
                                    embedding_interactions=embedding_interactions,
                                    transform=composed 
                                    )


#Dataloaders
batch_size = 4096
train_dataloader = DataLoader(mimic_train_dataset, batch_size=batch_size, shuffle=True,num_workers = 40)
test_dataloader_ = DataLoader(mimic_test_dataset, batch_size=batch_size,shuffle=False)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x,inp_x,inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x
    
class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout,
        embedding=embedding_interactions
    ):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        #self.flatten = nn.Flatten((1, -1))
        #self.layer_norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),  # Add a hidden layer
            nn.ReLU(),  # Add a ReLU activation function
            nn.Dropout(p=dropout),  # Add dropout to reduce overfitting
            nn.Linear(hidden_dim, num_classes)  # Output layer

            )
    def forward(self, x):

        # Apply Transforrmer
        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.squeeze(0)
        out = self.mlp_head(x)
        return out

       
class MMT(pl.LightningModule):
    def __init__(self, model_kwargs, lr,weight_decay,categories_weigths):
        super().__init__()
        self.save_hyperparameters()
        self.model = MultiModalTransformer(**model_kwargs)
        #self.num_classes = model_kwargs["num_classes"]
        self.example_input_array = next(iter(train_dataloader))["embedding"]
        self.acc = MultilabelAccuracy(num_labels=model_kwargs["num_classes"], average=None)
        self.acc_avr = MultilabelAccuracy(num_labels=model_kwargs["num_classes"])
        self.roc_avr= MultilabelAUROC(num_labels=model_kwargs["num_classes"], average="macro", thresholds=None)
        self.embedding=model_kwargs["embedding"]
        self.loss_fn=nn.MultiLabelSoftMarginLoss()
        self.lr=lr
        self.weight_decay=weight_decay
        
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return [optimizer]#, [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs= batch["embedding"]
        y= batch["label"]
        preds = self.model(imgs)
        acc=self.acc(preds, y)
        acc_avg=self.acc_avr(preds, y)
        loss=self.loss_fn(preds, y)
        roc_avg= self.roc_avr(preds, y.int())
        self.log("%s_loss" % mode, loss, on_epoch=True, on_step=False)
        self.log("%s_roc_avg" % mode, roc_avg, on_epoch=True, on_step=False)
        if(mode != "train"):
            for ij in range(len(categories)):
                self.log("%s_roc_avg" % mode, roc_avg, on_epoch=True, on_step=False)
                self.log("%s_acc_" % mode + categories[ij]  , acc[ij], on_epoch=True, on_step=False)
        return loss
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "/home/sebastian/data/HAIM/saved_models/")        
         
def train_model(train_dataloader_, val_dataloader_,test_dataloader_,**kwargs):
    #Commet logger
    logger = CometLogger(
    api_key="bcaqkNUKpOj0yxgcMGWJKWVLs",
    project_name="multi-modal-optuna")
    # train the model
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "MMT"),
         devices=1,accelerator="gpu",
         logger=logger,
        max_epochs=15,
        callbacks=[
            #EarlyStopping(monitor="val_loss", min_delta=0.001, patience=30, verbose=False, mode="min"),##Ojo con los textos de las metricas
            ModelCheckpoint(
                save_weights_only=True, 
                mode="min", 
                monitor="val_loss"

                ),
            LearningRateMonitor("epoch"),
        ],

    )
    #Best Model
    #Please change the path to the best model
    #pretrained_filename = os.path.join(CHECKPOINT_PATH, "MMT/multi-modal-optuna/b33f7cc11c934531a94c065846755d5d/checkpoints/epoch=0-step=56.ckpt")
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "MMT")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model_trans = MMT.load_from_checkpoint(pretrained_filename)
        optimizer = optim.AdamW(model_trans.parameters(), lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
        focal_loss=nn.MultiLabelSoftMarginLoss()
        model_trans.optimizer = optimizer
        model_trans.loss_fn = focal_loss
        trainer.fit(model_trans, train_dataloader_,val_dataloader_) 
    else:
        pl.seed_everything(43)  # To be reproducable
        model_trans = MMT(**kwargs)
        trainer.fit(model_trans, train_dataloader_,val_dataloader_)

    model_trans = MMT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model_trans, dataloaders=test_dataloader_, verbose=False)                                      
    logger.finalize("success")
    return model_trans,test_result[0]["test_roc_avg"]




embedding_size=np.shape(mimic_train_dataset[0]["embedding"])[1]
    

total_samples = len(mimic_train_dataset)
# Calculate sizes for train and validate sets
train_size = int(0.99 * total_samples)
validate_size = total_samples - train_size

def split_data(dataset, train_size, validate_size):
    gen = torch.Generator()
    gen.manual_seed(43)
    train_dataset, test_dataset = random_split(dataset, [train_size, validate_size],generator=gen)
    return train_dataset, test_dataset

train_data, val_data = split_data(mimic_train_dataset, train_size, validate_size )
train_dataloader_ = DataLoader(train_data,batch_size=batch_size , shuffle=True,num_workers = 32)
val_dataloader_ = DataLoader(val_data, batch_size=batch_size ,shuffle=False,num_workers = 16)


#Optuna hyperparameter optimization
def objective(trial):
    lr = trial.suggest_float("lr", 1e-6, 1e-5, log=True)
    num_heads = trial.suggest_int("num_heads", 2, 16, step=2)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    results = train_model(
    train_dataloader_,
    val_dataloader_,
        test_dataloader_,  
        model_kwargs={
            "embed_dim": embedding_size,
            "hidden_dim": embedding_size*2,
            "num_heads": 16,
            "num_layers": num_layers,
            "num_classes": 6,
            "dropout": dropout,
            "embedding": embedding_interactions
        },
        lr=lr,
        weight_decay=weight_decay,
        categories_weigths =torch.tensor([0.131334344, 0.789755979, 0.197211601, 0.394033846,0.117707435,0.069956795])
    )

    return results[1]

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=160)  
