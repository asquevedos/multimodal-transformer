# System                                                                                           
import os
import sys

# Base
import cv2

from dask.diagnostics import ProgressBar
ProgressBar().register()

# Core AI/ML
import torch
import torch.nn.functional as F


# NLP
from transformers import AutoTokenizer, AutoModel, logging
logging.set_verbosity_error()
# biobert_path = '../pretrained_models/bio_clinical_bert/biobert_pretrain_output_all_notes_150000/'
biobert_path = 'pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
biobert_model = AutoModel.from_pretrained(biobert_path)
# biobert_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# biobert_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Computer Vision
import cv2
import torchxrayvision as xrv

# Warning handling
import warnings
warnings.filterwarnings("ignore")

def get_single_chest_xray_embeddings(img):
    # Inputs:
    #   img -> Image array
    #
    # Outputs:
    #   densefeature_embeddings ->  CXR dense feature embeddings for image
    #   prediction_embeddings ->  CXR embeddings of predictions for image
    
    # densefeature_embeddings, prediction_embeddings = get_single_chest_xray_embeddings(img)
    
    # Clean out process bar before starting
    sys.stdout.flush()
    
    # Select if you want to use CUDA support for GPU (optional as it is usually pretty fast even in CPUT)
    cuda = True
    
    # Select model with a String that determines the model to use for Chest Xrays according to https://github.com/mlmed/torchxrayvision
    model_weights_name = "densenet121-res224-all" # Every output trained for all models
    #model_weights_name = "densenet121-res224-rsna" # RSNA Pneumonia Challenge
    #model_weights_name = "densenet121-res224-nih" # NIH chest X-ray8
    #model_weights_name = "densenet121-res224-pc") # PadChest (University of Alicante)
    #model_weights_name = "densenet121-res224-chex" # CheXpert (Stanford)
    #model_weights_name = "densenet121-res224-mimic_nb" # MIMIC-CXR (MIT)
    #model_weights_name = "densenet121-res224-mimic_ch" # MIMIC-CXR (MIT)
    #model_weights_name = "resnet50-res512-all" # Resnet only for 512x512 inputs
    # NOTE: The all model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset.
    
    # Extract chest x-ray image embeddings and preddictions
    densefeature_embeddings = []
    prediction_embeddings = []
    
    #img = skimage.io.imread(img_path) # If importing from path use this
    img = xrv.datasets.normalize(img, 255)

    # For each image check if they are 2D arrays
    if len(img.shape) > 2:
        img = img[:, :, 0]
    if len(img.shape) < 2:
        print("Error: Dimension lower than 2 for image!")
    
    # Add color channel for prediction
    #Resize using OpenCV
    img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
    img = img[None, :, :]
    cuda1 = torch.device('cuda:0')
    #Or resize using core resizer (thows error sometime)
    #transform = transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    #img = transform(img)
    model = xrv.models.DenseNet(weights = model_weights_name)
    # model = xrv.models.ResNet(weights="resnet50-res512-all") # ResNet is also available

    output = {}
    with torch.no_grad():
        img = torch.from_numpy(img).unsqueeze(0)
        if cuda:
            img = img.cuda(cuda1)
            model = model.cuda(cuda1)
          
        # Extract dense features
        feats = model.features(img)
        feats = F.relu(feats, inplace=True)
        feats = F.adaptive_avg_pool2d(feats, (1, 1))
        densefeatures = feats.cpu().detach().numpy().reshape(-1)
        densefeature_embeddings = densefeatures

        # Extract predicted probabilities of considered 18 classes:
        # Get by calling "xrv.datasets.default_pathologies" or "dict(zip(xrv.datasets.default_pathologies,preds[0].detach().numpy()))"
        # ['Atelectasis','Consolidation','Infiltration','Pneumothorax','Edema','Emphysema',Fibrosis',
        #  'Effusion','Pneumonia','Pleural_Thickening','Cardiomegaly','Nodule',Mass','Hernia',
        #  'Lung Lesion','Fracture','Lung Opacity','Enlarged Cardiomediastinum']
        preds = model(img).cpu()
        predictions = preds[0].detach().numpy()
        prediction_embeddings = predictions  

    # Return embeddings
    return densefeature_embeddings, prediction_embeddings

# OBTAIN BIOBERT EMBEDDINGS OF TEXT STRING
def get_biobert_embeddings(text):
    # Inputs:
    #   text -> Input text (str)
    #
    # Outputs:
    #   embeddings -> Final Biobert embeddings with vector dimensionality = (1,768)
    #   hidden_embeddings -> Last hidden layer in Biobert model with vector dimensionality = (token_size,768)
    # # %% EXAMPLE OF USE
    # embeddings, hidden_embeddings = get_biobert_embeddings(text)
  
    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    outputs = biobert_model(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings