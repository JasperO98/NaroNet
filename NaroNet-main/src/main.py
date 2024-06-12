import os
import json
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images
from NaroNet.architecture_search.architecture_search import architecture_search
from NaroNet.NaroNet import run_NaroNet
from NaroNet.NaroNet_dataset import get_BioInsights

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # Only use one GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import torch

# Set PYTORCH_CUDA_ALLOC_CONF environment variable
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def main(path):
    # Select Experiment parameters
    params = parameters(path, 'Value')
    possible_params = parameters(path, 'Object')
    best_params = parameters(path, 'Index')    

    # Preprocess Images
    # preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

    # #Patch Contrastive Learning
    # patch_contrastive_learning(path,params)    

    # #Architecture Search
    # params = architecture_search(path,params,possible_params)

    # with open('params_exp2_4.json', 'w') as file:
    #     json.dump(params, file)
    

    #EXP2: lung cancer parameters;   !!epochs and patch size 2/3; modified showHowNetworkIsTraining back to False 
    #epochs should be lowered - the model already overtrains
    params = {"path": "/home/carol/NaroNet-main/NaroNet-main/EXP2/", "PCL_embedding_dimensions": 256, "PCL_batch_size": 160, "PCL_epochs": 300, "PCL_patch_size": 32, "PCL_alpha_L": 1.3, "PCL_ZscoreNormalization": True, "PCL_width_CNN": 2,
    "PCL_depth_CNN": 50, "experiment_Label": ["Response"], "num_samples_architecture_search": 100, "epochs": 20, "epoch": 0, "lr_decay_factor": 0.5, "lr_decay_step_size": 12, "weight_decay": 0.01, "batch_size": 6, "lr": 0.001,
    "useOptimizer": "ADAM", "context_size": 15, "num_classes": 3, "MultiClass_Classification": 1, "showHowNetworkIsTraining": False, "visualizeClusters": True, "learnSupvsdClust": True, "recalculate": False, "folds": 13, 
    "device": "cuda", "normalizeFeats": False, "normalizeCells": False, "Batch_Normalization": True, "normalizePercentile": False, "dataAugmentationPerc": 0.0001, "hiddens": 44, "clusters1": 10, "clusters2": 9, "clusters3": 7,
    "LSTM": False, "GLORE": False, "Phenotypes": True, "DeepSimple": False, "isAttentionLayer": False, "ClusteringOrAttention": True, "1cell1cluster": False, "dropoutRate": 0.2, "AttntnSparsenss": False, "attntnThreshold": 0,
    "GraphConvolution": "ResNet", "n-hops": 3, "modeltype": "SAGE", "ObjectiveCluster": True, "ReadoutFunction": False, "NearestNeighborClassification": False, "NearestNeighborClassification_Lambda0": 1,
    "NearestNeighborClassification_Lambda1": 1, "NearestNeighborClassification_Lambda2": 1, "KinNearestNeighbors": 5, "pearsonCoeffSUP": False, "pearsonCoeffUNSUP": False, "orthoColor": True, "orthoColor_Lambda0": 0.1,
    "orthoColor_Lambda1": 1e-05, "ortho": False, "ortho_Lambda0": 0.1, "ortho_Lambda1": 0, "ortho_Lambda2": 0, "min_Cell_entropy": True, "min_Cell_entropy_Lambda0": 1, "min_Cell_entropy_Lambda1": 0.0001,
    "min_Cell_entropy_Lambda2": 0.01, "MinCut": True, "MinCut_Lambda0": 0, "MinCut_Lambda1": 0.1, "MinCut_Lambda2": 0.1, "F-test": False, "Max_Pat_Entropy": False, "Max_Pat_Entropy_Lambda0": 0.0001,
    "Max_Pat_Entropy_Lambda1": 0.1, "Max_Pat_Entropy_Lambda2": 0.1, "UnsupContrast": False, "UnsupContrast_Lambda0": 0, "UnsupContrast_Lambda1": 0, "UnsupContrast_Lambda2": 0, "Lasso_Feat_Selection": False,
    "Lasso_Feat_Selection_Lambda0": 0.1, "SupervisedLearning_Lambda0": 1, "SupervisedLearning_Lambda1": 1, "SupervisedLearning_Lambda2": 1, "SupervisedLearning_Lambda3": 1, "SupervisedLearning": True}

    #HYPER PARAMETERS SEARCH RESULTS
    #EXP1: endometrial cancer parameters


    #EXP1: copied from extra material  
    # params = {"path": "/home/carol/NaroNet-main/NaroNet-main/Endometrial_POLE/", "PCL_embedding_dimensions": 256, "PCL_batch_size": 160, "PCL_epochs": 500, "PCL_patch_size": 15, "PCL_alpha_L": 1.15,
    # "PCL_ZscoreNormalization": True, "PCL_width_CNN": 2, "PCL_depth_CNN": 50, "experiment_Label": ["POLE Mutation", "Copy number variation", "MSI Status", "Tumour Type"],
    # "num_samples_architecture_search": 500, "epochs": 10, "epoch": 0, "lr_decay_factor": 0.5, "lr_decay_step_size": 12, "weight_decay": 0.0001, "batch_size": 20, "lr": 0.001,
    # "useOptimizer": "ADAM", "context_size": 15, "num_classes": 3, "MultiClass_Classification": 1, "showHowNetworkIsTraining": False, "visualizeClusters": True, "learnSupvsdClust": True, 
    # "recalculate": False, "folds": 10, "device": "cuda", "normalizeFeats": False, "normalizeCells": False, "Batch_Normalization": False, "normalizePercentile": False,
    # "dataAugmentationPerc": 0.01, "hiddens": 44, "clusters1": 10, "clusters2": 9, "clusters3": 7, "LSTM": False, "GLORE": True, "Phenotypes": True, "DeepSimple": False, 
    # "isAttentionLayer": False, "ClusteringOrAttention": True, "1cell1cluster": False, "dropoutRate": 0.2, "AttntnSparsenss": False, "attntnThreshold": 0, "GraphConvolution": "ResNet",
    # "n-hops": 3, "modeltype": "SAGE", "ObjectiveCluster": True, "ReadoutFunction": False, "NearestNeighborClassification": False, "NearestNeighborClassification_Lambda0": 1,
    # "NearestNeighborClassification_Lambda1": 1, "NearestNeighborClassification_Lambda2": 1, "KinNearestNeighbors": 5, "pearsonCoeffSUP": False, "pearsonCoeffUNSUP": False, 
    # "orthoColor": True, "orthoColor_Lambda0": 0.1, "orthoColor_Lambda1": 1e-05, "ortho": False, "ortho_Lambda0": 0.1, "ortho_Lambda1": 0, "ortho_Lambda2": 0, "min_Cell_entropy": True,
    # "min_Cell_entropy_Lambda0": 1, "min_Cell_entropy_Lambda1": 0.0001, "min_Cell_entropy_Lambda2": 0.01, "MinCut": False, "MinCut_Lambda0": 0, "MinCut_Lambda1": 0.1, 
    # "MinCut_Lambda2": 0.1, "F-test": False, "Max_Pat_Entropy": True, "Max_Pat_Entropy_Lambda0": 0.0001, "Max_Pat_Entropy_Lambda1": 0.1, "Max_Pat_Entropy_Lambda2": 0.1, 
    # "UnsupContrast": False, "UnsupContrast_Lambda0": 0, "UnsupContrast_Lambda1": 0, "UnsupContrast_Lambda2": 0, "Lasso_Feat_Selection": False, "Lasso_Feat_Selection_Lambda0": 0.1, 
    # "SupervisedLearning_Lambda0": 1, "SupervisedLearning_Lambda1": 1, "SupervisedLearning_Lambda2": 1, "SupervisedLearning_Lambda3": 1, "SupervisedLearning": True}

    run_NaroNet(path,params)
    
    # BioInsights
    get_BioInsights(path,params)

if __name__ == "__main__":

    #path = '/home/carol/NaroNet-main/NaroNet-main/Endometrial_POLE/'    
    path = '/home/carol/NaroNet-main/NaroNet-main/EXP2/'    
    
    #path = '/home/carol/NaroNet-main/NaroNet-main/Endo2/'  
    #path = '/home/carol/NaroNet-main/NaroNet-main/Example_POLE/'
    #path = '/home/carol/NaroNet-main/NaroNet-main/Images-SyntheticCCI1'   
    main(path)