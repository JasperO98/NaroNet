# Benchmarking, improving and expanding the application of Naronet.

In this file, we present the steps followed to :
<ol>
<li>Reproduce the NaroNet experiment with the Endometrial Carcinoma dataset (Exp1)</li>
<li>Preprocess a Lung cancer in-house dataset and then make predictions with NaroNet (Exp2)</li>
<li>Reconstruct the embedding space of the PCL component with an AutoEncoder (AE) component aiming to improve the accuracy prediction of the system. Then making predictions with NaroNet (Exp3)</li>
</ol>

This Readme file closely follows the structure of the original NaroNet Readme, with some minor adjustments. The code repository for the NaroNet implementation can be accessed [*here*](https://github.com/djimenezsanchez/NaroNet/tree/main). The code for the AE component can be found [*here*](https://github.com/XifengGuo/IDEC-toy).

***Summary:*** NaroNet is an end-to-end interpretable learning method that can be used for the discovery of elements from the tumor microenvironment (phenotypes, cellular neighborhoods, and tissue areas) that have the highest predictive ability to classify subjects into predefined types. NaroNet works without any ROI extraction or patch-level annotation, just needing multiplex images and their corresponding subject-level labels. See the [*paper*](https://www.sciencedirect.com/science/article/pii/S1361841522000366) for further description of NaroNet.  

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Method_Overview_big.png' />

## Index (the usage of this code is explained step by step) 
[Requirements and installation](#Requirements-and-installation) • [Preparing datasets](#Preparing-datasets) • [Preparing parameter configuration](#Preparing-parameter-configuration) •[Running the experiments](#Running-the-experiments) • [Patch Contrastive Learning](#Patch-Contrastive-Learning) • [NaroNet](#NaroNet) • [BioInsights](#BioInsights) 

## Requirements and installation
* Linux. Tested on Ubuntu 22.04.4
* NVIDIA GPU. Tested on NVIDIA RTX A6000 compared to the server of the authors: Nvidia GeForce RTX 2080 Ti x 4 on GPU server)

An [*anaconda*](https://www.anaconda.com/distribution/) environment is recommended to run the experiments. You can create one starting from the descriptive file. The environment "rerunnaro" was used for Exp1 and Exp2. 

## Preparing datasets
Create the target folder (e.g., 'DATASET_DATA_DIR') with your image and subject-level information using the following folder structure:

```bash
DATASET_DATA_DIR/
    └──Raw_Data/
        ├── Images/
                ├── image_1.tiff
                ├── image_2.tiff
                └── ...
	├── Masks/ (Optional)
                ├── image_1.tiff
                ├── image_2.tiff
                └── ...
        └── Experiment_Information/
                ├── Channels.txt                
                ├── Image_Labels.xlsx
		└── Patient_to_Image.xlsx (Optional)
		
```
In the 'Raw_Data/Images' folder we expect multiplex image data consisting of multi-page '.tiff' files with one channel/marker per page.
In the 'Raw_Data/Masks' folder masks with the same size as the images with the same name can be stored. These should have 1's for the pixels that should be analyzed and 0's for the pixels that should be ignored.
In the 'Raw_Data/Experiment_Information' two files are expected:
* Channels.txt contains per row the name of each marker/channel present in the multiplex image. In case the name of the row is 'None' it will be ignored and not loaded from the raw image. See example [files](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Channels.txt) or example below:
```bash
Marker_1
Marker_2 
None
Marker_4    
```

* Image_Labels.xlsx contains the image names and their corresponding image-level labels. In column 'Image_Names' image names are specified. The next columns (e.g., 'Control vs. Treatment', 'Survival', etc.) specify image-level information, where 'None' means that the image is excluded from the experiment. In case more than one image is available per subject and you want to make sure that images from the same subject don't go to different train/val/test splits, it is possible to add one column named "Subject_Names" specifying, for each image, the subject to whom it corresponds. See example [file](https://github.com/djimenezsanchez/NaroNet/blob/main/examples/Image_Labels.xlsx) or example below:

| Image_Names | Control vs. Treatment | Survival | 
| :-- | :-:| :-: |
| image_1.tiff | Control  | Poor |
| image_2.tiff | None | High |
| image_3.tiff | Treatment | High |
| ... | ... | ... |

* Patient_to_Image.xlsx (optional) can be utilized in case more than one image is available per subject and you want to merge them into one subject graph. When images have the same subject identifier (e.g., 'Subject_Name') they will be joined into one disjoint graph. Please notice that when this file exists, you should change the 'Image_Names' column in 'Image_Labels.xlsx' with the new subject names (e.g., change 'image_1.tiff' with 'subject_1'). See example files in the [folder](https://github.com/CarolRameder/NaroNet/tree/main/Metadata%20for%20multiple%20images%20per%20patient) or the  small example below:

| Image_Name | Subject_Name |
| :-- | :-:| 
| image_1.tiff | subject_1 |
| image_2.tiff | subject_1 | 
| image_3.tiff | subject_2 | 
| ... | ... | ... |

This setup was tried with the hardware configuration mentioned in [Requirements and installation](#Requirements-and-installation). It raised a GPU memory overload issue for Exp1 and Exp2 as the Graphs get too large to be loaded.

## Preparing parameter configuration
In the following sections (i.e., preprocessing, PCL, NaroNet, BioInsights) several parameters are required to be set. Although parameters will be explained in each section, all of them should be specified in the file named 'DatasetParameters.py', which is located in the folder 'NaroNet/src/utils'. Change it to your own configuration, where 'DATASET_DATA_DIR' is your target folder. See example [file](https://github.com/CarolRameder/NaroNet/blob/main/NaroNet-main/src/NaroNet/utils/DatasetParameters.py) or example below:
```python
def parameters(path, debug):
    if 'DATASET_DATA_DIR' in path:        
        args['param1'] = value1
	args['param2'] = value2
	...		
```

## Running the experiments

There are 5 main methods called in the main.py file. We will refer to them according to the numbers to indicate how they should be commented on at a specific stage of the experiment. 

```python
#1
preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

#2
patch_contrastive_learning(path,params)    

#3
params = architecture_search(path,params,possible_params)

#4
run_NaroNet(path,params)

#5
get_BioInsights(path,params)
```

The path value should be set to 'DATASET_DATA_DIR' as shown in line 83 from [file](https://github.com/CarolRameder/NaroNet/blob/main/NaroNet-main/src/main.py).

```python
path = '/home/carol/NaroNet-main/NaroNet-main/Endometrial_POLE/'   
```

# Exp 1

The instructions required to run Exp1 and Exp2 are the same. Only the input data differs from one to the other. Exp1 consists of reproducing the NaroNet experiment with the Endometrial Carcinoma dataset. 


Step 0 (Optional): Set the required hyperparameters in DatasetParameters.py. Go to the if-branch on line 1684 from [file](https://github.com/CarolRameder/NaroNet/blob/main/NaroNet-main/src/NaroNet/utils/DatasetParameters.py)

Step 1: Create the environment

```sh
conda env create -f rerunnaro.yaml
conda activate rerunnaro
```
Step 2: Train the PCL component by running the main.py file within the virtual environment. Now, #3, #4 and #5 are commented to be disabled. 

```python
#1
preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

#2
patch_contrastive_learning(path,params)    

#3
#params = architecture_search(path,params,possible_params)

#4
#run_NaroNet(path,params)

#5
#get_BioInsights(path,params)
```

```sh
cd your_path/NaroNet/NaroNet-main/src/main.py
python3 main.py
```

Step 3: PCL inference process. With the trained weights, the model generates the embeddings. Rerun the main.py file in the same configuration and virtual environment.

Step 4: Run NaroNet and get BioInsights. Here, the commented methods are #1, #2. 

```python
#1
#preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

#2
#patch_contrastive_learning(path,params)    

#3
params = architecture_search(path,params,possible_params)

#4
run_NaroNet(path,params)

#5
get_BioInsights(path,params)
```

# Exp2

Exp2 consists of preprocessing a Lung cancer in-house dataset and then making predictions with NaroNet. Besides the preprocessing steps below, running Exp2 requires the same steps as Exp1. Steps 1 to 6 pertain to preprocessing. The steps to run NaroNet on the obtained data are the same as in Exp1. 

Step 0 (Optional): Set the required hyperparameters in DatasetParameters.py. Go to the if-branch on line 2063 from [file](https://github.com/CarolRameder/NaroNet/blob/main/NaroNet-main/src/NaroNet/utils/DatasetParameters.py)

Step 1: Install Zeiss Zen Lite from the [link](https://www.zeiss.com/microscopy/en/products/software/zeiss-zen-lite.html)

Step 2: Using Zeiss Zen Lite Interface, File -> Export image as ".tiff"
Result: 6 separate RGB images for each “.czi” input image converted, one for each marker. One folder\patient => 13 folders. For each patient, the 4 images need to be merged into one 4-dimensional image.

Step 3: 
Paste this [script](https://github.com/CarolRameder/NaroNet/blob/main/Preprocess%20scripts/final_short_imgprocess.ipynb) in each folder from Step 2.

Step 4:
Run the only existing cell. The script will:
- Load the .tiff image separate markers 
- Merge the markers to the same image file
- Save the file with the “.tif” extension

Result: 13 4-dim images in the parent folder.
Organise the outputs in the same folder for the next step. The path of the newly created folder is given as an input to the code required for Step 5.

Step 5: Run the cell from this [script](https://github.com/CarolRameder/NaroNet/blob/main/Preprocess%20scripts/Preprocessing.ipynb)
It loads the 4-dim image from Step 4
For each marker (displayed in [preprocess diagram](https://docs.google.com/drawings/d/1h2F0KIngJ7lNj3eaZ_c-uhKcFAJ-xlN-jDCNMfXsX7U/edit?usp=sharing) in the upper part):
- Determines the proportions for the RGB channels for the greyscale conversion
- Merges the RGB channels into a greyscale channel 
- Applies the rolling ball algorithm

Merges the transformed markers in a single file (3-dim)
Result: 13 3-dim images in the same folder as where the script is created.

Step 6:
Use this script for the last step, splitting the images into tiles. All the details are in this [script](https://github.com/CarolRameder/NaroNet/blob/main/Preprocess%20scripts/split_images.ipynb). The cutting positions were determined manually by selecting the region of interest areas.
Displayed in the lower left part of the [preprocess diagram](https://docs.google.com/drawings/d/1h2F0KIngJ7lNj3eaZ_c-uhKcFAJ-xlN-jDCNMfXsX7U/edit?usp=sharing).

The resulting 42 tiles will be moved to the “DATASET_DATA_DIR/Raw_Data/Images” folder as shown in the [Preparing datasets](#Preparing-datasets) Section.

Step 7: Run Naronet as described in Steps 1 to 4 from the previous Section.

# Exp 3

The environment "idec" was used for Experiment 3, only to to reconstruct the embeddings. When generating the embeddings with the PCL component and running the GNN component, the environment should be changed to "rerunnaro". The instructions regarding the methods in the main.py apply here as well. For this experiment, both the Endometrial Pole and the Lung Cancer datasets can be used. If the Lung cancer dataset is used, the preprocessing steps described in the section above should only be done once. 

Step 0 (Optional): Set the required hyperparameters in DatasetParameters.py. Choose the if-branch according to the Dataset used.

Step 1: Create the environment
```sh
conda env create -f idec.yaml
conda activate rerunnaro
```

Step 2: Train the PCL component (with the initial environment !!). The commented methods in the NaroNet main.py file are #3, #4 and #5.

```python
#1
preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

#2
patch_contrastive_learning(path,params)    

#3
#params = architecture_search(path,params,possible_params)

#4
#run_NaroNet(path,params)

#5
#get_BioInsights(path,params)
```

```sh
cd your_path/NaroNet/NaroNet-main/src/main.py
python3 main.py
```

Step 3: PCL inference process. With the trained weights, the model generates the embeddings. Rerun the main.py file as in Step 2.

Step 4: Run the three cells (pasted below) from the "Column-wise processing" section in the [file](https://github.com/CarolRameder/NaroNet/blob/main/EXP3/Pipeline.ipynb). This normalizes the embedding to the scale required by the Auto Encoder(AE) component. All the embeddings from all images in the dataset are merged in one ".npy". This is the input format for the AE.

```python
#merging outputs of the PCL to the AE format
#PCL -> AE
def load(folder_path):
    all_data = []
    min_val, max_val = -10, 7.5
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".npy"):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)


            for line in data:
                all_data.append(line)
    
    # Concatenate all data
    all_data = np.vstack(all_data)
    
    return all_data

folder_path = '/home/carol/NaroNet-main/NaroNet-main/Endometrial_POLE/Patch_Contrastive_Learning/Image_Patch_Representation'
allembs = load(folder_path)
print(np.shape(allembs))
np.save('ConcInp_noNorm.npy', allembs)
```

```python
def normal(data):
    # Assuming `data` is your (300k, 256) NumPy array
    data_min = data.min(axis=0)  # Minimum value of each column
    data_max = data.max(axis=0)  # Maximum value of each column

    # Min-Max normalization to scale data to the [0, 1] range
    data_normalized = (data - data_min) / (data_max - data_min)

    # Handle potential divisions by zero if a column is constant
    data_normalized = np.nan_to_num(data_normalized, nan=0.0)
    return data_normalized

data = np.load("/home/carol/NaroNet-main/EXP3/ConcInp_noNorm.npy")
normal_data  = normal(data)
np.save('ConcInpCol.npy', normal_data)
```

Step 5: Check if the path for the AE is set correctly in the "load_embs" from [file](https://github.com/CarolRameder/NaroNet/blob/main/EXP3/IDEC-toy/datasets.py). The path should point to the ".npy" file generated in Step 4.

Step 6: Switch the environment

```sh
conda activate idec
```

For this experiment, you can set the hyperparameters by setting the default values of the argument parser to the desired values and not adding any other argument to the second command line from Step 7. This can be done from line 190 to line 198 in [file](https://github.com/CarolRameder/NaroNet/blob/main/EXP3/IDEC-toy/IDEC.py), as shown below:

```python
parser.add_argument('--n_clusters', default=10, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--maxiter', default=2e4, type=int)
parser.add_argument('--pretrain_epochs', default=10, type=int)
parser.add_argument('--gamma', default=0.08, type=float, help='coefficient of clustering loss')
parser.add_argument('--update_interval', default=0, type=int)
parser.add_argument('--tol', default=0.001, type=float)
parser.add_argument('--ae_weights', default=None)
parser.add_argument('--save_dir', default='results/embs')
```

Step 7: Run the AE to get the reconstructed embeddings

```sh
cd your_path/NaroNet/EXP3/IDEC-toy
python3 IDEC.py embs
```

Step 8: Rescale the embeddings. For this, the Rescale section from [file](https://github.com/CarolRameder/NaroNet/blob/main/EXP3/Pipeline.ipynb). The embedding of each image will be generated in the "EXP3/Data/Split reconstruction" directory


```python
def rescale_col(data_normalized):
    # Assuming `data_normalized` is your normalized data array
    # Define min and max values for the rescaling process
    min_val_first_two = -10
    max_val_first_two = 1750
    min_val_rest = -10
    max_val_rest = 7.5

    # Initialize an array to hold the rescaled values, same shape as `data_normalized`
    data_rescaled = np.zeros_like(data_normalized)

    # Rescale the first two columns
    for i in range(2):
        data_rescaled[:, i] = data_normalized[:, i] * (max_val_first_two - min_val_first_two) + min_val_first_two
        # Cap values at 1750 for the first two columns
        data_rescaled[:, i] = np.where(data_rescaled[:, i] > max_val_first_two, max_val_first_two, data_rescaled[:, i])

    # Rescale the rest of the columns
    for i in range(2, data_normalized.shape[1]):
        data_rescaled[:, i] = data_normalized[:, i] * (max_val_rest - min_val_rest) + min_val_rest
        # No specific capping mentioned for these, but you could apply similar logic if needed

    # Now `data_rescaled` contains the rescaled values, with the first two columns capped at 1750
    return data_rescaled
```

```python
#splitting and renormalzie AE output to the Naronet format
#AE->Naronet
#Image patch Rep needs to be non-empty
init_embs_path = "/home/carol/NaroNet-main/NaroNet-main/Endometrial_POLE/Patch_Contrastive_Learning/Image_Patch_Representation"
rec_path = "/home/carol/NaroNet-main/EXP3/IDEC-toy/results/embs/reconstructed_x.npy"
out_path = "/home/carol/NaroNet-main/EXP3/Data/Split reconstruction"
reconstructed_embeddings = np.load(rec_path)
reconstructed_embeddings = rescale_col(reconstructed_embeddings) #rescale to the original value range
current_position= 0
for file_name in os.listdir(init_embs_path):
    file_path = os.path.join(init_embs_path, file_name)
    cr_emb = np.load(file_path)  
    line_count = cr_emb.shape[0]
    
    extracted_embeddings = reconstructed_embeddings[current_position : current_position + line_count]
    current_position += line_count
    new_file_path = os.path.join(out_path, file_name)
    np.save(new_file_path, extracted_embeddings)
```

Step 9: Replace the initial embeddings from "DATASET_DATA_DIR/Patch_Contrastive_Learning/Preprocessed_Images/" with the new embeddings from Step 8.
This was done manually:
```
# Remove the old embeddings
rm DATASET_DATA_DIR/Patch_Contrastive_Learning/Preprocessed_Images/*

# Copy the new embeddings into the folder
cp -rf NaroNet-main/EXP3/Data/Split reconstruction/* DATASET_DATA_DIR/Patch_Contrastive_Learning/Preprocessed_Images/.
```

Step 10: Switch the environment back

```sh
conda activate rerunnaro
```

Step 11: Run NaroNet and get BioInsights (as previously). Here, the commented methods in the NaroNet main.py file are #1 and #2. 

```python
#1
#preprocess_images(path,params['PCL_ZscoreNormalization'],params['PCL_patch_size'])

#2
#patch_contrastive_learning(path,params)    

#3
params = architecture_search(path,params,possible_params)

#4
run_NaroNet(path,params)

#5
get_BioInsights(path,params)
```

```sh
cd your_path/NaroNet/NaroNet-main/src/main.py
python3 main.py
```

## Patch Contrastive Learning
The goal of PCL in our pipeline is to convert each high-dimensional multiplex image of the cohort into a list of low-dimensional embedding vectors. To this end, each image is divided into patches -our basic units of representation containing one or two cells of the tissue-, and each patch is converted by the PCL module -a properly trained CNN- into a low-dimensional vector that embeds both the morphological and spectral information of the patch.

To this end, 'NaroNet.patch_contrastive_learning' function is used with the following parameters:
* `args['PCL_embedding_dimensions']`: size of the embedding vector generated for each image patch. Default: 256 (values)
* `args['PCL_batch_size']`: batch size of image patches used to train PCL's CNN. Example: 80 (image patches)
* `args['PCL_epochs']`: epochs to train PCL's CNN. Example: 500 (epochs) 
* `args['PCL_alpha_L']`: size ratio between image crops and augmented views used to train PCL's CNN. Default: 1.15. 
* `args['PCL_width_CNN']`: CNN's width multiplication factor. Default: 2.
* `args['PCL_depth_CNN']`: CNN's depth [50 or 101]. Default: 101 (ResNet101),.

When executed, PCL checks whether a trained CNN is already in a previously created folder named 'Model_Training_xxxx', where xxxx are random letters. In case the folder does not exist, PCL creates a new model, stores it in a new 'Model_Training_xxxx' folder, and trains it using the parameter configuration. To check whether the CNN has been trained successfully, check the 'Model_training_xxxx' folder and open the 'Contrast_accuracy_plot.png', where you should expect a final contrast accuracy value over 50%. 

Once the CNN is trained, execute again 'NaroNet.preprocess_images' to infer image patch representations from the whole dataset. Here, image patches are introduced in the CNN sequentially getting representation vectors back. For each image in the dataset, a npy data structure is created consisting of a matrix, where rows are patches, and columns are representation values. Here, the two first column values specify the x and y position of the patch in the image that will be later used to create a graph. In case Patient_to_Image.xlsx exists the npy structure will contain patches from more than one image.

Once executed you should expect the following folder structure, where Model_Training_xxxx is created during training, and Image_Patch_Representation during inference (in green):

```diff
DATASET_DATA_DIR/
    ├── Raw_Data/        
        └── ...
    └── Patch_Contrastive_Learning/
 	├── Preprocessed_Images/    		
		└── ...
+	├── Model_Training_xxxx/
+    		├── model.ckpt-0.index
+		├── model.ckpt-0-meta
+		├── model.ckpt-0.data-00000-of-00001
+		├── event.out.tfevents...
+		├── checkpoint
+		└── ...
+	└── Image_Patch_Representation/
+    		├── image_1.npy
+		├── image_2.npy
+		└── ...
```

## NaroNet
NaroNet inputs graphs of patches (stored in 'DATASET_DATA_DIR/Patch_Contrastive_Learning/Image_Patch_Representation') and subject-level labels (stored in 'DATASET_DATA_DIR/Raw_Data/Experiment_Information/Image_Labels.xlsx') to output subject's predictions from the abundance of learned phenotypes, neighborhoods, and areas. To this end, execute 'NaroNet.NaroNet.run_NaroNet' with the following parameters (most relevant parameters are shown, where additional ones are explained in DatasetParameters.py):

* `args['experiment_Label']`: Subject-level column labels from Image_Labels.xlsx to specify how you want to differentiate subjects. You can provide more than one column label name to perform multilabel classification. Example1: ['Survival'], Example2: ['Survival','Control vs. Treatment'].
* `args['epochs']`: Number of epochs to train NaroNet. Default: 20 (epochs).
* `args['weight_decay']`: Weight decay value. Default: 0.0001.
* `args['batch_size']`: Batch size. Default: 8 (subjects).
* `args['lr']`: Learning rate. Default: 0.001.
* `args['folds']`: Number of folds to perform cross-validation. Default: 10 (folds).
* `args['device']`: Specify whether to use cpu or gpu. Examples: 'cpu', 'cuda:0', 'cuda:1'.
* `args['clusters1']`: Number of phenotypes to be learned by NaroNet. Example: 10 (phenotypes). 
* `args['clusters2']`: Number of neighborhoods to be learned by NaroNet. Example: 11 (neighborhoods). 
* `args['clusters3']`: Number of areas to be learned by NaroNet. Example: 6 (areas). 

When executed, NaroNet selects the images included in the experiment and creates a graph of patches in the pytorch's format .pt. Next, k-fold-cross validation is carried out, training the model with 90% of the data and testing in the remaining 10%. See in green all folders created:

```diff
DATASET_DATA_DIR/
    ├── Raw_Data/        
        └── ...
    ├── Patch_Contrastive_Learning/		
	└── ...
+   └── NaroNet/		
+	├── Survival/ (experiment name example)
+		├── Subject_graphs/
+    			├── data_0_0.pt
+    			├── data_1_0.pt
+			└── ...
+		├── Cell_type_assignment/
+    			├── cluster_assignment_Index_0_ClustLvl_10.npy (phenotypes)
+    			├── cluster_assignment_Index_0_ClustLvl_11.npy (neighborhoods)
+    			├── cluster_assignment_Index_0_ClustLvl_6.npy (areas)
+    			├── cluster_assignment_Index_1_ClustLvl_10.npy (phenotypes)
+    			├── cluster_assignment_Index_1_ClustLvl_11.npy (neighborhoods)
+    			├── cluster_assignment_Index_1_ClustLvl_6.npy (areas)
+			└── ...
+		└── Cross_validation_results/
+			├── ROC_AUC_Survival.png
+			├── ConfusionMatrix_Survival.png
+			└── ...
```

## BioInsights
NaroNet's learned phenotypes, neighborhoods, and areas (stored in 'Cell_type_assignment'), can be analyzed _a posteriori_ by the BioInsights module. Here, elements of the tumor microenvironment are extracted, visualized, and associated to subject types. Execute 'NaroNet.NaroNet_dataset.get_BioInsights' with the same parameters as done in the NaroNet module to automatically generate the following folders:

* Cell_type_characterization: Contains heatmaps with the marker expression levels of phenotypes, neighborhoods, and areas. Also contains examples of patches assigned to the phenotypes, neighborhoods and areas.

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Figura_Phentypes.gif' />

* Cell_type_abundance: Contains heatmaps with the abundance of phenotypes, neighborhoods, and areas per subject. This information is then used to perform the differential TME composition analysis.

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Figura_ConfidencePredictions_ithub.gif' />

* Differential_abundance_analysis: provides information about the differential TME composition analysis (p-values specify predictive power of TMEs). It also provides statistical tests showing if found TMEs are cohort-differenting. Examples of predicted subjects are stored in the folder 'Locate_TME_in_image'.

<img src='https://github.com/djimenezsanchez/NaroNet/blob/main/images/Figura_Areas_github.gif' />


```diff
DATASET_DATA_DIR/
    ├── Raw_Data/        
        └── ...
    ├── Patch_Contrastive_Learning/		
	└── ...
    ├── NaroNet/    	
	└── Survival/ (experiment name example)
		└── ...
+   └── BioInsights/
+    	└── Survival/ (experiment name example)
+		├── Cell_type_characterization/
+			└── ...
+		├── Cell_type_abundance/
+			└── ...
+		├── Differential_abundance_analysis/
+			└── ...
+ 		└── Locate_TME_in_image/
+			└── ...

```


