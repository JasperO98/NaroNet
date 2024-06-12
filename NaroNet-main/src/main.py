from NaroNet.utils.DatasetParameters import parameters
from NaroNet.Patch_Contrastive_Learning.patch_contrastive_learning import patch_contrastive_learning
from NaroNet.Patch_Contrastive_Learning.preprocess_images import preprocess_images
# from NaroNet.architecture_search.architecture_search import architecture_search
# from NaroNet.NaroNet import run_NaroNet
# from NaroNet.NaroNet_dataset import get_BioInsights


#deleted that- import torch , import tensorflow as tf


def main(path):
    # Select Experiment parameters
    params = parameters(path, 'Value')
    possible_params = parameters(path, 'Object')
    best_params = parameters(path, 'Index')
    
    # Preprocess Images
    preprocess_images(path, params['PCL_ZscoreNormalization'], params['PCL_patch_size'])
    
    #Patch Contrastive Learning
    patch_contrastive_learning(path, params)

    # Architecture Search
    #params = architecture_search(path,params,possible_params)

    #run_NaroNet(path, params)

    # BioInsights
    #get_BioInsights(path, params)


def generate_xlsx(path):
    """
    Generates an xlsx file with two columns: "Image_Name" and "Subject_Name".
    The "Image_Name" column contains the names of the files found in the specified directory path.
    The "Subject_Name" column contains the ID extracted from the file names.
    """
    # Create an empty DataFrame to hold the data
    data = pd.DataFrame(columns=['Image_Name', 'Subject_Name'])

    # Iterate over each file in the directory
    for filename in os.listdir(path):
        # Extract the ID from the file name
        id = filename.split('_')[2:4]
        id = '_'.join(id)

        # Add the file name and ID to the DataFrame
        data = data.append({'Image_Name': filename, 'Subject_Name': id}, ignore_index=True)

    # Write the DataFrame to an xlsx file
    writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
    data.to_excel(writer, index=False)
    writer.save()

if __name__ == "__main__":

    #path = "../Example_POLE_small/"
    #path = "../Endometrial_POLE/"
    path = "../Example_POLE/"
    
    #path = "../Images-SyntheticCCI1/"

    #path = ".."
    main(path)