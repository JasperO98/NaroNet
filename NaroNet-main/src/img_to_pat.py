import os
import pandas as pd

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


def merge_xlsx(file1_path, file2_path):
    """
    Reverses the order of lines for file1, checks if the values for the "Image_Name" column from file1
    are the same as the "Image_Names" column from file2, and replaces the values from the "Image_Names"
    column in file2 with the corresponding value from the "Subject_Name" column in file1 if they match.
    """
    # Load both files into Pandas dataframes
    file1 = pd.read_excel(file1_path)
    file2 = pd.read_excel(file2_path)

    # Reverse the order of lines in file1
    file1 = file1.iloc[::-1]

    # Check if the values in the "Image_Name" column of file1 are present in the "Image_Names" column of file2
    matching_indices = file2[file2['Image_Names'].isin(file1['Image_Name'])].index

    if len(matching_indices) == 0:
        print("No matches found between the two files.")
        return

    # Replace the values in the "Image_Names" column of file2 with the corresponding values from the "Subject_Name" column of file1
    file2.loc[matching_indices, 'Image_Names'] = file1.loc[
        file1['Image_Name'].isin(file2['Image_Names']), 'Subject_Name'].tolist()

    # Save the modified file2 as a new xlsx file
    writer = pd.ExcelWriter('merged.xlsx', engine='xlsxwriter')
    file2.to_excel(writer, index=False)
    writer.save()

    print(f"{len(matching_indices)} matches found between the two files. Merged file saved as 'merged.xlsx'.")

if __name__ == "__main__":
    path = "../Endometrial_POLE/Raw_data/Images/"
    generate_xlsx(path)