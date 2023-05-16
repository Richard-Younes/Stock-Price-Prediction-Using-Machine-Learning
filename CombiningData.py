import pandas as pd
import os

# set the path to the folder containing the CSV files
folder_path = os.getcwd()

# get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# initialize an empty DataFrame to hold the combined data
combined_df = pd.DataFrame()

# loop over each CSV file and append its data to the combined DataFrame
for csv_file in csv_files:
    csv_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(csv_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# save the combined data to a new CSV file
combined_df.to_csv('combined_data.csv', index=False)