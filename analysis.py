import pickle
import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_folder = 'Data/pain_study/'
result_folder =os.path.join(data_folder, "result/")
pickle_file = os.path.join(result_folder, 'formatted_topic_word.pickle')

# open formatted topic words
with open(pickle_file, 'rb') as file:
    # Load the pickle object.
    loaded_object = pickle.load(file) # it's a dict

# save the topic word into an excel file
excel_writer = pd.ExcelWriter(os.path.join(result_folder,'topic-words.xlsx'), engine='openpyxl')

# Iterate over the dictionary and write each list of tuples to a separate sheet
for key, value in loaded_object.items():
    # Convert the list of tuples to a DataFrame
    df = pd.DataFrame(value, columns=["Term", "Probability"])
    # Write the DataFrame to a sheet named after the key
    df.to_excel(excel_writer, sheet_name=key, index=False)

# Save the Excel writer
excel_writer.close()

# read the predicted label
model_output = pd.read_csv(os.path.join(result_folder, 'ETM_labels_.csv'),names=["output"])
# read the original label
origin_label = pd.read_csv(os.path.join(data_folder, 'pain_preprocessed_data.csv'))
# get test set labels
test_indecies = origin_label[origin_label['primary_label'].notna()].index
test_set = origin_label.loc[test_indecies]
model_output = model_output.iloc[test_indecies]

# add model output to the test set
test_set['model_output'] = model_output['output']
test_set['predicted_label'] = np.where((test_set['model_output'] == test_set['primary_label']) | (test_set['model_output'] == test_set['secondary_lebel']), test_set['primary_label'], test_set['model_output'])

# precision
precision_score(test_set['primary_label'], test_set['predicted_label'], average='macro')
recall_score(test_set['primary_label'], test_set['predicted_label'], average='macro')
f1_score(test_set['primary_label'], test_set['predicted_label'], average='macro')
f1_score(test_set['primary_label'], test_set['predicted_label'], average='micro')
