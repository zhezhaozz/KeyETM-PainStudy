import pickle
import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data_folder = 'Data/pain_study/'
result_folder =os.path.join(data_folder, "cyclical/")
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
test_set.loc[:, 'model_output'] = test_set['model_output'] + 1
test_set['predicted_label'] = np.where((test_set['model_output'] == test_set['primary_label']) | (test_set['model_output'] == test_set['secondary_lebel']), test_set['primary_label'], test_set['model_output'])

# precision
precision_score(test_set['primary_label'], test_set['predicted_label'], average='macro', zero_division=np.nan)
recall_score(test_set['primary_label'], test_set['predicted_label'], average='macro', zero_division=np.nan)
f1_score(test_set['primary_label'], test_set['predicted_label'], average='micro', zero_division=np.nan)
f1_score(test_set['primary_label'], test_set['predicted_label'], average='macro', zero_division=np.nan)

classes=list(range(1, 16))
cm = confusion_matrix(test_set['primary_label'], test_set['predicted_label'], labels=classes)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

plot_confusion_matrix(cm, classes=classes, title='KeyETM')
