import pandas as pd
import numpy as np

def main():
      # read data
      pain_grants = pd.read_csv('Data/pain_study/pain_grants.csv')
      test_dt = pd.read_csv('Data/pain_study/test_data.csv')

      # get test labels
      test_label = test_dt.loc[:, ['primary_label','secondary_label','tertiary_label']].to_numpy()
      print(test_label.shape)

      # lower case
      pain_grants['combined_text'] = pain_grants['combined_text'].str.lower()

      # remove \t and \r and \n
      pattern = r'[\r|\n|\t]'
      pain_grants['combined_text'] = pain_grants['combined_text'].str.replace(pattern, ' ', regex=True)

      # remove grants section titles
      pattern = r'project narrative|narrative|public health relevance|project summary|abstract'
      pain_grants['combined_text'] = pain_grants['combined_text'].str.replace(pattern, ' ', regex=True)

      # save data
      pain_grants.to_csv('Data/pain_study/pain_preprocessed_data.csv')
      np.save('Data/pain_study/test_label', test_label)

if __name__ == '__main__':
      main()

      