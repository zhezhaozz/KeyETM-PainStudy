import pandas as pd

#if __name__ == '__main__':
# read data
studies = pd.read_csv('updated_pain.csv')

# data cleaning
# rename columns for convivience
studies = studies.rename(columns={'Project.Title': 'title', 
                                  'Public.Health.Relevance.y': 'relevance', 
                                  'Project.Abstract': 'abstract',
                                  'Primary.Assignment..anthony.domenichiello.nih.gov.': 'primary_label',
                                  'Secondary.Assignment..anthony.domenichiello.nih.gov.': 'secondary_lebel',
                                  'Tertiary.Assignment..anthony.domenichiello.nih.gov.': 'tertiary_label'
                                  }
                            )
# convert data types to string
studies['title'] = studies['title'].astype('string')
studies['relevance'] = studies['relevance'].astype('string')
studies['abstract'] = studies['abstract'].astype('string')
studies['pain_confirmed'] = studies['pain_confirmed'].astype('string')
studies['HEAL.Pain.'] = studies['HEAL.Pain.'].astype('string')
#studies['HEAL.Pain.'] = studies['HEAL.Pain.'].astype('string')
studies.info()

# Checking for missing values per column
print(studies.isnull().sum())

# drop NA values
studies = studies.dropna(subset=['title', 'abstract'])
print(studies.isnull().sum())

# combine all text columns
text_cols = ['title', 'relevance', 'abstract']
studies['text_cleaned'] = studies[f'{text_cols[0]}'] + ' ' + studies[f'{text_cols[1]}'] + ' ' + studies[f'{text_cols[2]}']

# only consider heal pain studies
cleaned_dt = studies[(studies['pain_confirmed'] == 'Yes') | (studies['HEAL.Pain.'] == 'True')]

# select relevant columns
cleaned_dt = cleaned_dt[['ID', 'text_cleaned', 'primary_label','secondary_lebel','tertiary_label']]

# remove \t and \r and \n
pattern = r'[\r|\n|\t]'
cleaned_dt['text_cleaned'] = cleaned_dt['text_cleaned'].str.replace(pattern, ' ', regex=True)

# separate training set and testing set
train_dt = cleaned_dt[cleaned_dt['primary_label'].isnull()] #648
test_dt = cleaned_dt[cleaned_dt['primary_label'].notnull()] #47

print(f"Number of rows for training set: {train_dt.shape[0]}")
print(f"Number of rows for testing set: {test_dt.shape[0]}")

# save data
cleaned_dt.to_csv('pain_preprocessed_data.csv')
train_dt.to_csv('pain_train_data.csv')
test_dt.to_csv('pain_test_data.csv')