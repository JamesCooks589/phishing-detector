import pandas as pd

# Load the dataset
df = pd.read_csv('data/CEAS_08.csv', encoding='latin1')  # adjust encoding if needed

# Count values in the 'label' column
label_counts = df['label'].value_counts()
print("\n📊 Label Distribution:")
print(label_counts)
print("\n% Breakdown:")
print((label_counts / label_counts.sum()) * 100)
