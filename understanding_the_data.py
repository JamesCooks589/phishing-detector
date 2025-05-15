import pandas as pd
import glob
import os

data_files = glob.glob('data/*.csv')

total_label_counts = {0: 0, 1: 0}  # initialize counters

for file in data_files:
    print(f"\n=== Analyzing: {os.path.basename(file)} ===")
    df = pd.read_csv(file, encoding='latin1')

    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print("📊 Label Distribution:")
        print(label_counts)
        print("% Breakdown:")
        print((label_counts / label_counts.sum()) * 100)

        # Update cumulative totals
        for label in [0, 1]:
            if label in label_counts:
                total_label_counts[label] += label_counts[label]
    else:
        print("No 'label' column found.")

# --- Final Combined Summary ---
total = sum(total_label_counts.values())
print("\n=== Combined Total Across All CSV Files ===")
print(f"Label 0 (Legitimate): {total_label_counts[0]}")
print(f"Label 1 (Phishing):   {total_label_counts[1]}")

print("\n% Breakdown:")
print(f"Label 0: {total_label_counts[0] / total * 100:.2f}%")
print(f"Label 1: {total_label_counts[1] / total * 100:.2f}%")
