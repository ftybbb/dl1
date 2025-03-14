import pandas as pd

# Replace with your actual file paths
df1 = pd.read_csv('outputs/test_predictions1.csv')
df2 = pd.read_csv('outputs/warmup-only/test_predictions.csv')

# Merge the two dataframes on the 'ID' column
merged_df = pd.merge(df1, df2, on='ID', how='outer')

# Rows in df1 not in df2
diff1 = pd.concat([df1, df2]).drop_duplicates(keep=False)

# Rows in df2 not in df1
diff2 = pd.concat([df2, df1]).drop_duplicates(keep=False)


print("Rows in file1.csv but not in file2.csv:")
print(diff1)

print("\nRows in file2.csv but not in file1.csv:")
print(diff2)
