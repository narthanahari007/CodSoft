import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the file path
file_path = (r"D:\NarthanaHari Project\Iris Flower Classification\IRIS.csv")

# Check if the file exists
if os.path.exists(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Check for missing values
    print("\nMissing values in each column:\n", data.isnull().sum())
    
    # Display basic descriptive statistics
    print("\nDescriptive Statistics:\n", data.describe())
    
    # Plot histograms for each feature
    data.hist(bins=20, figsize=(12, 8))
    plt.suptitle('Histograms of Features')
    plt.show()

    # Optionally, you can also use Seaborn to create more advanced plots
    # Pairplot
    sns.pairplot(data, hue='species')  # Assuming 'species' is the column for class labels
    plt.suptitle('Pairplot of Features by Species')
    plt.show()
else:
    print(f"File not found: {file_path}")
