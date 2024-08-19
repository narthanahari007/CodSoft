import pandas as pd

# Load the dataset
file_path = r"D:\NarthanaHari Project\Titanic Survival Prediction\Titanic-Dataset.csv"
data = pd.read_csv(file_path)

# Print column names to verify if 'Name' exists
print("Column names:", data.columns)

# Check if 'Name' column exists before proceeding
if 'Name' in data.columns:
    # Extract titles from the Name column
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+).', expand=False)

    # Standardize and replace rare titles
    title_replacements = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare', 'Sir': 'Rare',
        'Jonkheer': 'Rare', 'Dona': 'Rare', 'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    data['Title'] = data['Title'].replace(title_replacements)

    # Encode 'Sex' as 0 for male and 1 for female
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # Encode 'Embarked' as numeric values (and handle missing values)
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    # Encode titles using one-hot encoding
    data = pd.get_dummies(data, columns=['Title'])

    # Drop columns that are not needed for the model
    data = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    # Impute missing values in the Age and Fare columns with the median value
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Save the preprocessed data to a new CSV file
    output_path = r"D:\NarthanaHari Project\Titanic Survival Prediction\titanic_modified_encoded_cleaned.csv"
    data.to_csv(output_path, index=False)

    # Print success message
    print(f"The file has been saved successfully to {output_path}")

else:
    print("'Name' column is not found in the dataset.")
