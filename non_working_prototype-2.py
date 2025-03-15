import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data=r'C:\Users\96176\Desktop\university\MachineLearning\PROJ-2\dataset\dataset_traffic_accident_prediction1.csv'
df = pd.read_csv(data)

df.dropna()
# Feature Scaling
scaler = StandardScaler()
numerical_features = ['Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Using label encoding for ordinal categories
columns_to_encode1 = ['Accident_Severity', 'Traffic_Density', 'Driver_Alcohol', 'Accident']
for column in columns_to_encode1:
    label_encoder = LabelEncoder()  # Create a new instance for each column
    df['Encoded_' + column] = label_encoder.fit_transform(df[column])

# Using get_dummies for nominal categories
columns_to_encode2 = ['Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition']
df = pd.get_dummies(df, columns=columns_to_encode2, drop_first=True)

print(df)

corr=df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()