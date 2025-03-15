import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

data=r'C:\Users\96176\Desktop\university\MachineLearning\PROJ-2\dataset\dataset_traffic_accident_prediction1.csv'
df = pd.read_csv(data)

#Road_Type	Time_of_Day	Traffic_Density	Speed_Limit	Number_of_Vehicles
#Driver_Alcohol	Accident_Severity	Road_Condition	Vehicle_Type
#Driver_Age	Driver_Experience	Road_Light_Condition	Accident

# Fill null values based on conditions for each column
df['Weather'] = df['Weather'].astype('string')
df['Time_of_Day'] = df['Time_of_Day'].astype('string')
df['Road_Type'] = df['Road_Type'].astype('string')
df['Accident_Severity'] = df['Accident_Severity'].astype('string')
df['Road_Condition'] = df['Road_Condition'].astype('string')
df['Vehicle_Type'] = df['Vehicle_Type'].astype('string')
df['Road_Light_Condition'] = df['Road_Light_Condition'].astype('string')

df['Traffic_Density'] = df['Traffic_Density'].astype('float64')
df['Speed_Limit'] = df['Speed_Limit'].astype('float64')
df['Number_of_Vehicles'] = df['Number_of_Vehicles'].astype('float64')
df['Driver_Age'] = df['Driver_Age'].astype('float64')
df['Driver_Alcohol'] = df['Driver_Alcohol'].astype('float64')
df['Driver_Experience'] = df['Driver_Experience'].astype('float64')
df['Accident'] = df['Accident'].astype('float64')
#print(df.dtypes)

# Weather: If null, fill with 'Unknown'
df.loc[df['Weather'].isnull() & (df['Road_Condition'].isin(['Wet', 'Under Construction']))].replace('Rainy')
df.loc[df['Weather'].isnull() & (df['Road_Condition']=='Dry')] = 'Clear'
df.loc[df['Weather'].isnull() & (df['Road_Condition']=='Icy')] = 'Stormy'

#Time of the day:
df.loc[df['Time_of_Day'].isnull() & (df['Road_Light_Condition'] == 'Daylight') & (df['Traffic_Density'] == 0)] = 'Morning'
df.loc[df['Time_of_Day'].isnull() & (df['Road_Light_Condition'] == 'Daylight') & (df['Traffic_Density'] > 0)] = 'Afternoon'
df.loc[df['Time_of_Day'].isnull() & (df['Road_Light_Condition'].isin(['No Light', 'Artificial Light']))] = 'Night'

# Road_Type: If null, fill with 'Unknown'
df.loc[df['Road_Type'].isnull() & (df['Number_of_Vehicles']>3)] = 'Highway'
df.loc[df['Road_Type'].isnull() & (df['Number_of_Vehicles']==3)] = 'City Road'
df.loc[df['Road_Type'].isnull() & (df['Number_of_Vehicles']<3)] = 'Rural Road'

# Traffic_Density: If null and Speed_Limit > 80, fill with 0
df.loc[(df['Traffic_Density'].isnull()), 'Traffic_Density'] = 1

# Speed_Limit: If null, fill with the mean of the column
df.loc[df['Speed_Limit'].isnull(), 'Speed_Limit'] = df['Speed_Limit'].mean()

# Number_of_Vehicles: If null, fill with the median of the column
df.loc[df['Number_of_Vehicles'].isnull(), 'Number_of_Vehicles'] = df['Number_of_Vehicles'].median()

# Driver_Alcohol: If null, fill with 0 (indicating no alcohol)
df.loc[df['Driver_Alcohol'].isnull(), 'Driver_Alcohol'] = 1
 # 1 To be on the safer side where assuming that a driver is drunk will make the decision more cautious

# Accident_Severity: If null, fill with 'Unknown'
df.loc[df['Accident_Severity'].isnull(), 'Accident_Severity'] = 'Moderate'

# Accident: If null, fill with accident
df.loc[df['Accident'].isnull(), 'Accident'] = 1

# Road_Condition: If null, fill with 'Dry'
df.loc[df['Road_Condition'].isnull() & (df['Weather'].isin(['Rainy', 'Foggy']))] = 'Wet'
df.loc[df['Road_Condition'].isnull() & (df['Weather'].isin(['Snowy', 'Stormy']))] = 'Icy'
df.loc[df['Road_Condition'].isnull() & (df['Weather']=='Clear')] = 'Dry'

# Vehicle_Type: If null, fill with 'Unknown'
df.loc[df['Vehicle_Type'].isnull(), 'Vehicle_Type'] = 'Car'

# Driver_Age: If null, fill with the mean of the column
df.loc[df['Driver_Age'].isnull(), 'Driver_Age'] = df['Driver_Age'].mean()

# Driver_Experience: If null, fill with 0 (indicating no experience)
df.loc[df['Driver_Experience'].isnull(), 'Driver_Experience'] = df['Driver_Experience'].mean()

# Road_Light_Condition: If null, fill with 'Daylight'
df.loc[df['Road_Light_Condition'].isnull() & df['Time_of_Day'].isin(['Morning', 'Afternoon'])] = 'Daylight'
df.loc[df['Road_Light_Condition'].isnull() & df['Time_of_Day'].isin(['Evening', 'Night'])] = 'No Light'
# No light for extra safety as predicting more dangerous scenarios would make drivers more cautious

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


"""
# Define features and target variable
X = df.drop(columns=['Accident_Severity'])  # Features
y = df['Accident_Severity']  # Target variable (for multi-class classification)

# Check for missing values
print("Missing values in each column:\n", df.isnull().sum())

# One-Hot Encoding for categorical features
X_encoded = pd.get_dummies(X, columns=['Weather', 'Road_Type', 'Time_of_Day', 
                                        'Road_Condition', 'Vehicle_Type', 
                                        'Road_Light_Condition'], drop_first=True)

"""
#print(df)
"""
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, df['Accident_Severity'], test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning (optional)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters from Grid Search:", grid_search.best_params_)
"""