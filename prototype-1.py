# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

# %%
data=r'original_dataset\dataset_traffic_accident_prediction1.csv'
df = pd.read_csv(data)
df.sample(5)

# %%
df.info()

# %% [markdown]
# # 468

# %% [markdown]
# fill null weather

# %%
def fill_null_weather(row):
    if pd.isnull(row['Weather']):  
        if row['Road_Condition'] == 'Icy':
            return 'Stormy'
        elif row['Road_Condition'] in ['Wet', 'Under Construction']:
            return 'Rainy'
        elif row['Road_Condition'] == 'Dry':
            return 'Clear'
        else:
            return 'Clear'  
    return row['Weather'] 


df['Weather'] = df.apply(fill_null_weather, axis=1)



# %%
df.iloc[[468]]

# %%
df['Road_Condition'].value_counts()

# %% [markdown]
# fill null time of the day

# %%
def fill_null_Time_of_Day(row):
    if pd.isnull(row['Time_of_Day']):  
        if row['Road_Light_Condition'] == 'Daylight':
            return 'Morning'
        elif row['Road_Light_Condition'] in ['No Light', 'Artificial Light']:
            return 'Night'
        else:
            return 'Afternoon' 
    return row['Time_of_Day']  

# Apply row-wise
df['Time_of_Day'] = df.apply(fill_null_Time_of_Day, axis=1)



# %%
df['Road_Light_Condition'].value_counts()

# %% [markdown]
# Fill null road type
# 

# %%
def fill_null_Road_Type(row):
    if pd.isnull(row['Road_Type']):
        if row['Number_of_Vehicles'] > 3:
            return 'Highway'
        elif row['Number_of_Vehicles'] == 3:
            return 'City Road'
        elif row['Number_of_Vehicles'] < 3:
            return 'Rural Road'
        else:
            return 'Highway'  
    return row['Road_Type']  


df['Road_Type'] = df.apply(fill_null_Road_Type, axis=1)

# %%
df["Road_Type"].value_counts()

# %% [markdown]
# fill null traffic density

# %%
def fill_null_Traffic_Density(row):
    if pd.isnull(row['Traffic_Density']):
        return 1
    
    return row['Traffic_Density']  


df['Traffic_Density'] = df.apply(fill_null_Traffic_Density, axis=1)

# %% [markdown]
# fill null speed limit

# %%
def fill_null_Speed_Limit(row):
    if pd.isnull(row['Speed_Limit']):  
        return df['Speed_Limit'].mean()
    
    return row['Speed_Limit'] 


df['Speed_Limit'] = df.apply(fill_null_Speed_Limit, axis=1)



# %% [markdown]
# fill null number of vehicles

# %%
def fill_null_Number_of_vehicles(row):
    if pd.isnull(row['Number_of_Vehicles']):
        return df['Number_of_Vehicles'].mean()

    return row['Number_of_Vehicles']

df['Number_of_Vehicles'] = df.apply(fill_null_Number_of_vehicles, axis=1)

# %% [markdown]
# fill null driver alcohol

# %%
def fill_null_Driver_drunk(row):
    if pd.isnull(row['Driver_Alcohol']):
        return 1

    return row['Driver_Alcohol']

df['Driver_Alcohol'] = df.apply(fill_null_Driver_drunk, axis=1)

# %% [markdown]
# fill null accident severity

# %%
def fill_null_Accident_Severity(row):
    if pd.isnull(row['Accident_Severity']):
        return 'Moderate'

    return row['Accident_Severity']

df['Accident_Severity'] = df.apply(fill_null_Accident_Severity, axis=1)

# %% [markdown]
# fill null accident

# %%
def fill_null_Accident(row):
    if pd.isnull(row['Accident']):
        return 1

    return row['Accident']

df['Accident'] = df.apply(fill_null_Accident, axis=1)

# %% [markdown]
# fill null road condition

# %%
def fill_null_Road_condition(row):
    if pd.isnull(row['Road_Condition']):  # Check for null Weather
        if row['Weather'] == 'Rainy' or 'Foggy':
            return 'Wet'
        elif row['Weather'] == 'Snowy' or 'Stormy':
            return 'Icy'
        elif row['Weather'] == 'Clear':
            return 'Dry'
        else: 
            return 'Dry'
    return row['Road_Condition']  # Keep the existing value if not null

# Apply row-wise
df['Road_Condition'] = df.apply(fill_null_Road_condition, axis=1)

# %% [markdown]
# fill null vehicle type

# %%
def fill_null_Vehicle_type(row):
    if pd.isnull(row['Vehicle_Type']):
        return 'Car'
    
    return row['Vehicle_Type']

df['Vehicle_Type'] = df.apply(fill_null_Vehicle_type, axis=1)

# %% [markdown]
# fill null driver age

# %%
def fill_null_Driver_age(row):
    if pd.isnull(row['Driver_Age']):
        return df['Driver_Age'].mean()
    
    return row['Driver_Age']

df['Driver_Age'] = df.apply(fill_null_Driver_age, axis=1)

# %% [markdown]
# fill null driver exp

# %%
def fill_null_Driver_exp(row):
    if pd.isnull(row['Driver_Experience']):
        return df['Driver_Experience'].mean()
    
    return row['Driver_Experience']

df['Driver_Experience'] = df.apply(fill_null_Driver_exp, axis=1)

# %% [markdown]
# fill null road light condition

# %%
def fill_null_Road_light_cond(row):
    if pd.isnull(row['Road_Light_Condition']):  # Check for null Weather
        if row['Time_of_Day'] == 'Morning' or 'Afternoon':
            return 'Daylight'
        elif row['Time_of_Day'] in ['Night', 'Evening']:
            return 'No Light'
        else:
            return 'Artificial Light'

    return row['Road_Light_Condition']  # Keep the existing value if not null

# Apply row-wise
df['Road_Light_Condition'] = df.apply(fill_null_Road_light_cond, axis=1)



# %% [markdown]
# Feature scaling

# %%
scaler = StandardScaler()
numerical_features = ['Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience']
df[numerical_features] = scaler.fit_transform(df[numerical_features])


# %% [markdown]
# Using label encoding for ordinal categories

# %%
columns_to_encode1 = ['Accident_Severity', 'Traffic_Density', 'Driver_Alcohol','Weather', 'Road_Type', 'Time_of_Day', 'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition', 'Accident']
"""
le = LabelEncoder()
df[columns_to_encode1] = df[columns_to_encode1].apply(le.fit_transform)
"""
for column in columns_to_encode1:
    le=LabelEncoder()
    df[column] = le.fit_transform(df[column])

# %% [markdown]
# Using get_dummies for nominal categories

# %%
#columns_to_encode2 = []
#df = pd.get_dummies(df, columns=columns_to_encode2, drop_first=True)


# %% [markdown]
# Define features and target variable

# %% [markdown]
X = df.drop(columns=['Accident_Severity'])  # Features
y = df['Accident_Severity']

# %% [markdown]
# Check for missing values

# %%
print("Missing values in each column:\n", df.isnull().sum())

# %%
df.sample(5)

# %% [markdown]
# sample

# %%
df.sample(5)

# %% [markdown]
# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# %% [markdown]
# see sample

# %%
print(df)

# %% [markdown]
# initializing model and training

# %%
model = RandomForestClassifier(random_state=42,n_estimators=270, min_samples_split=2, max_depth=10, criterion='entropy')
# n_estimators is the number of trees in the forest,
# min_samples_split is the minimum number of samples required to split an internal node,
# max_depth is the maximum depth of the tree,
# criterion is the function to measure the quality of a split.
model.fit(X_train, y_train)

# %% [markdown]
# prediction test

# %%
y_pred = model.predict(X_test)

# %%
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# %% [markdown]
# Evaluate the model

# %%
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# %% [markdown]
# # Hyperparameter tuning 
# 

# %%
param_grid = {
    'n_estimators': [100, 200, 270],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'criterion':['entropy','gini']
}

# %%
"""grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters from Grid Search:", grid_search.best_params_)
"""
# %%
print(df.corr())


