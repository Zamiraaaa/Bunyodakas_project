import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle


# Data Collection and Import
data = pd.read_csv(r'C:\Users\User\Desktop\Model_Deploying2\dataset .csv')
print(data)

# Label encoding for categorical features
label_encoder = LabelEncoder()
data['Ship Mode'] = label_encoder.fit_transform(data['Ship Mode'])
data['Category'] = label_encoder.fit_transform(data['Category'])
data['Segment'] = label_encoder.fit_transform(data['Segment'])



X = data.drop(['Row ID','Order ID','Order Date','Ship Date','Customer ID','Customer Name','Country','City','State',
              'Postal Code','Region',
               'Product ID','Sub-Category','Product Name'],axis = 1) # Features
y = data['Profit']   # Target variable


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)








