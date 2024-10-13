import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras import Sequential
from keras import layers
from keras import optimizers
from keras import callbacks
from keras import regularizers
# Load the dataset
file_path = './train_dataset.csv'  # Adjust path as per your setup
df = pd.read_csv(file_path)

# Preprocess the dataset
# Label encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 
                       'Spending_Score', 'Specific_food_categories', 'preferred_activities', 
                       'Travel_Preferences', 'Preferred_cuisines', 'Climate_preference']

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Target label encoding
le_segmentation = LabelEncoder()
df['Segmentation'] = le_segmentation.fit_transform(df['Segmentation'])

# Split the dataset into features and target
X = df.drop(columns=['Segmentation'])
y = df['Segmentation']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_columns = ['Age', 'Work_Experience', 'Family_Size']
X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

# Define the deep learning model
model = Sequential()
model.add(layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Input layer
model.add(layers.Dense(32, activation='relu'))  # Hidden layer
model.add(layers.Dense(16, activation='relu'))  # Hidden layer
model.add(layers.Dense(len(y.unique()), activation='softmax'))  # Output layer for segmentation

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc}")




# Define the improved deep learning model
model = Sequential()

# Input Layer
model.add(layers.Dense(128, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=regularizers.l2(0.01)))

# Hidden Layers with Dropout
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.3))  # Drop 30% of the neurons randomly to prevent overfitting

model.add(layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.Dropout(0.3))

# Output Layer with softmax activation for multiclass classification
model.add(layers.Dense(len(y.unique()), activation='softmax'))

# Compile the model with a lower learning rate for more controlled learning
optimizer = optimizers.Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Implement Early Stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy Imp: {test_acc}")
