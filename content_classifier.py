import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import os
import glob

import numpy as np
data_set = pd.DataFrame()
all_df = []
folder_path = "./embeddings_2/"
for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
    df_temp = pd.read_csv(csv_file)
    df_temp["X"] = df_temp.X.apply(eval).apply(np.array)
    all_df.append(df_temp)
data_set = pd.concat(all_df, ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(data_set["X"], data_set["y"], test_size=0.2, random_state=42)

X_train_array = X_train.to_numpy()
X_train_3d = np.stack([x.reshape(-1, 1) for x in X_train_array])
X_test_array = X_test.to_numpy()
X_test_3d = np.stack([x.reshape(-1, 1) for x in X_test_array])

model = Sequential()
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_3d, y_train, epochs=50, batch_size=64,shuffle=True, validation_data=(X_test_3d, y_test))
# Access the history data
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']

# If validation data was used, you can also access validation loss and accuracy
validation_loss = history.history.get('val_loss')
validation_accuracy = history.history.get('val_accuracy')

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('./results/training_history_fullsets_1.csv', index=False)