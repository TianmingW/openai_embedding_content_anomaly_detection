import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np

# Prepare the datasets
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

# Set stop when converged
early_stopping = EarlyStopping(
    monitor='val_loss',     # The metric to monitor (e.g., validation loss)
    min_delta=0.001,        # Minimum change to qualify as an improvement
    patience=10,            # How many epochs to wait for improvement
    verbose=1,              # Verbose output
    mode='min',             # The mode ('min' for minimizing, 'max' for maximizing)
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

# Train the model
model = Sequential()
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train_3d, 
    y_train, 
    epochs=50, 
    batch_size=64,
    shuffle=True, 
    validation_data=(X_test_3d, y_test),
    callbacks=[early_stopping]
    )

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv('./results/history_fullsets_2.csv', index=False)