import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix
import os
import glob
import numpy as np
import sys

# Prepare the datasets
# data_set = pd.DataFrame()
# all_df = []
# folder_path = "./iot_2023_embeddings/"

# for csv_file in glob.glob(os.path.join(folder_path, '*.csv')):
#     df_temp = pd.read_csv(csv_file)
#     df_temp["X"] = df_temp.X.apply(eval).apply(np.array)
#     all_df.append(df_temp)
# data_set = pd.concat(all_df, ignore_index=True)

file_list = [
    "./iot23/CTU-Honeypot-Capture-5-1/embeddings.h5",
    "./iot23/CTU-IoT-Malware-Capture-34-1/embeddings.h5"
]
data_set = pd.DataFrame()
for h5_file in file_list:
    h5 = pd.HDFStore(h5_file)
    df = h5['/df']
    data_set.append(df, ignore_index = True)

data_benign = data_set[data_set[-1] == 0]
data_attack = data_set[data_set[-1] == 1]

data_benign['X'] = data_benign[:-1].apply(lambda row: row.tolist(), axis=1)
data_attack['X'] = data_attack[:-1].apply(lambda row: row.tolist(), axis=1)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(data_benign["X"], data_benign[-1], test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(data_attack["X"], data_attack[-1], test_size=0.2, random_state=42)

X_train = pd.concat([X_train_b, X_train_a], ignore_index=True)
X_test = pd.concat([X_test_b, X_test_a], ignore_index=True)
y_train = pd.concat([y_train_b, y_train_a], ignore_index=True)
y_test = pd.concat([y_test_b, y_test_a], ignore_index=True)

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

class ConfusionMatrixCallback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        y_pred_classes = (y_pred > 0.5).astype('int32')  # Convert probabilities to binary labels
        cm = confusion_matrix(self.y_val, y_pred_classes)
        TN, FP, FN, TP = cm.ravel()

        # Store the values in the logs dictionary
        logs['val_TN'] = TN
        logs['val_FP'] = FP
        logs['val_FN'] = FN
        logs['val_TP'] = TP
cm_callback = ConfusionMatrixCallback(X_test_3d, y_test)

# Train the model
model = Sequential()
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Reading the iteration number from command line
iteration = sys.argv[1] if len(sys.argv) > 1 else 1

history = model.fit(X_train_3d, 
                    y_train, 
                    epochs=50, 
                    batch_size=64,
                    shuffle=True, 
                    validation_data=(X_test_3d, y_test),
                    callbacks=[early_stopping, cm_callback])

history_df = pd.DataFrame(history.history)
history_df['epoch'] = range(1, len(history_df) + 1)

# Use the iteration number in the filename
output_filename = f'./result_iot23/cm_0705_{iteration}.h5'
history_df.to_hdf(output_filename, key='df', mode='a', complevel=5)