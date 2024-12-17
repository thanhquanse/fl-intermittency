import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import json

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping

class FLDigitalTwin:
    def __init__(self, config) -> None:
        self.config = config
    
    # Function to load all sheets from an Excel file into a dictionary with error handling
    def load_data(self, file_path, sheets):
        data_dict = {}
        for key, sheet in sheets.items():
            try:
                data_dict[key] = pd.read_excel(file_path, sheet_name=sheet, skiprows=0, nrows=5000)
                # data_dict[key] = pd.read_excel(file_path, sheet_name=sheet)
                print(f"Loaded data for {key} from sheet {sheet}")
            except Exception as e:
                print(f"Failed to load data for {key} from sheet {sheet}: {e}")
        return data_dict

    def load_val_data(self, file_path, sheets):
        data_dict = {}
        for key, sheet in sheets.items():
            try:
                data_dict[key] = pd.read_excel(file_path, sheet_name=sheet, skiprows=5000, nrows=5000, names=['date', 'value'])
                print(f"Loaded data for {key} from sheet {sheet}")
            except Exception as e:
                print(f"Failed to load data for {key} from sheet {sheet}: {e}")
        return data_dict

    # Function to visualize data
    def visualize_data(self, df, title, xlabel, ylabel):
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
        sns.set(rc={"figure.figsize": (15, 6)})
        sns.histplot(df["value"], bins=50, kde=True, color="blue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    # Function to create and compile the model
    def create_model(self, input_shape, output_shape):
        model = Sequential()
        model.add(LSTM(64, activation="tanh", input_shape=input_shape))
        # model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        # model.add(Dropout(0.2))
        model.add(Dense(output_shape))
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipvalue=1.0), loss="mean_squared_error", metrics=["mse", "mae", "mape"]
        )
        return model

    # Function to plot data
    def plot_the_data(self, orig_plot, predict_train_plot, predict_test_plot, title):
        plt.figure(figsize=(15, 6))
        plt.plot(orig_plot, color="blue", label="Actual")
        plt.plot(predict_train_plot, color="red", label="Predicted on training")
        plt.plot(predict_test_plot, color="green", label="Predicted on testing")
        plt.legend()
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value Summation")
        plt.show()

    # Data preparation functions
    def scale_split_datasets(self, data, train_size, lookback):
        sc_X = StandardScaler()
        daily_consumption_scaled = sc_X.fit_transform(data.values.reshape(-1, 1))
        num_train = int(train_size * len(data))
        training_data = daily_consumption_scaled[:num_train]
        test_data = daily_consumption_scaled[num_train - lookback :]
        return training_data, test_data, sc_X

    def create_rnn_dataset(self, data, lookback):
        data_x, data_y = [], []
        for i in range(len(data) - lookback - 1):
            a = data[i : (i + lookback), 0]
            data_x.append(a)
            data_y.append(data[i + lookback, 0])
        x = np.array(data_x)
        y = np.array(data_y)
        x = np.reshape(x, (x.shape[0], 1, x.shape[1]))
        return x, y

    def plot_data_preparation(self, data, predict_on_train, predict_on_test, lookback):
        total_size = len(predict_on_train) + len(predict_on_test)
        orig_data = data.to_numpy().reshape(len(data), 1)
        orig_plot = np.empty((total_size, 1))
        orig_plot[:, :] = np.nan
        orig_plot[:total_size, :] = orig_data[lookback:-2]
        predict_train_plot = np.empty((total_size, 1))
        predict_train_plot[:, :] = np.nan
        predict_train_plot[: len(predict_on_train), :] = predict_on_train
        predict_test_plot = np.empty((total_size, 1))
        predict_test_plot[:, :] = np.nan
        predict_test_plot[len(predict_on_train) : total_size, :] = predict_on_test
        return orig_plot, predict_train_plot, predict_test_plot

    # Federated Learning Model Preparation
    def create_train_test_dataset(self, df, lookback):
        # df["value"] = df.iloc[:, 1:-1].sum(axis=1)
        sc_X = StandardScaler()
        daily_consumption = df["value"]
        num_train = int(self.config['TRAIN_SIZE'] * len(daily_consumption))
        daily_consumption_scaled = sc_X.fit_transform(
            daily_consumption.values.reshape(-1, 1)
        )
        training_set = daily_consumption_scaled[:num_train]
        x_train, y_train = self.create_rnn_dataset(training_set, lookback)
        test_data = daily_consumption_scaled[num_train - lookback :]
        x_test, y_test = self.create_rnn_dataset(test_data, lookback)
        return x_train, y_train, x_test, y_test, sc_X

    def train_model(self, model, x_train, y_train, log_dir):
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        early_stopping = EarlyStopping(
            monitor="loss", min_delta=0.001, patience=5, verbose=1, mode="auto"
        )
        model.fit(
            x_train,
            y_train,
            epochs=self.config['EPOCHS'],
            batch_size=self.config['BATCH_SIZE'],
            verbose=1,
            callbacks=[tensorboard_callback, early_stopping],
            # callbacks=[tensorboard_callback],
        )

    # Evaluate the models
    def evaluate_model(self, model, x_train, y_train, x_test, y_test):
        model.evaluate(x_train, y_train)
        model.evaluate(x_test, y_test)

    # Federated learning aggregation sections
    def train_fl_full_updates(self, models, x_train, y_train, x_test, y_test, rounds=1):
        history_dict = {}
        for r in range(rounds):
            print(f"Round {r}:")
            history_dict[str(r)] = {}
            weights = [model.get_weights() for model in models]
            new_weights = [
                np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))
            ]
            for i in range(len(models)):
                models[i].set_weights(new_weights)
                history = models[i].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.config['CLIENT_EPOCHS'], batch_size=self.config['BATCH_SIZE'], verbose=1)
                history_dict[str(r)][str(i)] = history.history

        return history_dict
    
    def _train_fl_full_updates(self, models, x_train, y_train, x_test, y_test, model_id, global_weights):
        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0.001, patience=3, verbose=1, mode="auto", restore_best_weights=True
        )

        print(f"Training model {model_id}:")
        
        print(f"Updating global weights for model {model_id}...")
        if len(global_weights) > 0:
            models[model_id].set_weights(global_weights)
        
        history = models[model_id].fit(
            x_train, y_train,
            validation_data=(x_test, y_test),
            epochs=self.config['CLIENT_EPOCHS'],
            batch_size=self.config['BATCH_SIZE'],
            verbose=1,
            callbacks=[early_stopping]
        )

        # Loss
        # plt.figure(figsize=(10, 5))
        # plt.plot(history.history['loss'], label='Train')
        # plt.plot(history.history['val_loss'], label='Test')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.title(f'Model {model_id} Loss')
        # plt.show()

        return history.history

    def save_weights_to_csv(self, model, filename):
        """Save the model weights to a CSV file."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            for layer in model.weights:
                writer.writerow(layer.numpy().flatten())

    def load_weights_from_csv(self, filename, model):
        """Load weights from a CSV file and set them to the model."""
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            new_weights = []
            for layer, row in zip(model.weights, reader):
                reshaped_weights = np.array(row, dtype=np.float32).reshape(layer.shape)
                new_weights.append(reshaped_weights)
            model.set_weights(new_weights)

    def sum_weights_from_two_csvs(self, file1, file2, model):
        """Sum weights from two CSV files and set them to the model."""
        weights1 = []
        weights2 = []

        # Read weights from the first file
        with open(file1, 'r') as f1:
            reader1 = csv.reader(f1)
            weights1 = [np.array(row, dtype=np.float32) for row in reader1]

        # Read weights from the second file
        with open(file2, 'r') as f2:
            reader2 = csv.reader(f2)
            weights2 = [np.array(row, dtype=np.float32) for row in reader2]

        # Compute the sum of weights
        summed_weights = [0.2*w1 + 0.8*w2 for w1, w2 in zip(weights1, weights2)]

        # Reshape summed weights and set them to the model
        reshaped_weights = [
            weight.reshape(layer.shape) for weight, layer in zip(summed_weights, model.weights)
        ]
        # model.set_weights(reshaped_weights)
        return reshaped_weights

    def train_fl_digital_twin(self, models, x_train, y_train, x_test, y_test, client_matrix, round=1, has_weights_mechanism=False):
        history_dict = {}    
        for r in range(round):
            print(f"Round {r}:")
            history_dict[str(r)] = {}
            weights = [] #[model.get_weights() for model in models]
            for i in range(len(models)):
                if r >= 2 and client_matrix[r, i] == 'N':
                    if has_weights_mechanism:
                        csv_filename1 = f'model_client_{str(i)}_round_{str(r - 2)}.csv'
                        csv_filename2 = f'model_client_{str(i)}_round_{str(r - 1)}.csv'

                        prev_csv_1 = os.path.join(self.config['WEIGHT_TRACKING_DIR'], csv_filename1)
                        prev_csv_2 = os.path.join(self.config['WEIGHT_TRACKING_DIR'], csv_filename2)
                        
                        self.sum_weights_from_two_csvs(prev_csv_1, prev_csv_2, models[i])
                        print(f"Updated weights using the sum of round {r - 1} and {r - 2}")
                    else:
                        zero_w = [0.0*np.random.rand(*w.shape) for w in models[i].get_weights()]
                        models[i].set_weights(zero_w)
                
                weights.append(models[i].get_weights())
            
            new_weights = [
                np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))
            ]
            for i in range(len(models)):
                models[i].set_weights(new_weights)
                history = models[i].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=self.config['CLIENT_EPOCHS'], batch_size=self.config['BATCH_SIZE'], verbose=1)
                history_dict[str(r)][str(i)] = history.history
                
                # Save weights
                if has_weights_mechanism:
                    csv_filename = f'model_client_{str(i)}_round_{str(r)}.csv'
                    csv_file = os.path.join(self.config['WEIGHT_TRACKING_DIR'], csv_filename)
                    self.save_weights_to_csv(models[i], csv_file)
                    print(f"Saved weights to {csv_file}")
            
        return history_dict

    # Save results to json file
    def to_json(self, filepath, history_dict):
        with open(filepath, "w") as f:
            json.dump(history_dict, f)

    # Predictions and plotting
    def inverse_transform_predictions(self, predictions, scaler):
        return scaler.inverse_transform(predictions)

    # Plot predictions
    def prepare_and_plot(self, data, train_predictions, test_predictions, lookback, title):
        orig_plot, train_plot, test_plot = self.plot_data_preparation(
            data, train_predictions, test_predictions, lookback
        )
        self.plot_the_data(orig_plot, train_plot, test_plot, title)