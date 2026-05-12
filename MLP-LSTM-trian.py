import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from scipy import stats
import scipy.integrate as integrate
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, BatchNormalization, Flatten, Reshape, Dropout, Multiply
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
import joblib
import os
import traceback

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected, using CPU")


class Config:
    SEED = 42
    DATA_PATH = "3886simple.xlsx"
    MODEL_SAVE_DIR = "saved_models"
    RESULT_SAVE_DIR = "results"

    TARGET_MODEL = {
        'epochs': 500,
        'batch_size': 32,
        'patience': 30,
        'k_folds': 5
    }

    CURVE_MODEL = {
        'epochs': 300,
        'batch_size': 32,
        'patience': 30,
        'k_folds': 5
    }


def init_environment():
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.RESULT_SAVE_DIR, exist_ok=True)
    np.random.seed(Config.SEED)
    tf.random.set_seed(Config.SEED)


def load_and_preprocess_data():
    try:
        df = pd.read_excel(Config.DATA_PATH, header=None, engine='openpyxl')
        X = df.iloc[:, :24].values.astype(np.float32)
        y_curve = df.iloc[:, 24:].values.astype(np.float32)

        X_scaled = X.copy()
        print(f"\nInput feature preprocessing complete - samples: {X.shape[0]}, features: {X.shape[1]}")

        y_curve_with_zero = np.hstack([np.zeros((y_curve.shape[0], 1)), y_curve])
        x_points = np.linspace(0, 28, 201)
        energy = np.array([integrate.simpson(np.abs(curve), x_points) for curve in y_curve_with_zero])
        max_force = np.max(y_curve_with_zero[:, :30], axis=1)
        y_targets = np.column_stack([energy, max_force])

        print(f"Target value range - Energy: [{energy.min():.2f}, {energy.max():.2f}]")
        print(f"               Max Force: [{max_force.min():.2f}, {max_force.max():.2f}]")
        print(f"Force curve range: [{y_curve.min():.2f}, {y_curve.max():.2f}]")

        return X_scaled, y_targets, y_curve

    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")


class TargetModel:
    def __init__(self):
        self.scaler = None
        self.input_dim = None
        self.model = None

    def preprocess_targets(self, y):
        self.scaler = RobustScaler()
        y_normalized = self.scaler.fit_transform(y)
        return y_normalized

    def inverse_transform(self, y_normalized):
        return self.scaler.inverse_transform(y_normalized)

    def build_model(self):
        if self.input_dim is None:
            raise ValueError("input_dim not initialized. Call train method first.")

        inputs = Input(shape=(self.input_dim,))
        x = Dense(128, activation='gelu')(inputs)
        x = BatchNormalization()(x)
        x = Dense(64, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Dense(32, activation='gelu')(x)
        x = BatchNormalization()(x)
        outputs = Dense(2)(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.0005),
                      loss='mse',
                      metrics=['mae'])
        return model

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call train method first.")
        return self.model.predict(X, verbose=0)

    def evaluate(self, model, X_val, y_val, val_indices, X_original):
        y_pred = model.predict(X_val, verbose=0)
        y_true = self.inverse_transform(y_val)
        y_pred = self.inverse_transform(y_pred)

        energy_mae = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
        force_mae = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
        energy_r2 = r2_score(y_true[:, 0], y_pred[:, 0])
        force_r2 = r2_score(y_true[:, 1], y_pred[:, 1])

        results = []
        for i in range(len(y_val)):
            results.append({
                'input': X_original[val_indices[i]][:24].tolist(),
                'true_energy': float(y_true[i, 0]),
                'pred_energy': float(y_pred[i, 0]),
                'true_max_force': float(y_true[i, 1]),
                'pred_max_force': float(y_pred[i, 1]),
                'energy_error': float(np.abs(y_true[i, 0] - y_pred[i, 0])),
                'force_error': float(np.abs(y_true[i, 1] - y_pred[i, 1]))
            })

        return (energy_mae, force_mae, energy_r2, force_r2), results

    def train(self, X, y, X_original):
        print("\n" + "=" * 50)
        print("Starting target model training (energy and max force)")
        print("=" * 50)

        self.input_dim = X.shape[1]

        y_normalized = self.preprocess_targets(y)

        kf = KFold(n_splits=Config.TARGET_MODEL['k_folds'], shuffle=True, random_state=Config.SEED)
        all_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n=== Fold {fold}/{Config.TARGET_MODEL['k_folds']} ===")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_normalized[train_idx], y_normalized[val_idx]

            model = self.build_model()
            callbacks = [
                EarlyStopping(patience=Config.TARGET_MODEL['patience'], restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=Config.TARGET_MODEL['patience'] // 2),
                ModelCheckpoint(
                    f"{Config.MODEL_SAVE_DIR}/target_fold_{fold}_best.h5",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=Config.TARGET_MODEL['epochs'],
                batch_size=Config.TARGET_MODEL['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            np.save(f"{Config.MODEL_SAVE_DIR}/target_fold_{fold}_history.npy", history.history)

            self.model = load_model(f"{Config.MODEL_SAVE_DIR}/target_fold_{fold}_best.h5")

            metrics, results = self.evaluate(self.model, X_val, y_val, val_idx, X_original)
            all_results.append((*metrics, results))
            print(f"Fold {fold} results - Energy MAE: {metrics[0]:.2f}, R2: {metrics[2]:.4f}")
            print(f"              Max Force MAE: {metrics[1]:.2f}, R2: {metrics[3]:.4f}")

            pd.DataFrame(results).to_csv(
                f"{Config.RESULT_SAVE_DIR}/target_fold_{fold}_results.csv",
                index=False
            )

        final_metrics = np.mean([r[:4] for r in all_results], axis=0)
        print("\n" + "=" * 50)
        print(f"Final results - Energy MAE: {final_metrics[0]:.2f}, R2: {final_metrics[2]:.4f}")
        print(f"            Max Force MAE: {final_metrics[1]:.2f}, R2: {final_metrics[3]:.4f}")
        print("=" * 50)

        all_results_flattened = [res for *_, fold_results in all_results for res in fold_results]
        pd.DataFrame(all_results_flattened).to_csv(
            f"{Config.RESULT_SAVE_DIR}/target_final_results.csv",
            index=False
        )

        joblib.dump(self.scaler, f"{Config.MODEL_SAVE_DIR}/target_scaler.pkl")


class CurveModel:
    def __init__(self):
        self.scaler = None
        self.normalization_type = None
        self.min_val = None
        self.target_model = None

    def set_target_model(self, target_model):
        self.target_model = target_model

    def preprocess_curves(self, y):
        data_range = np.max(y) - np.min(y)

        if data_range > 1e6:
            print("Using Log+Robust Scaling")
            self.min_val = np.min(y)
            y_log = np.log1p(y - self.min_val + 1e-6)
            self.scaler = RobustScaler()
            y_normalized = self.scaler.fit_transform(y_log)
            self.normalization_type = 'log_robust'
        else:
            print("Using Robust Scaling")
            iqr = stats.iqr(y, axis=0)
            if np.any(iqr == 0):
                print("Adding small noise (1e-6) to columns with zero IQR")
                y = y + np.random.normal(0, 1e-6, y.shape)
            self.scaler = RobustScaler()
            y_normalized = self.scaler.fit_transform(y)
            self.normalization_type = 'robust'

        return y_normalized

    def inverse_transform(self, y_normalized):
        if self.normalization_type == 'robust':
            return self.scaler.inverse_transform(y_normalized)
        elif self.normalization_type == 'log_robust':
            y_scaled = self.scaler.inverse_transform(y_normalized)
            return np.exp(y_scaled) + self.min_val - 1e-6
        else:
            raise ValueError("Unknown normalization type")

    def attention_block(self, inputs):
        a = Dense(inputs.shape[-1], activation='softmax')(inputs)
        return Multiply()([inputs, a])

    def build_model(self, input_dim):
        original_input = Input(shape=(input_dim,), name='original_input')
        target_input = Input(shape=(2,), name='target_input')

        x_original = Dense(64, activation='gelu')(original_input)
        x_original = BatchNormalization()(x_original)
        x_original = Dropout(0.3)(x_original)

        x_target = Dense(16, activation='gelu')(target_input)
        x_target = BatchNormalization()(x_target)
        x_target = Dropout(0.3)(x_target)

        merged = tf.keras.layers.concatenate([x_original, x_target])

        x = Dense(128, activation='gelu')(merged)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        curve = Dense(200, activation='gelu')(x)
        curve = Reshape((200, 1))(curve)
        lstm = LSTM(128, return_sequences=True)(curve)

        attention = self.attention_block(lstm)
        flattened = Flatten()(attention)

        outputs = Dense(200)(flattened)

        model = Model(inputs=[original_input, target_input], outputs=outputs)
        model.compile(optimizer=Adam(0.001),
                      loss='huber',
                      metrics=['mae'])
        return model

    def evaluate(self, model, X_val, y_val, val_indices, X_original):
        target_pred = self.target_model.predict(X_val)

        y_pred = model.predict([X_val, target_pred], verbose=0)
        y_true = self.inverse_transform(y_val)
        y_pred = self.inverse_transform(y_pred)

        mae_values = [np.mean(np.abs(y_true[i] - y_pred[i])) for i in range(len(y_val))]
        avg_mae = np.mean(mae_values)
        r2 = r2_score(y_true.flatten(), y_pred.flatten())

        results = []
        for i in range(len(y_val)):
            results.append({
                'input': X_original[val_indices[i]][:24].tolist(),
                'true_curve': y_true[i].tolist(),
                'pred_curve': y_pred[i].tolist(),
                'mae': float(mae_values[i]),
                'target_pred': target_pred[i].tolist()
            })

        return avg_mae, r2, results

    def train(self, X, y, X_original):
        print("\n" + "=" * 50)
        print("Starting force curve model training (using target predictions)")
        print("=" * 50)

        y_normalized = self.preprocess_curves(y)

        kf = KFold(n_splits=Config.CURVE_MODEL['k_folds'], shuffle=True, random_state=Config.SEED)
        all_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"\n=== Fold {fold}/{Config.CURVE_MODEL['k_folds']} ===")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_normalized[train_idx], y_normalized[val_idx]

            target_train = self.target_model.predict(X_train)
            target_val = self.target_model.predict(X_val)

            model = self.build_model(X.shape[1])
            callbacks = [
                EarlyStopping(patience=Config.CURVE_MODEL['patience'], restore_best_weights=True),
                ReduceLROnPlateau(factor=0.2, patience=Config.CURVE_MODEL['patience'] // 2),
                ModelCheckpoint(
                    f"{Config.MODEL_SAVE_DIR}/curve_fold_{fold}_best.h5",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ]

            history = model.fit(
                [X_train, target_train], y_train,
                validation_data=([X_val, target_val], y_val),
                epochs=Config.CURVE_MODEL['epochs'],
                batch_size=Config.CURVE_MODEL['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            np.save(f"{Config.MODEL_SAVE_DIR}/curve_fold_{fold}_history.npy", history.history)

            model = load_model(
                f"{Config.MODEL_SAVE_DIR}/curve_fold_{fold}_best.h5",
                custom_objects={'attention_block': self.attention_block}
            )
            mae, r2, results = self.evaluate(model, X_val, y_val, val_idx, X_original)
            all_results.append((mae, r2, results))
            print(f"Fold {fold} results - MAE: {mae:.2f}, R2: {r2:.4f}")

            pd.DataFrame(results).to_csv(
                f"{Config.RESULT_SAVE_DIR}/curve_fold_{fold}_results.csv",
                index=False
            )

        final_mae = np.mean([r[0] for r in all_results])
        final_r2 = np.mean([r[1] for r in all_results])
        print("\n" + "=" * 50)
        print(f"Final results - Average MAE: {final_mae:.2f}, Average R2: {final_r2:.4f}")
        print("=" * 50)

        all_results_flattened = [res for _, _, fold_results in all_results for res in fold_results]
        pd.DataFrame(all_results_flattened).to_csv(
            f"{Config.RESULT_SAVE_DIR}/curve_final_results.csv",
            index=False
        )

        joblib.dump({
            'scaler': self.scaler,
            'normalization_type': self.normalization_type,
            'min_val': self.min_val
        }, f"{Config.MODEL_SAVE_DIR}/curve_scaler.pkl")


def main():
    init_environment()

    try:
        X, y_targets, y_curves = load_and_preprocess_data()

        target_model = TargetModel()
        target_model.train(X, y_targets, X)

        curve_model = CurveModel()
        curve_model.set_target_model(target_model)
        curve_model.train(X, y_curves, X)

    except Exception as e:
        print(f"\nProgram error: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nProgram finished")


if __name__ == "__main__":
    main()