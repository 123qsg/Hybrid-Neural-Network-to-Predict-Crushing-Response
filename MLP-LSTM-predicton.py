import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
import tensorflow as tf
import os
from tqdm import tqdm


class Config:
    SEED = 42
    MODEL_SAVE_DIR = "saved_models"
    RESULT_SAVE_DIR = "results"
    TARGET_MODEL = {'k_folds': 5}
    CURVE_MODEL = {'k_folds': 5}


class Predictor:
    def __init__(self):
        self.target_models = []
        self.curve_models = []
        self.target_scaler = None
        self.curve_scaler_info = None

        self._load_models()
        print("Predictor initialized - loaded "
              f"{len(self.target_models)} target models and "
              f"{len(self.curve_models)} curve models")

    def _load_models(self):
        try:
            for i in range(1, Config.TARGET_MODEL['k_folds'] + 1):
                model_path = f"{Config.MODEL_SAVE_DIR}/target_fold_{i}_best.h5"
                if os.path.exists(model_path):
                    self.target_models.append(load_model(model_path))

            self.target_scaler = joblib.load(
                f"{Config.MODEL_SAVE_DIR}/target_scaler.pkl"
            )

            for i in range(1, Config.CURVE_MODEL['k_folds'] + 1):
                model_path = f"{Config.MODEL_SAVE_DIR}/curve_fold_{i}_best.h5"
                if os.path.exists(model_path):
                    self.curve_models.append(load_model(
                        model_path,
                        custom_objects={'attention_block': self._attention_block}
                    ))

            self.curve_scaler_info = joblib.load(
                f"{Config.MODEL_SAVE_DIR}/curve_scaler.pkl"
            )

            if not self.target_models:
                raise ValueError("No target model files found")
            if not self.curve_models:
                raise ValueError("No curve model files found")

        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    @staticmethod
    def _attention_block(inputs):
        a = tf.keras.layers.Dense(
            inputs.shape[-1],
            activation='softmax',
            kernel_initializer='glorot_uniform'
        )(inputs)
        return tf.keras.layers.Multiply()([inputs, a])

    def preprocess_input(self, X):
        if X.ndim != 2:
            raise ValueError(f"Input must be 2-dimensional, got {X.ndim}")

        expected_features = 24
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Incorrect number of features. Expected {expected_features}, got {X.shape[1]}"
            )

        return X.astype(np.float32)

    def predict_targets(self, X, return_std=False):
        X_processed = self.preprocess_input(X)

        norm_preds = [model.predict(X_processed) for model in self.target_models]
        norm_preds = np.stack(norm_preds)

        avg_norm = np.mean(norm_preds, axis=0)
        std_norm = np.std(norm_preds, axis=0)

        avg_original = self.target_scaler.inverse_transform(avg_norm)
        std_original = self.target_scaler.transform_std(std_norm)

        return (avg_original, std_original) if return_std else avg_original

    def inverse_transform_curve(self, y_normalized):
        scaler = self.curve_scaler_info['scaler']
        norm_type = self.curve_scaler_info['normalization_type']
        min_val = self.curve_scaler_info.get('min_val', 0)

        if norm_type == 'robust':
            return scaler.inverse_transform(y_normalized)
        elif norm_type == 'log_robust':
            y_scaled = scaler.inverse_transform(y_normalized)
            return np.exp(y_scaled) + min_val - 1e-6
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    def predict_curves(self, X, batch_size=32):
        X_processed = self.preprocess_input(X)

        target_preds_norm = [model.predict(X_processed) for model in self.target_models]
        avg_target_norm = np.mean(target_preds_norm, axis=0)

        curve_preds_norm = []
        for model in self.curve_models:
            pred = model.predict(
                [X_processed, avg_target_norm],
                batch_size=batch_size,
                verbose=0
            )
            curve_preds_norm.append(pred)

        avg_curve_norm = np.mean(curve_preds_norm, axis=0)
        curves = self.inverse_transform_curve(avg_curve_norm)

        targets = self.target_scaler.inverse_transform(avg_target_norm)

        return curves, targets

    def predict_from_csv(self, input_file, output_file=None, chunk_size=10000):
        try:
            reader = pd.read_csv(input_file, chunksize=chunk_size)
            full_results = []
            total_rows = sum(1 for _ in open(input_file)) - 1

            with tqdm(total=total_rows, desc="Prediction progress") as pbar:
                for chunk in reader:
                    try:
                        X = chunk.iloc[:, 0:24].values

                        curves, targets = self.predict_curves(X)

                        chunk_results = []
                        for i in range(len(X)):
                            chunk_results.append({
                                'input_features': X[i].tolist(),
                                'predicted_energy': float(targets[i, 0]),
                                'predicted_max_force': float(targets[i, 1]),
                                'predicted_curve': curves[i].tolist()
                            })

                        full_results.extend(chunk_results)
                        pbar.update(len(X))
                    except Exception as e:
                        print(f"Error processing chunk: {str(e)}")
                        continue
            result_df = pd.DataFrame(full_results)
            if output_file:
                os.makedirs(Config.RESULT_SAVE_DIR, exist_ok=True)
                save_path = f"{Config.RESULT_SAVE_DIR}/{output_file}"
                result_df.to_csv(save_path, index=False)
                print(f"Predictions saved to {save_path}")
            return result_df
        except Exception as e:
            raise RuntimeError(f"CSV prediction failed: {str(e)}")


if __name__ == "__main__":
    try:
        predictor = Predictor()
        demo_input = np.random.rand(3, 24)
        curves, targets = predictor.predict_curves(demo_input)
        print("\nExample 1 predictions:")
        print("Target predictions:\n", targets)
        print("Curve shape:", curves.shape)
        input_csv = "All-simple.csv"
        output_csv = "predictions.csv"
        print(f"\nStarting prediction for file: {input_csv}")
        results = predictor.predict_from_csv(input_csv, output_csv)
        print("\nPrediction complete. First 5 samples summary:")
        print(results[['predicted_energy', 'predicted_max_force']].head())
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()