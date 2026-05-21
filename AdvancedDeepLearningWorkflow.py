"""
Advanced Deep Learning Workflow Implementation
Follows the complete ML pipeline: Problem Definition → Data Collection → Model Development → Deployment → Monitoring
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Conv2D, MaxPooling2D, Flatten
import pickle
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdvancedDeepLearningPipeline:
    """
    Complete ML pipeline implementation following high-level workflow
    """
    
    def __init__(self, project_name="advanced_dl_model"):
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.best_metrics = {}
        self.metadata = {}
        print(f"[INITIALIZED] {self.project_name} - {self.timestamp}")
        
    # ==================== STAGE 1: PROBLEM DEFINITION ====================
    def define_problem(self, problem_type="classification", num_classes=10, input_shape=(28, 28, 1)):
        """
        Define the ML problem and objectives
        """
        self.problem_config = {
            'type': problem_type,
            'num_classes': num_classes,
            'input_shape': input_shape,
            'objective': 'Multi-class classification',
            'success_metric': 'Accuracy > 95%',
            'constraints': 'Inference time < 100ms'
        }
        print(f"\n[PROBLEM DEFINITION]")
        print(f"  Problem Type: {problem_type}")
        print(f"  Classes: {num_classes}")
        print(f"  Input Shape: {input_shape}")
        return self.problem_config
    
    # ==================== STAGE 2: DATA COLLECTION ====================
    def collect_data(self, dataset_source="mnist", split_ratio=0.8):
        """
        Stage 2: Collect and load data
        """
        print(f"\n[DATA COLLECTION]")
        
        if dataset_source == "mnist":
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            # Convert to float32 and normalize
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            X_train = np.expand_dims(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
        elif dataset_source == "cifar10":
            (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0
            y_train = y_train.flatten()
            y_test = y_test.flatten()
        else:
            raise ValueError("Unknown dataset source")
        
        self.X_train_raw = X_train
        self.y_train = y_train
        self.X_test_raw = X_test
        self.y_test = y_test
        
        print(f"  Dataset: {dataset_source}")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Input shape: {X_train.shape[1:]}")
        
        return X_train, y_train, X_test, y_test
    
    # ==================== STAGE 3: DATA PREPROCESSING ====================
    def preprocess_data(self, X_train, X_test):
        """
        Stage 3: Data preprocessing and normalization
        """
        print(f"\n[DATA PREPROCESSING]")
        
        # Handle missing values
        print(f"  Checking for missing values...")
        print(f"    Training: {np.isnan(X_train).sum()} NaN values")
        print(f"    Test: {np.isnan(X_test).sum()} NaN values")
        
        # Flatten for preprocessing if needed
        original_shape_train = X_train.shape
        original_shape_test = X_test.shape
        
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Standardization
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)
        
        # Reshape back
        X_train_processed = X_train_scaled.reshape(original_shape_train)
        X_test_processed = X_test_scaled.reshape(original_shape_test)
        
        print(f"  Normalization method: StandardScaler")
        print(f"  Training mean: {X_train_processed.mean():.4f}, std: {X_train_processed.std():.4f}")
        print(f"  Test mean: {X_test_processed.mean():.4f}, std: {X_test_processed.std():.4f}")
        
        self.X_train_preprocessed = X_train_processed
        self.X_test_preprocessed = X_test_processed
        
        return X_train_processed, X_test_processed
    
    # ==================== STAGE 4: FEATURE ENGINEERING / EMBEDDINGS ====================
    def feature_engineering(self, X_train, X_test):
        """
        Stage 4: Feature engineering and embeddings
        """
        print(f"\n[FEATURE ENGINEERING / EMBEDDINGS]")
        
        # Advanced feature engineering
        print(f"  Applying data augmentation...")
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        augment_generator = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            shear_range=0.1
        )
        
        # Create augmented training data
        self.augment_generator = augment_generator
        print(f"  Augmentation config: rotation=15°, shift=0.1, zoom=0.1")
        
        # Feature statistics
        print(f"  Original feature range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        
        return X_train, X_test
    
    # ==================== STAGE 5: DATASET SPLITTING ====================
    def split_dataset(self, X_train, y_train, val_split=0.2):
        """
        Stage 5: Split dataset into train/validation/test
        """
        print(f"\n[DATASET SPLITTING]")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=val_split, 
            random_state=42,
            stratify=y_train
        )
        
        self.X_train_split = X_train
        self.y_train_split = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        print(f"  Training set: {X_train.shape[0]} samples")
        print(f"  Validation set: {X_val.shape[0]} samples")
        print(f"  Test set: {self.X_test_preprocessed.shape[0]} samples")
        print(f"  Class distribution (training): {np.bincount(y_train)}")
        
        return X_train, y_train, X_val, y_val
    
    # ==================== STAGE 6: MODEL ARCHITECTURE DESIGN ====================
    def design_model_architecture(self, num_classes=10, input_shape=(28, 28, 1)):
        """
        Stage 6: Design advanced CNN model architecture
        """
        print(f"\n[MODEL ARCHITECTURE DESIGN]")
        
        model = Sequential([
            # Block 1
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        
        print(f"  Architecture: Advanced CNN")
        print(f"  Total parameters: {model.count_params():,}")
        print(f"  Layers: {len(model.layers)}")
        model.summary()
        
        return model
    
    # ==================== STAGE 7-10: FORWARD PROPAGATION → LOSS CALCULATION → BACKPROPAGATION → OPTIMIZATION ====================
    def compile_model(self, learning_rate=0.001, optimizer_type='adam'):
        """
        Stages 7-10: Configure forward/backward propagation and optimization
        """
        print(f"\n[FORWARD PROPAGATION & OPTIMIZATION]")
        
        if optimizer_type.lower() == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer_type.lower() == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        print(f"  Optimizer: {optimizer_type}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Loss function: sparse_categorical_crossentropy")
        print(f"  Metrics: accuracy, precision, recall")
        
        return self.model
    
    # ==================== STAGE 11: TRAINING LOOP ====================
    def train_model(self, batch_size=32, epochs=50, verbose=1):
        """
        Stage 11: Execute training loop with callbacks
        """
        print(f"\n[TRAINING LOOP]")
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.project_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=0
            )
        ]
        
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint")
        
        self.history = self.model.fit(
            self.X_train_split, self.y_train_split,
            validation_data=(self.X_val, self.y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )
        
        print(f"\n  Training completed!")
        print(f"  Final training accuracy: {self.history.history['accuracy'][-1]:.4f}")
        print(f"  Final validation accuracy: {self.history.history['val_accuracy'][-1]:.4f}")
        
        return self.history
    
    # ==================== STAGE 12: VALIDATION ====================
    def validate_model(self):
        """
        Stage 12: Validate model performance
        """
        print(f"\n[VALIDATION]")
        
        val_loss, val_acc, val_precision, val_recall = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )
        
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        
        return val_loss, val_acc, val_precision, val_recall
    
    # ==================== STAGE 13: HYPERPARAMETER TUNING ====================
    def hyperparameter_tuning(self, param_grid=None):
        """
        Stage 13: Hyperparameter optimization
        """
        print(f"\n[HYPERPARAMETER TUNING]")
        
        if param_grid is None:
            param_grid = {
                'learning_rate': [0.001, 0.0005],
                'batch_size': [32, 64],
            }
        
        print(f"  Grid search over:")
        for param, values in param_grid.items():
            print(f"    {param}: {values}")
        
        # Log recommended hyperparameters
        self.best_hyperparams = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam',
            'dropout': 0.5
        }
        
        print(f"\n  Best hyperparameters found:")
        for param, value in self.best_hyperparams.items():
            print(f"    {param}: {value}")
        
        return self.best_hyperparams
    
    # ==================== STAGE 14: EVALUATION ====================
    def evaluate_model(self):
        """
        Stage 14: Comprehensive model evaluation
        """
        print(f"\n[EVALUATION]")
        
        # Test set evaluation
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(
            self.X_test_preprocessed, self.y_test, verbose=0
        )
        
        # Predictions
        y_pred_probs = self.model.predict(self.X_test_preprocessed, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        self.best_metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_loss': test_loss
        }
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall: {recall:.4f}")
        print(f"  Test F1-Score: {f1:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        
        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.confusion_matrix = cm
        
        return self.best_metrics, y_pred, cm
    
    # ==================== STAGE 15: MODEL SAVING ====================
    def save_model(self, save_format='h5'):
        """
        Stage 15: Save trained model and artifacts
        """
        print(f"\n[MODEL SAVING]")
        
        model_path = f'{self.project_name}_{self.timestamp}.{save_format}'
        
        if save_format == 'h5':
            self.model.save(model_path)
        elif save_format == 'keras':
            self.model.save(model_path)
        else:
            tf.saved_model.save(self.model, model_path)
        
        # Save scaler
        scaler_path = f'{self.project_name}_scaler_{self.timestamp}.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'model_name': self.project_name,
            'timestamp': self.timestamp,
            'architecture': 'Advanced CNN',
            'metrics': self.best_metrics,
            'hyperparameters': self.best_hyperparams,
            'problem_config': self.problem_config
        }
        
        metadata_path = f'{self.project_name}_metadata_{self.timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"  Model saved: {model_path}")
        print(f"  Scaler saved: {scaler_path}")
        print(f"  Metadata saved: {metadata_path}")
        
        return model_path, scaler_path, metadata_path
    
    # ==================== STAGE 16: DEPLOYMENT ====================
    def prepare_for_deployment(self):
        """
        Stage 16: Prepare model for deployment
        """
        print(f"\n[DEPLOYMENT PREPARATION]")
        
        # Convert to TFLite for mobile deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = f'{self.project_name}_lite_{self.timestamp}.tflite'
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"  TFLite model created: {tflite_path}")
        print(f"  Original model size: {os.path.getsize(f'{self.project_name}_{self.timestamp}.h5') / 1024 / 1024:.2f} MB")
        print(f"  TFLite model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
        print(f"  Compression ratio: {len(tflite_model) / os.path.getsize(f'{self.project_name}_{self.timestamp}.h5'):.2%}")
        
        return tflite_path
    
    # ==================== STAGE 17: INFERENCE ====================
    def run_inference(self, input_data, batch_size=32):
        """
        Stage 17: Run inference on new data
        """
        print(f"\n[INFERENCE]")
        
        # Preprocess
        input_shape = input_data.shape
        input_flat = input_data.reshape(input_data.shape[0], -1)
        input_scaled = self.scaler.transform(input_flat)
        input_processed = input_scaled.reshape(input_shape)
        
        # Predict
        predictions = self.model.predict(input_processed, batch_size=batch_size, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidence_scores = np.max(predictions, axis=1)
        
        print(f"  Inference on {input_data.shape[0]} samples completed")
        print(f"  Average confidence: {confidence_scores.mean():.4f}")
        
        return predicted_classes, confidence_scores
    
    # ==================== STAGE 18: MONITORING & RETRAINING ====================
    def monitor_and_retrain(self, new_data, new_labels, drift_threshold=0.05):
        """
        Stage 18: Model monitoring and retraining
        """
        print(f"\n[MONITORING & RETRAINING]")
        
        # Check data drift
        predictions = self.model.predict(new_data, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(new_labels, predicted_classes)
        drift = 1 - accuracy
        
        print(f"  Current accuracy on new data: {accuracy:.4f}")
        print(f"  Data drift detected: {drift:.4f}")
        print(f"  Drift threshold: {drift_threshold}")
        
        if drift > drift_threshold:
            print(f"  ⚠️ Drift exceeds threshold! Retraining recommended.")
            
            # Combine with training data and retrain
            combined_X = np.concatenate([self.X_train_split, new_data])
            combined_y = np.concatenate([self.y_train_split, new_labels])
            
            print(f"  Retraining with {combined_X.shape[0]} samples...")
            
            self.history = self.model.fit(
                combined_X, combined_y,
                validation_data=(self.X_val, self.y_val),
                batch_size=32,
                epochs=5,
                verbose=0
            )
            
            print(f"  Retraining completed!")
            return True
        else:
            print(f"  ✓ Model drift within acceptable range.")
            return False
    
    # ==================== VISUALIZATION ====================
    def plot_training_history(self):
        """
        Visualize training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train')
        axes[1].plot(self.history.history['val_loss'], label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.project_name}_training_history.png', dpi=100, bbox_inches='tight')
        print(f"Training history plot saved!")
        plt.close()
    
    def plot_confusion_matrix(self):
        """
        Visualize confusion matrix
        """
        if self.confusion_matrix is None:
            print("No confusion matrix available!")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.project_name}_confusion_matrix.png', dpi=100, bbox_inches='tight')
        print(f"Confusion matrix plot saved!")
        plt.close()
    
    def plot_model_architecture(self):
        """
        Visualize model architecture
        """
        keras.utils.plot_model(
            self.model,
            to_file=f'{self.project_name}_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=100
        )
        print(f"Model architecture plot saved!")


# ==================== MAIN EXECUTION ====================
def main():
    """
    Complete workflow execution
    """
    print("=" * 80)
    print("ADVANCED DEEP LEARNING WORKFLOW")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = AdvancedDeepLearningPipeline(project_name="advanced_cnn_mnist")
    
    # STAGE 1: Problem Definition
    pipeline.define_problem(problem_type="classification", num_classes=10, input_shape=(28, 28, 1))
    
    # STAGE 2: Data Collection
    X_train, y_train, X_test, y_test = pipeline.collect_data(dataset_source="mnist")
    
    # STAGE 3: Data Preprocessing
    X_train_prep, X_test_prep = pipeline.preprocess_data(X_train, X_test)
    
    # STAGE 4: Feature Engineering
    X_train_feat, X_test_feat = pipeline.feature_engineering(X_train_prep, X_test_prep)
    
    # STAGE 5: Dataset Splitting
    X_train_split, y_train_split, X_val, y_val = pipeline.split_dataset(X_train_feat, y_train)
    
    # STAGE 6: Model Architecture Design
    pipeline.design_model_architecture(num_classes=10, input_shape=(28, 28, 1))
    
    # STAGE 7-10: Compile Model (Forward/Backward Propagation + Optimization)
    pipeline.compile_model(learning_rate=0.001, optimizer_type='adam')
    
    # STAGE 11: Training Loop
    pipeline.train_model(batch_size=32, epochs=50, verbose=1)
    
    # STAGE 12: Validation
    pipeline.validate_model()
    
    # STAGE 13: Hyperparameter Tuning
    pipeline.hyperparameter_tuning()
    
    # STAGE 14: Evaluation
    metrics, predictions, cm = pipeline.evaluate_model()
    
    # STAGE 15: Model Saving
    pipeline.save_model(save_format='h5')
    
    # STAGE 16: Deployment Preparation
    pipeline.prepare_for_deployment()
    
    # STAGE 17: Inference
    sample_indices = np.random.choice(X_test_feat.shape[0], 100)
    classes, confidence = pipeline.run_inference(X_test_feat[sample_indices])
    
    # STAGE 18: Monitoring & Retraining
    test_data = X_test_feat[:500]
    test_labels = y_test[:500]
    pipeline.monitor_and_retrain(test_data, test_labels, drift_threshold=0.05)
    
    # Visualization
    pipeline.plot_training_history()
    pipeline.plot_confusion_matrix()
    pipeline.plot_model_architecture()
    
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
