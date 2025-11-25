"""
Neural Network to Predict Fantasy Football Player Value
Predicts fantasy points based on historical stats, ADP, and team strength
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt

class FantasyPlayerNN:
    """
    Neural network that predicts fantasy football player performance
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def build_model(self, input_size):
        """
        Build the neural network architecture
        
        Args:
            input_size: number of input features
        """
        
        # Input layer
        inputs = keras.Input(shape=(input_size,))
        
        # Hidden layers - progressively smaller
        x = keras.layers.Dense(128, activation='relu', name='hidden1')(inputs)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(64, activation='relu', name='hidden2')(x)
        x = keras.layers.Dropout(0.3)(x)
        
        x = keras.layers.Dense(32, activation='relu', name='hidden3')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output layer - predicted fantasy points
        outputs = keras.layers.Dense(1, activation='linear', name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='fantasy_predictor')
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean squared error
            metrics=['mae', 'mse']  # Mean absolute error and MSE
        )
        
        return self.model
    
    def prepare_features(self, df):
        """
        Prepare feature matrix from player dataframe
        
        Args:
            df: dataframe with player data
            
        Returns:
            numpy array of features
        """
        
        # Define feature columns (update based on your CSV structure)
        feature_cols = [
            'pos_qb', 'pos_rb', 'pos_wr', 'pos_te',
            'team_win_percentage',  # Updated from team
            'fantasy_adp',
            'age',
            'is_rookie',  # Now 0 or 1
            'nfl_draft_round',
            'seasons_played',
            'avg_games',
            'avg_completions',
            'avg_attempts',
            'avg_passing_yards',
            'avg_passing_tds',
            'avg_interceptions',
            'avg_carries',
            'avg_rushing_yards',
            'avg_rushing_tds',
            'avg_receptions',
            'avg_targets',
            'avg_receiving_yards',
            'avg_receiving_tds',
            'avg_points_per_game'
        ]
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Extract features
        X = df[feature_cols].values
        
        return X
    
    def train(self, train_df, actual_df, test_size=0.2, epochs=100, batch_size=32):
        """
        Train the neural network
        
        Args:
            train_df: dataframe with player features (condensed data)
            actual_df: dataframe with actual fantasy points (2024 stats)
            test_size: fraction of data to use for validation
            epochs: number of training epochs
            batch_size: batch size for training
            
        Returns:
            training history
        """
        
        print("=" * 80)
        print("TRAINING FANTASY FOOTBALL NEURAL NETWORK")
        print("=" * 80)
        
        # Merge training data with actual results
        print("\nMerging feature data with actual results...")
        training_data = train_df.merge(
            actual_df[['first_name', 'last_name', 'fantasy_points_ppr']],
            on=['first_name', 'last_name'],
            how='inner'
        )
        
        print(f"Training on {len(training_data)} players")
        
        # Prepare features and targets
        print("\nPreparing features...")
        X = self.prepare_features(training_data)
        y = training_data['fantasy_points_ppr'].values
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Scale features
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Build model
        print("\nBuilding model...")
        self.build_model(input_size=X_train_scaled.shape[1])
        self.model.summary()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
        
        # Train
        print("\nTraining model...")
        print("-" * 80)
        
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        val_loss, val_mae, val_mse = self.model.evaluate(
            X_val_scaled, y_val, verbose=0
        )
        
        print(f"\nValidation Metrics:")
        print(f"  Loss (MSE): {val_loss:.2f}")
        print(f"  MAE: {val_mae:.2f} fantasy points")
        print(f"  RMSE: {np.sqrt(val_mse):.2f} fantasy points")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def predict(self, player_df):
        """
        Predict fantasy points for players
        
        Args:
            player_df: dataframe with player features
            
        Returns:
            numpy array of predictions
        """
        X = self.prepare_features(player_df)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()
    
    def plot_training_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # MAE plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.title('Training and Validation MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("\n✅ Training plots saved to 'training_history.png'")
        plt.close()
    
    def save(self, filepath='fantasy_nn_model.h5'):
        """Save model and scaler"""
        self.model.save(filepath)
        
        with open(filepath.replace('.h5', '_scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(filepath.replace('.h5', '_features.pkl'), 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"\n✅ Model saved to '{filepath}'")
    
    def load(self, filepath='fantasy_nn_model.h5'):
        """Load model and scaler"""
        self.model = keras.models.load_model(filepath)
        
        with open(filepath.replace('.h5', '_scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(filepath.replace('.h5', '_features.pkl'), 'rb') as f:
            self.feature_columns = pickle.load(f)
        
        print(f"✅ Model loaded from '{filepath}'")


def main():
    """Main training script"""
    
    # Load data
    print("Loading player data...")
    condensed_df = pd.read_csv('nfl_players_condensed.csv')
    actual_2024_df = pd.read_csv('nfl_players_2024_stats.csv')
    
    print(f"Condensed data: {len(condensed_df)} players")
    print(f"2024 actual data: {len(actual_2024_df)} players")
    
    # Initialize and train network
    nn = FantasyPlayerNN()
    history = nn.train(condensed_df, actual_2024_df, epochs=150, batch_size=32)
    
    # Save model
    nn.save('fantasy_nn_model.h5')
    
    # Test predictions on sample players
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    
    # Merge for testing
    test_data = condensed_df.merge(
        actual_2024_df[['first_name', 'last_name', 'fantasy_points_ppr']],
        on=['first_name', 'last_name'],
        how='inner'
    )
    
    # Get predictions
    predictions = nn.predict(test_data)
    
    # Show top 10 by ADP
    test_data['predicted_points'] = predictions
    test_data = test_data.sort_values('fantasy_adp')
    
    print("\nTop 10 Players by ADP:")
    print("-" * 80)
    
    for i in range(min(10, len(test_data))):
        row = test_data.iloc[i]
        
        # Determine position
        if row['pos_qb'] == 1:
            pos = 'QB'
        elif row['pos_rb'] == 1:
            pos = 'RB'
        elif row['pos_wr'] == 1:
            pos = 'WR'
        else:
            pos = 'TE'
        
        print(f"\n{i+1}. {row['first_name']} {row['last_name']} ({pos})")
        print(f"   ADP: {row['fantasy_adp']:.1f}")
        print(f"   Predicted: {row['predicted_points']:.1f} points")
        print(f"   Actual: {row['fantasy_points_ppr']:.1f} points")
        print(f"   Error: {abs(row['predicted_points'] - row['fantasy_points_ppr']):.1f} points")
    
    # Calculate overall accuracy
    print("\n" + "=" * 80)
    print("OVERALL ACCURACY")
    print("=" * 80)
    
    mae = np.mean(np.abs(test_data['predicted_points'] - test_data['fantasy_points_ppr']))
    rmse = np.sqrt(np.mean((test_data['predicted_points'] - test_data['fantasy_points_ppr'])**2))
    
    print(f"\nMean Absolute Error: {mae:.2f} points")
    print(f"Root Mean Squared Error: {rmse:.2f} points")
    
    # Accuracy by position
    print("\nAccuracy by Position:")
    for pos_col, pos_name in [('pos_qb', 'QB'), ('pos_rb', 'RB'), ('pos_wr', 'WR'), ('pos_te', 'TE')]:
        pos_data = test_data[test_data[pos_col] == 1]
        if len(pos_data) > 0:
            pos_mae = np.mean(np.abs(pos_data['predicted_points'] - pos_data['fantasy_points_ppr']))
            print(f"  {pos_name}: {pos_mae:.2f} points (n={len(pos_data)})")


if __name__ == "__main__":
    main()