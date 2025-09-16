import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import combinations

from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras import regularizers, Model
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

# Silence TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# =============================================================================
# SECTION 1: DATA GENERATION
# =============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_synthetic_data(n_samples=5000, n_features=25, scenario='education', kappa=1.0, seed=42):
    """Generates synthetic data for multi-treatment causal inference."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    Y_potential = np.zeros((n_samples, 4))
    scores = np.zeros((n_samples, 4))
    epsilon = np.random.randn(n_samples) * 0.1

    if scenario == 'education':
        Y_potential[:, 0] = 5 * sigmoid(X[:, 0]) + 7 + epsilon
        Y_potential[:, 1] = 3 * X[:, 1] + 2 * X[:, 3] + (X[:, 1] * X[:, 3]) + 7 + epsilon
        Y_potential[:, 2] = 4 * X[:, 2] - 2 * np.power(1 - sigmoid(X[:, 1]), 2) + 7 + epsilon
        Y_potential[:, 3] = 8 - 3 * sigmoid(X[:, 0]) + 7 + epsilon
        scores[:, 0] = 1.5 * X[:, 0]
        scores[:, 1] = 1.2 * X[:, 1] + 0.5 * X[:, 3]
        scores[:, 2] = 1.5 * X[:, 2]
        scores[:, 3] = -1.5 * X[:, 0]
    elif scenario == 'medication':
        doses = np.array([0, 20, 40, 60])
        X[:, 3] = np.random.binomial(1, 0.5, n_samples)
        for t in range(4):
            dose_val = doses[t]
            efficacy = (10 * (1 + sigmoid(X[:, 0]))) / (1 + np.exp(-((dose_val / 20) * (1 - 0.5 * X[:, 3]) - 2)))
            side_effect = np.power(dose_val / 20, 2) * (0.2 * sigmoid(X[:, 1]) + 0.2 * (1 - sigmoid(X[:, 2])))
            Y_potential[:, t] = efficacy - side_effect + 5 + epsilon
        scores[:, 0] = -1.5 * X[:, 0] 
        scores[:, 1] = 1.0 * X[:, 0] - 0.5 * X[:, 1] - 0.5 * (1 - X[:, 2])
        scores[:, 2] = 2.0 * X[:, 0] - 1.0 * X[:, 1] - 1.0 * (1 - X[:, 2])
        scores[:, 3] = 2.5 * X[:, 0] - 2.0 * X[:, 1] - 2.0 * (1 - X[:, 2])
    elif scenario == 'fertilizer':
        # Ordered, monotonic treatments: T0=0kg, T1=15kg, T2=30kg, T3=45kg
        # Covariates: X1=Soil Quality, X2=Sunlight
        doses = np.array([0, 15, 30, 45])
        
        for t in range(4):
            # Logarithmic dose-response, interacting with soil quality and sunlight
            Y_potential[:, t] = (10 + 5 * sigmoid(X[:, 0])) * np.log(doses[t] + 1) + 2 * sigmoid(X[:, 1]) + 5 + epsilon

        # Treatment assignment scores (confounding based on soil quality)
        scores[:, 0] = 0 # Baseline for no fertilizer
        scores[:, 1] = 1.0 * X[:, 0]
        scores[:, 2] = 2.0 * X[:, 0]
        scores[:, 3] = 3.0 * X[:, 0]
    else:
        raise ValueError("Scenario must be 'education', 'medication', or 'fertilizer'.")

    scores_kappa = scores * kappa
    propensity_scores = np.exp(scores_kappa - np.max(scores_kappa, axis=1, keepdims=True))
    propensity_scores /= np.sum(propensity_scores, axis=1, keepdims=True)
    
    t = np.array([np.random.choice(4, p=p) for p in propensity_scores])
    y_factual = Y_potential[np.arange(n_samples), t]
    
    return X, t, y_factual, Y_potential

# =============================================================================
# SECTION 2: MODEL ARCHITECTURE & LOSS
# (This is the code you provided, with a minor tweak for clarity)
# =============================================================================
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras import regularizers, Model
from tensorflow.keras.losses import Loss

# --- Model 1: Baseline MCFRNet (Independent Heads) ---
def make_mcfr_net(input_dim, num_treatments, reg_l2):
    x = Input(shape=(input_dim,), name='input')
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(x)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_2')(phi)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_3')(phi)
    
    output_heads = []
    for i in range(num_treatments):
        head_name = f'y{i}'
        hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name=f'{head_name}_hidden_1')(phi)
        hidden = Dense(units=100, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name=f'{head_name}_hidden_2')(hidden)
        predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name=f'{head_name}_predictions')(hidden)
        output_heads.append(predictions)

    concat_pred = Concatenate(axis=1, name='concat_predictions')(output_heads)
    model_output = Concatenate(axis=1, name='final_output_with_phi')([concat_pred, phi])
    model = Model(inputs=x, outputs=model_output)
    return model

# --- Model 2: Structured-MCFRNet (Shared Head) ---
def make_structured_mcfr_net(input_dim, num_treatments, reg_l2):
    x_input = Input(shape=(input_dim,), name='x_input')
    t_input = Input(shape=(1,), name='t_input') 
    
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(x_input)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_2')(phi)
    phi = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_3')(phi)

    concatenated_input = Concatenate(axis=1)([phi, t_input])

    hidden = Dense(units=256, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='h_hidden_1')(concatenated_input)
    hidden = Dense(units=256, activation='elu', kernel_regularizer=regularizers.l2(reg_l2), name='h_hidden_2')(hidden)
    y_pred = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y_predictions')(hidden)

    model_output = Concatenate(axis=1, name='final_output_with_phi')([y_pred, phi])
    
    model = Model(inputs=[x_input, t_input], outputs=model_output)
    return model

# --- Model 3: Hierarchical-MCFRNet (Partially Shared) ---
def make_hierarchical_mcfr_net(input_dim, num_treatments, reg_l2, scenario):
    x_input = Input(shape=(input_dim,), name='x_input')

    # --- ADJUSTMENT: These sizes were chosen to make params comparable ---
    phi_0 = Dense(units=256, activation='elu', kernel_initializer='RandomNormal', name='phi_0_1')(x_input)
    phi_0 = Dense(units=256, activation='elu', kernel_initializer='RandomNormal', name='phi_0_2')(phi_0)

    y_preds = [None] * num_treatments

    # --- THE FIX: ADDED 'fertilizer' to this condition ---
    if scenario in ['medication', 'fertilizer', 'education']:
        # --- Chained Hierarchy for Ordered Treatments ---
        phi_1 = Dense(units=128, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(phi_0)
        phi_2 = Dense(units=64, activation='elu', kernel_initializer='RandomNormal', name='phi_2')(phi_1)

        # Hypothesis heads connect to their deepest relevant representation
        h0_hidden = Dense(units=32, activation='elu', name='h0_hidden')(phi_0)
        h0_out = Dense(units=1, name='y0_pred')(h0_hidden)

        h1_hidden = Dense(units=32, activation='elu', name='h1_hidden')(phi_1)
        h1_out = Dense(units=1, name='y1_pred')(h1_hidden)

        h2_hidden = Dense(units=32, activation='elu', name='h2_hidden')(phi_2)
        h2_out = Dense(units=1, name='y2_pred')(h2_hidden)
        
        # Head for T3 connects to the last specialized layer, phi_2
        h3_hidden = Dense(units=32, activation='elu', name='h3_hidden')(phi_2)
        h3_out = Dense(units=1, name='y3_pred')(h3_hidden)

        y_preds = [h0_out, h1_out, h2_out, h3_out]

    # elif scenario == 'education':
    #     # --- Branched Hierarchy for Unordered Treatments ---
    #     phi_A = Dense(units=128, activation='elu', kernel_initializer='RandomNormal', name='phi_A')(phi_0)
    #     phi_B = Dense(units=128, activation='elu', kernel_initializer='RandomNormal', name='phi_B')(phi_0)

    #     # Re-usable hidden layers for each branch
    #     hA_hidden_layer = Dense(units=64, activation='elu', name='hA_hidden')
    #     hB_hidden_layer = Dense(units=64, activation='elu', name='hB_hidden')
        
    #     # Define final prediction layers
    #     y0_pred_layer = Dense(units=1, name='y0_pred')
    #     y1_pred_layer = Dense(units=1, name='y1_pred')
    #     y2_pred_layer = Dense(units=1, name='y2_pred')
    #     y3_pred_layer = Dense(units=1, name='y3_pred')

    #     # Connect the layers
    #     y_preds[0] = y0_pred_layer(hA_hidden_layer(phi_A))
    #     y_preds[2] = y2_pred_layer(hA_hidden_layer(phi_A))
        
    #     y_preds[1] = y1_pred_layer(hB_hidden_layer(phi_B))
    #     y_preds[3] = y3_pred_layer(hB_hidden_layer(phi_B))

    else:
        raise ValueError("Scenario not recognized in hierarchical model builder.")

    concat_y_preds = Concatenate(axis=1, name='concat_predictions')(y_preds)
    model_output = Concatenate(axis=1, name='final_output_with_phi_0')([concat_y_preds, phi_0])
    
    model = Model(inputs=x_input, outputs=model_output)
    return model

class MCFRNet_Loss(tf.keras.losses.Loss):
    def __init__(self, num_treatments, treatment_proportions, alpha=1.0, sigma=1.0, name='mcfrnet_loss'):
        # This line is the fix. It calls the parent constructor correctly.
        super().__init__(name=name)
        self.num_treatments = num_treatments
        self.treatment_proportions = tf.constant(treatment_proportions, dtype=tf.float32)
        self.alpha = alpha
        self.rbf_sigma = sigma

    def split_pred(self, concat_pred):
        preds = {}
        preds['y_preds'] = concat_pred[:, :self.num_treatments]
        preds['phi'] = concat_pred[:, self.num_treatments:]
        return preds

    def rbf_kernel(self, x, y):
        x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)
        y2 = tf.reduce_sum(y ** 2, axis=-1, keepdims=True)
        dist = x2 + tf.transpose(y2, (1, 0)) - 2. * tf.matmul(x, tf.transpose(y, (1, 0)))
        return tf.exp(-dist / tf.square(self.rbf_sigma))

    def regression_loss(self, concat_true, concat_pred):
        y_true = concat_true[:, 0]
        t_true = tf.cast(tf.squeeze(concat_true[:, 1]), dtype=tf.int32)
        p = self.split_pred(concat_pred)
        t_one_hot = tf.one_hot(t_true, self.num_treatments)
        y_pred_factual = tf.reduce_sum(p['y_preds'] * t_one_hot, axis=1)
        loss = tf.reduce_mean(tf.square(y_true - y_pred_factual))
        return loss


    def mmdsq_loss(self, concat_true, concat_pred):
        t_true = concat_true[:, 1]
        p = self.split_pred(concat_pred)
        phi = p['phi']

        phi_partitions = tf.dynamic_partition(phi, tf.cast(tf.squeeze(t_true), tf.int32), self.num_treatments)
        
        total_mmd = tf.constant(0.0, dtype=tf.float32)
        
        for i, j in combinations(range(self.num_treatments), 2):
            phi_i = phi_partitions[i]
            phi_j = phi_partitions[j]
            
            # Define a function to calculate MMD for this pair
            def calculate_mmd_for_pair():
                K_ii = self.rbf_kernel(phi_i, phi_i)
                K_jj = self.rbf_kernel(phi_j, phi_j)
                K_ij = self.rbf_kernel(phi_i, phi_j)
                
                m = tf.cast(tf.shape(phi_i)[0], phi.dtype)
                n = tf.cast(tf.shape(phi_j)[0], phi.dtype)
                
                # The unbiased MMD estimator
                mmd_ij = (tf.reduce_sum(K_ii - tf.linalg.diag(K_ii)) / (m * (m - 1.0)) +
                          tf.reduce_sum(K_jj - tf.linalg.diag(K_jj)) / (n * (n - 1.0)) -
                          2.0 * tf.reduce_sum(K_ij) / (m * n))
                
                weight = self.treatment_proportions[i] + self.treatment_proportions[j]
                return weight * mmd_ij

            # Define a function that returns zero if the condition is not met
            def return_zero():
                return tf.constant(0.0, dtype=tf.float32)

            # --- THE FIX ---
            # Use tf.cond to conditionally execute the MMD calculation
            # This is the TensorFlow equivalent of a Python 'if' statement
            condition = tf.logical_and(tf.shape(phi_i)[0] > 1, tf.shape(phi_j)[0] > 1)
            mmd_pair = tf.cond(condition, calculate_mmd_for_pair, return_zero)
            total_mmd += mmd_pair
            
        return total_mmd

    def call(self, concat_true, concat_pred):
        lossR = self.regression_loss(concat_true, concat_pred)
        lossIPM = self.mmdsq_loss(concat_true, concat_pred)
        return lossR + self.alpha * lossIPM

# =============================================================================
# SECTION 3: EVALUATION
# =============================================================================

def evaluate_model(model, X_test, Y_potential_test, y_scaler, num_treatments, model_type):
    """Calculates the PEHE and returns CATEs for a trained model."""
    
    # --- This part is new: Initialize est_cate for the return statement ---
    est_cate = None

    if model_type in ['mcfrnet', 'hierarchical_mcfr']:
        model_preds = model.predict(X_test)
        y_preds_scaled = model_preds[:, :num_treatments]
        y_preds = y_scaler.inverse_transform(y_preds_scaled)
        est_cate = y_preds[:, 1:] - y_preds[:, [0]]
    
    elif model_type == 'structured_mcfr':
        y_preds_scaled_list = []
        for t in range(num_treatments):
            t_input = np.full((X_test.shape[0], 1), t, dtype=np.float32)
            model_preds = model.predict([X_test, t_input])
            y_pred_t = model_preds[:, 0]
            y_preds_scaled_list.append(y_pred_t)
        y_preds_scaled = np.stack(y_preds_scaled_list, axis=1)
        y_preds = y_scaler.inverse_transform(y_preds_scaled)
        est_cate = y_preds[:, 1:] - y_preds[:, [0]]
        
    elif model_type == 'causal_forest':
        # Causal Forest's .effect() directly gives the CATE.
        est_cate = np.stack([model.effect(X_test, T0=0, T1=t) for t in range(1, num_treatments)], axis=1)
        
    else:
        raise ValueError(f"Evaluation logic for {model_type} not defined.")

    # Now calculate PEHE using the consistent est_cate variable
    true_cate = Y_potential_test[:, 1:] - Y_potential_test[:, [0]]
    pehe = np.sqrt(np.mean(np.square(true_cate - est_cate), axis=0))
    
    # --- THE FIX ---
    # Ensure all branches return a dictionary with the same keys.
    return {
        'pehe': pehe,
        'est_cate': est_cate,
        'true_cate': true_cate
    }

# =============================================================================
# SECTION 4: MAIN EXECUTION SCRIPT
# =============================================================================

def main(args):
    # 1. Generate Data
    print(f"--- Generating dataset for scenario: {args.scenario} ---")
    X, T, Y_factual, Y_potential = generate_synthetic_data(
        n_samples=args.n_samples, 
        scenario=args.scenario,
        kappa=args.kappa,
        seed=args.seed
    )
    
    # 2. Split Data
    indices = np.arange(args.n_samples)
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=args.seed)
    
    X_train, X_test = X[train_indices], X[test_indices]
    T_train, T_test = T[train_indices], T[test_indices]
    Y_factual_train, _ = Y_factual[train_indices], Y_factual[test_indices]
    _, Y_potential_test = Y_potential[train_indices], Y_potential[test_indices]
    
    # 3. Prepare Data for Model
    y_scaler = StandardScaler().fit(Y_factual_train.reshape(-1, 1))
    ys_train = y_scaler.transform(Y_factual_train.reshape(-1, 1))
    
    yt_train = np.concatenate([ys_train, T_train.reshape(-1, 1)], axis=1)
    
    num_treatments = len(np.unique(T))
    treatment_proportions = np.array([np.mean(T_train == i) for i in range(num_treatments)])
    
    # 4. Select and Train Model
    print(f"--- Training model: {args.model_type} ---")
    
    # Common callbacks and loss function for all our NN models
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.TerminateOnNaN()
    ]
    loss = MCFRNet_Loss(
        num_treatments=num_treatments,
        treatment_proportions=treatment_proportions,
        alpha=args.alpha_ipm
    )
    
    # --- Model Selection Logic ---
    if args.model_type == 'mcfrnet':
        model = make_mcfr_net(
            input_dim=X_train.shape[1], num_treatments=num_treatments, reg_l2=args.l2_reg)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=loss)
        
        model.fit(
            X_train, yt_train,
            validation_split=0.2, epochs=args.epochs,
            batch_size=args.batch_size, callbacks=callbacks, verbose=1)

    elif args.model_type == 'structured_mcfr':
        model = make_structured_mcfr_net(
            input_dim=X_train.shape[1], num_treatments=num_treatments, reg_l2=args.l2_reg)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=loss)
        
        X_train_main, X_val, yt_train_main, yt_val = train_test_split(X_train, yt_train, test_size=0.2, random_state=args.seed)
        
        model.fit(
            [X_train_main, yt_train_main[:, [1]]],
            yt_train_main,
            validation_data=([X_val, yt_val[:, [1]]], yt_val),
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=callbacks, verbose=1)
            
    elif args.model_type == 'hierarchical_mcfr':
        model = make_hierarchical_mcfr_net(
            input_dim=X_train.shape[1], num_treatments=num_treatments, 
            reg_l2=args.l2_reg, scenario=args.scenario)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss=loss)
        
        model.fit(
            X_train, yt_train,
            validation_split=0.2, epochs=args.epochs,
            batch_size=args.batch_size, callbacks=callbacks, verbose=1)
            
    elif args.model_type == 'causal_forest':
            try:
                model_t = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=args.seed)
                model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=args.seed)
                
                model = CausalForestDML(model_y=model_y,
                                        model_t=model_t,
                                        discrete_treatment=True,
                                        random_state=args.seed)
                
                # EconML requires T to be a 1D array for classification
                model.fit(Y=Y_factual_train, T=T_train.ravel(), X=X_train)
            except Exception as e:
                print(f"Causal Forest failed to train for this run. Error: {e}")
                # Exit gracefully so the bash script can continue with the next run
                return 
            
    # 5. Evaluate Model
    eval_results = evaluate_model(model, X_test, Y_potential_test, y_scaler, num_treatments, args.model_type)
    
    pehe_results = eval_results['pehe']
    print("\n--- Evaluation Results ---")
    print(f"PEHE (T1 vs T0): {pehe_results[0]:.4f}")
    print(f"PEHE (T2 vs T0): {pehe_results[1]:.4f}")
    print(f"PEHE (T3 vs T0): {pehe_results[2]:.4f}")

    if not args.dry_run:
        np.savez(f"{args.output_dir}/cate_results.npz",
                est_cate=eval_results['est_cate'],
                true_cate=eval_results['true_cate'],
                pehe=eval_results['pehe'],
                X_test=X_test)
        print(f"CATE results and PEHE saved in {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Causal Inference Model Experiments")
    parser.add_argument('--scenario', type=str, default='education', choices=['education', 'medication', 'fertilizer'], help='Simulation scenario')
    parser.add_argument('--model_type', type=str, default='mcfrnet', 
                        choices=['mcfrnet', 'structured_mcfr', 'hierarchical_mcfr', 'causal_forest'])
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results.')
    parser.add_argument('--dry_run', action='store_true', help='If true, script will not train or evaluate.')
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--kappa', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--l2_reg', type=float, default=1e-3)
    parser.add_argument('--alpha_ipm', type=float, default=1.0)
    
    args = parser.parse_args()
    main(args)