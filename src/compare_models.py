import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras import regularizers, Model

# =============================================================================
# PASTE YOUR THREE MODEL ARCHITECTURES HERE
# =============================================================================

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

    phi_0 = Dense(units=256, activation='elu', kernel_initializer='RandomNormal', name='phi_0_1')(x_input)
    phi_0 = Dense(units=256, activation='elu', kernel_initializer='RandomNormal', name='phi_0_2')(phi_0)

    y_preds = [None] * num_treatments

    if scenario == 'medication':
        phi_1 = Dense(units=256, activation='elu', kernel_initializer='RandomNormal', name='phi_1')(phi_0)
        phi_2 = Dense(units=128, activation='elu', kernel_initializer='RandomNormal', name='phi_2')(phi_1)
        phi_3 = Dense(units=64, activation='elu', kernel_initializer='RandomNormal', name='phi_3')(phi_2)

        # Make the final prediction heads larger as well
        h0_out = Dense(units=32, activation='elu', name='h0_hidden')(phi_0)
        h0_out = Dense(units=1, name='y0_pred')(h0_out)

        h1_out = Dense(units=32, activation='elu', name='h1_hidden')(phi_1)
        h1_out = Dense(units=1, name='y1_pred')(h1_out)

        h2_out = Dense(units=32, activation='elu', name='h2_hidden')(phi_2)
        h2_out = Dense(units=1, name='y2_pred')(h2_out)
        
        # For the max dose, one can think about connecting it to phi_2 for stability. Maybe next time
        h3_out = Dense(units=1, name='y3_pred')(phi_3) 

        y_preds = [h0_out, h1_out, h2_out, h3_out]

    elif scenario == 'education':
        phi_A = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_A')(phi_0)
        phi_B = Dense(units=200, activation='elu', kernel_initializer='RandomNormal', name='phi_B')(phi_0)

        # Make the final prediction heads larger
        hA_hidden = Dense(units=64, activation='elu', name='hA_hidden')
        hB_hidden = Dense(units=64, activation='elu', name='hB_hidden')
        
        y0_hidden = hA_hidden(phi_A)
        y2_hidden = hA_hidden(phi_A)
        
        y1_hidden = hB_hidden(phi_B)
        y3_hidden = hB_hidden(phi_B)

        y_preds[0] = Dense(units=1, name='y0_pred')(y0_hidden)
        y_preds[2] = Dense(units=1, name='y2_pred')(y2_hidden)
        y_preds[1] = Dense(units=1, name='y1_pred')(y1_hidden)
        y_preds[3] = Dense(units=1, name='y3_pred')(y3_hidden)

    concat_y_preds = Concatenate(axis=1, name='concat_predictions')(y_preds)
    model_output = Concatenate(axis=1, name='final_output_with_phi_0')([concat_y_preds, phi_0])
    model = Model(inputs=x_input, outputs=model_output)
    return model

# =============================================================================
# MAIN SCRIPT TO BUILD AND COMPARE MODELS
# =============================================================================
if __name__ == '__main__':
    # --- Define common parameters for model instantiation ---
    INPUT_DIM = 25
    NUM_TREATMENTS = 4
    L2_REG = 1e-3

    # --- Instantiate and summarize each model ---
    print("-" * 65)
    print("Model 1: Baseline MCFRNet (Independent Heads)")
    print("-" * 65)
    model_1 = make_mcfr_net(INPUT_DIM, NUM_TREATMENTS, L2_REG)
    model_1.summary()

    print("\n" + "=" * 65)
    print("Model 2: Structured-MCFRNet (Fully Shared Head)")
    print("=" * 65)
    model_2 = make_structured_mcfr_net(INPUT_DIM, NUM_TREATMENTS, L2_REG)
    model_2.summary()

    print("\n" + "=" * 65)
    print("Model 3a: Hierarchical-MCFRNet (Education Scenario)")
    print("=" * 65)
    model_3a = make_hierarchical_mcfr_net(INPUT_DIM, NUM_TREATMENTS, L2_REG, scenario='education')
    model_3a.summary()
    
    print("\n" + "=" * 65)
    print("Model 3b: Hierarchical-MCFRNet (Medication Scenario)")
    print("=" * 65)
    model_3b = make_hierarchical_mcfr_net(INPUT_DIM, NUM_TREATMENTS, L2_REG, scenario='medication')
    model_3b.summary()