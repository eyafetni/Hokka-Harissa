# ----------------------------------------------------------------
# IMPORTANT: This template will be used to evaluate your solution.
#
# Do NOT change the function signatures.
# And ensure that your code runs within the time limits.
# The time calculation is computed end-to-end for preprocess/load_model/predict.
#
# Good luck!
# ----------------------------------------------------------------


# Import necessary libraries here


def preprocess(df):
    # Implement any preprocessing steps required for your model here.
    # Return a Pandas DataFrame of the data
    #
    # Note: Don't drop the 'id' column here.
    # It will be used in the predict function to return the final predictions.

    return df


def load_model():
    model = None
    # ------------------ MODEL LOADING LOGIC ------------------

    # Inside this block, load your trained model.
    # --- Example ---
    # import joblib
    # model = joblib.load('model.pkl')

    # ------------------ END MODEL LOADING LOGIC ------------------
    return model


def predict(df, model):
    predictions = None
    # ------------------ PREDICTION LOGIC ------------------

    # Inside this block, generate predictions using your model.
    # This function should only contain prediction logic.
    # It must be efficient and run within the time limits.
    #
    # You must return a Pandas DataFrame with exactly two columns:
    #
    #   id,Heart Disease
    #   0,0.85
    #   1,0.23
    #   2,0.91
    #   ...
    #
    # The 'Heart Disease' column must contain probabilities (0 to 1) for the
    # 'Presence' class.
    #
    # --- Example ---
    # import pandas as pd
    # preds = model.predict_proba(df.drop(columns=['id']))[:, 1]
    # predictions = pd.DataFrame({
    #     'id': df['id'],
    #     'Heart Disease': preds
    # })

    # ------------------ END PREDICTION LOGIC ------------------
    return predictions


# ----------------------------------------------------------------
# Your code will be called in the following way:
# Note that we will not be using the function defined below.
# ----------------------------------------------------------------


def run(df) -> tuple[float, float, float]:
    from time import time

    # Time the full pipeline: preprocess -> load_model -> predict
    start = time.perf_counter()

    # Load the processed data:
    df_processed = preprocess(df)

    # Load the model:
    model = load_model()
    size = get_model_size(model)

    # Get the predictions:
    predictions = predict(
        df_processed, model
    )  # NOTE: Don't call the `preprocess` function here.

    duration = time.perf_counter() - start
    accuracy = get_model_accuracy(predictions)

    return size, accuracy, duration


# ----------------------------------------------------------------
# Helper functions you should not disturb yourself with.
# ----------------------------------------------------------------


def get_model_size(model):
    pass


def get_model_accuracy(predictions):
    pass
