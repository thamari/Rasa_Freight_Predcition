import joblib
import numpy as np
import pandas as pd
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionPredictFreightValue(Action):
    def name(self) -> Text:
        return "predict_freight_value"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Load the model and data
        scaler = joblib.load('actions/scaler.pkl')
        model = joblib.load('actions/best_xgb_model.pkl')
        freight_data = pd.read_csv('actions/freight_data.csv')

        # Get slot values
        weight_g = tracker.get_slot("product_weight_g")
        length = tracker.get_slot("product_length_cm")
        height = tracker.get_slot("product_height_cm")
        width = tracker.get_slot("product_width_cm")

        # Ensure slots are not None and convert to float
        try:
            weight_g = float(weight_g)
            length = float(length)
            height = float(height)
            width = float(width)
        except (TypeError, ValueError) as e:
            dispatcher.utter_message(text="Please provide valid numerical values for weight, length, height, and width.")
            return []

        # Cap outliers
        weight_g = self.cap_outliers_iqr(weight_g, 'product_weight_g', freight_data)
        length = self.cap_outliers_iqr(length, 'product_length_cm', freight_data)
        height = self.cap_outliers_iqr(height, 'product_height_cm', freight_data)
        width = self.cap_outliers_iqr(width, 'product_width_cm', freight_data)

        # Feature engineering
        weight_log = np.log1p(weight_g)
        length_log = np.log1p(length)
        height_log = np.log1p(height)
        width_log = np.log1p(width)
        volume = length_log * height_log * width_log
        density = weight_log / volume

        # Create a dataframe for the new data
        new_data = pd.DataFrame({
            'product_weight_g': [weight_log],
            'product_length_cm': [length_log],
            'product_height_cm': [height_log],
            'product_width_cm': [width_log],
            'volume': [volume],
            'density': [density]
        })

        # Scale the new data
        new_data_scaled = scaler.transform(new_data)

        # Predict the freight value
        freight_value_log = model.predict(new_data_scaled)
        freight_value = np.expm1(freight_value_log)[0]

        dispatcher.utter_message(text=f"The predicted freight value is: {freight_value:.2f} units.")
        return []

    def cap_outliers_iqr(self, value, column, data):
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        if value < lower_bound:
            return lower_bound
        elif value > upper_bound:
            return upper_bound
        else:
            return value