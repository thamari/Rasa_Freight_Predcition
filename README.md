# Predict Freight Value Chatbot

This project is a Rasa-based chatbot designed to predict the freight value of a product based on its dimensions and weight.

## Project Overview

This chatbot interacts with users to gather information about a product's weight, length, height, and width. It then uses a pre-trained machine learning model to predict the freight value based on these inputs.

## Features

- Interactive form for collecting product details.
- Custom action to predict freight value using a machine learning model.
- Handles user inputs and validates them.

## Setup

### Prerequisites

- Anaconda or Miniconda installed on your system.
- Rasa 3.1 or later.
- Python 3.10.14.

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo.git
    cd your-repo
    ```

2. **Create and activate a new Anaconda environment:**

    ```bash
    conda create --name rasa_env python=3.10.14
    conda activate rasa_env
    ```

3. **Install the required packages:**

    ```bash
    pip install rasa==3.1.0
    pip install joblib numpy pandas
    ```

4. **Install Rasa SDK:**

    ```bash
    pip install rasa-sdk
    ```

### Data Files

Ensure that the following files are in the `actions` directory:

- `scaler.pkl`
- `best_xgb_model.pkl`
- `freight_data.csv`

## Usage

1. **Train the Rasa model:**

    ```bash
    rasa train
    ```

2. **Run the Rasa actions server:**

    ```bash
    rasa run actions
    ```

3. **Run the Rasa server:**

    ```bash
    rasa shell
    ```

4. **Interact with the chatbot:**

    Start a conversation and provide the required product details when prompted.

## Project Structure

├── actions
│ ├── scaler.pkl
│ ├── best_xgb_model.pkl
│ ├── freight_data.csv
│ ├── actions.py
├── data
│ ├── nlu.yml
│ ├── rules.yml
│ ├── stories.yml
├── models
├── config.yml
├── credentials.yml
├── domain.yml
├── endpoints.yml
├── README.md

markdown
Copy code

- `actions/`: Contains the custom action code and model files.
- `data/`: Contains the NLU, stories, and rules data.
- `models/`: Stores the trained models.
- `config.yml`: Configuration for Rasa NLU and core.
- `credentials.yml`: Configuration for Rasa channels.
- `domain.yml`: Domain file defining intents, entities, slots, and actions.
- `endpoints.yml`: Configuration for action server.

## Customization

To customize the chatbot:

1. **Modify NLU data** in `data/nlu.yml` to add or change intents and training examples.
2. **Update stories** in `data/stories.yml` to define new conversation flows.
3. **Adjust rules** in `data/rules.yml` to change how the bot responds to different intents.
4. **Edit domain** in `domain.yml` to add new entities, slots, or responses.
5. **Modify custom actions** in `actions/actions.py` to change the logic of the freight value prediction.

## Troubleshooting

- Ensure that all required files (`scaler.pkl`, `best_xgb_model.pkl`, and `freight_data.csv`) are in the correct directory (`actions/`).
- Check the Rasa logs for any errors or warnings.
- Verify that the Anaconda environment has all the necessary packages installed.

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.
