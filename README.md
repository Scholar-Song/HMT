# HMT: Hybrid mechanistic Transformer for bio-fabrication prediction under complex environmental conditions

This framework is designed to facilitate the experimentation and comparison of various time series forecasting models.

## Directory Structure

- `config`: Contains configuration files such as `config.json` that can be used to specify model parameters and training settings.
- `experiment`: 
  - `fusion-exp`: Experiments related to feature fusion in time series forecasting.
  - `mech-exp`: Experiments incorporating mechanisms into time series forecasting models.
- `main`: Entry points for baseline time series prediction experiments with different models.
- `model`: Implementation of all models utilized in the experiments, including but not limited to LSTM, GRU, Transformer, etc.
- `utils`: Helper scripts for data preprocessing, training, and testing of models.

## Usage
To run experiments for time series forecasting:

1. Configure the model parameters and training settings in the config.json.
2. Navigate to the main directory and execute the desired model script, such as python GRUmain.py for running GRU based forecasting.
3. To conduct experiments related to feature fusion, access the fusion-exp folder and run the corresponding scripts.
4. For mechanism-based experiments, explore the mech-exp directory.
5. The utils directory contains essential utilities like data loaders and preprocessing scripts that can be utilized in your experiments.

##Preprocessing
Data preprocessing steps including scaling, splitting, and transforming are handled by the utilities in the utils directory. Make sure to preprocess your data before running any experiments to ensure the models perform optimally.

##Dependencies
Ensure all dependencies are installed by running pip install -r requirements.txt (this file should be created and maintained with all necessary packages).

For any issues or further information, please refer to the documentation provided within each script or reach out through the Issues section on this repository's page.
