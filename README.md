# Electricity-Load-Forecasting-using-Deep-Learning
It's a machine learning project completed for my MSc course Data Mining and Machine Learning

Electricity Load Forecasting using Deep Learning

Overview

This project develops a deep learning model, specifically using Long Short-Term Memory (LSTM) networks, to forecast electricity consumption based on historical load data for multiple clients. The goal is to accurately predict future energy demand at a 15-minute resolution.

Dataset

- Name: Electricity Load Diagrams 2011-2014
- Source: UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- Description: Contains electricity consumption data (in kW) for 370 clients from 2011 to 2014 at 15-minute intervals. The data is provided in a single .txt file (LD2011_2014.txt) with values separated by semicolons.

Methodology

The project is implemented in a Jupyter Notebook using Python and leverages several libraries including:
- pandas & numpy for data manipulation.
- scikit-learn for data preprocessing (scaling).
- pytorch for building and training the deep learning model.
- matplotlib for visualization.

Key steps include:

Data Loading & Cleaning:
Reading the semicolon-separated data, converting timestamps, handling numeric conversion, and casting to float32 for memory efficiency.

Feature Engineering:
- Creating time-based features (Hour, DayOfWeek, Month, etc.).
- Encoding cyclical time features using sine/cosine transformations.
- Generating lagged consumption features (e.g., 1h, 1d, 1w lags).
- Generating rolling window statistics (e.g., 24h, 7d rolling means).
- Note: Due to memory constraints, lagged/rolling features were generated for a configurable subset of clients.

Data Preprocessing for DL:

- Chronological splitting into Training (2011-2012), Validation (2013), and Test (2014) sets.
- Handling NaNs introduced by feature engineering (dropping from the training set).
- Scaling features using StandardScaler.
- Generating input sequences for the LSTM using a custom PyTorch Dataset to create sequences on-the-fly, avoiding large memory allocation.

Modeling:

- Defining an LSTM network architecture.
- Training the model using MSE loss, Adam optimizer (with weight decay), a learning rate scheduler (ReduceLROnPlateau), and early stopping based on validation loss.
- Evaluation: Assessing model performance on the test set using MSE, RMSE, MAE, and MAPE metrics. Visualizing predictions against actual values.

How to Run

- Dependencies: Ensure you have Python and the required libraries installed (pandas, numpy, scikit-learn, pytorch, matplotlib).
- Data: Download LD2011_2014.txt from the UCI link above and place it in the same directory as the notebook.
- Notebook: Open and run the cells sequentially in the main Jupyter Notebook (YourNotebookName.ipynb - replace with your actual notebook name).
- Configuration: Key parameters like num_clients_subset, model hyperparameters (lstm_hidden_size, dropout_rate, etc.), and training settings (num_epochs, batch_size) can be adjusted in the configuration cell near the beginning of the notebook.

Project Files

- Final_Project.ipynb: The main Jupyter Notebook containing all code.
- LD2011_2014.txt: The raw dataset file.
- best_lstm_model_subset_v2.pth: Saved state dictionary of the best trained model (for the client subset).
- README.md: This file.
