## Big Data Analysis Project 
Goal: Sleep stage classification using PPG Signals 

Run `main.py` to train and validate on 5 folds MESA 100 participants. 
- PPG Data is normalized to min max scale and resampled at 25 Hz. 
- Baseline models are saved in saved_model with default arguments. 
- Do not change the seed. 
- Full version of dataset can be downloaded at MESA(https://sleepdata.org/datasets/mesa) - Permission required
- Baseline model is an EEG-based sleep staging model called TinySleepNet: An Efficient Deep Learning Model for Sleep Stage Scoring based on Raw Single-Channel EEG by Akara Supratak and Yike Guo 