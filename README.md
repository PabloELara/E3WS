# E3WS: Earthquake Early Warning starting from 3 seconds of records on a single station with machine learning

Welcome to the E3WS. Do you have a station? Start monitoring earthquakes!

If you use this software, you should cite the code as follow:   

**E3WS, Pablo Lara et al. 2023** [![DOI](https://zenodo.org/badge/637827897.svg)](https://zenodo.org/badge/latestdoi/637827897)

And the paper:   

**Pablo Lara, Quentin Bletery, Jean-Paul Ampuero, Adolfo Inza, Hernando Tavera. Earthquake Early Warning starting from 3 seconds of records on a single station with machine learning. Journal of Geophysical Research: Solid Earth. 2023.**

E3WS dependences:
- `python = 3.9`
- `xgboost = 1.6.1`
- `scipy = 1.8.1`
- `python-speech-features = 0.6`
- `scikit-learn = 1.1.1`
- `PyGeodesy = 23.3.23`
- `obspy =1.3.0`

## E3WS models

E3WS consists of 3 stages: detection, P-phase picking and source characterization.

For P-phase picking and source characterization (magnitude and location) the models are already defined. You can find them in the models folder. For example MAG7tp3*.joblib, it means the magnitude model uses 7 seconds before P-phase and 3 seconds after.

For detection, you must create your own model with intrinsic noises of the station to be installed. Relax, it is not difficult.

Inside the DET/build_DET/ folder:
1. Have 10 days (or more) of continuous data to extract the noise (we must reach 900000 samples) and add it to the 'data/' folder.
2. Eliminate the seismic records in these 10 days. I made an automatic program to remove it ('gen_catag.py'), feel free to make it visual and adapt it with the output format in the 'picked/' folder.
3. Generate the feature vector with the program 'pb_FV_noise.py', it will create a csv file in the 'atr_noise/' folder.
4. Download the earthquake feature vector folder 'atr_eq/' at https://mega.nz/folder/E2UTXIQZ#rH_k9nNrUIgU3D04rzPzzQ
5. Finally, we have our noise ('atr_noise') and earthquake ('atr_eq') attributes. Now we have to train our detector model! Adapt the file pb_save_DET_model.py, and it will create our model in 'saved_models/'.

## Run E3WS
Time to monitor earthquakes.
An example of how E3WS works is in the 'real_time/' folder. E3WS detected the M5.6 earthquake of January 7, 2022 in Lima, Peru. 

The first E3WS estimate was M5.33 for 3 seconds P-wave. Continuous updates converged to M5.6 and are captured in the results/ folder.
