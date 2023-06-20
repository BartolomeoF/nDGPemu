# nDGPemu
Emulator of the boost factor for the dark matter power spectrum in nDGP gravity models

# Installation
It is recommended to install the package in a dedicated environment. The package requires:
- numpy,
- joblib,
- scipy,
- scikit-learn.

To install the package, access the package directory from a terminal window and execute:

    pip install .

# Using the emulator
The emulator model can be imported in python with

    from nDGPemu import BoostPredictor

Instantiating a BoostPredictor object loads the MLP model and the auxiliary data

    model = BoostPredictor()

The model can be used to predict the boost factor with the syntax

    Bk = model.predict(H0rc, z, cosmo_params)

where H0rc, z, cosmo_params are user defined paramters.

A minimal working example for the emulator is shown in the [example notebook](notebooks/example.ipynb).
