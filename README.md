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

The emulator assumes a flat $\Lambda$-CDM cosmology and neglets the energy density of radiation and neutrinos. The cosmological parameters required in the dictonary cosmo_params to obtain the nDGP boost factors and their allowed ranges are: 
- $\Omega_{\rm m} \in [0.28,0.36]$,
- $\Omega_{\rm b} \in [0.04,0.06]$,
- $n_{\rm s} \in [0.92,1]$,
- $A_{\rm s} \in [1.7e-9,2.5e-9]$,
- $h \in [0.61,0.73]$.

Notice that the parameter $\Omega_{\rm m}$ accounts for the sum of CDM and baryonic matter and the amplitude of the primordial power spectrum $A_{\rm s}$ is defined at $k_{\rm pivot} = 0.05 \, {\rm Mpc}^{-1}$. Also the modified gravity parameter $H_0r_c$ and the redsfhit $z$ are required and their interpolation ranges are:
- $z \in [0,2]$,
- $H_0 r_c \in [0.2,20]$.

A minimal working example for the emulator is shown in the [example notebook](notebooks/example.ipynb).
