# Import required packages/functions --- The emulator requires sklearn installed ---
import joblib
import pickle as pk
import numpy as np
from pkg_resources import resource_stream
import os
from scipy.interpolate import InterpolatedUnivariateSpline


# Bounds of the interpolation range
input_bounds = {'H0rc':[0.2,20],
                'Om':[0.28,0.36],
                'ns':[0.92,1],
                'As':[1.7e-9,2.5e-9],
                'h':[0.61,0.73],
                'Ob':[0.04,0.06],
                'a':[1/3,1],
                'z':[0,2]}

required_params = ['Om', 'ns', 'As', 'h', 'Ob']

def rescale_param(cosmo_params,key):
    '''
    Rescale cosmological paramter in [0,1].

    Input
        cosmo_params: dict of cosmological parameters keys and corresponding values. Must contain Om, ns, As, h, Ob.
        key: str identifier of cosmological paramter amongst Om, ns, As, h, Ob.

    Output
        rescaled_value: float; normalised value of the cosmological parameter.
    '''
    value = cosmo_params[key]
    param_min, param_max = input_bounds[key]
    rescaled_value = (value-param_min)/(param_max-param_min)
    if rescaled_value<0 or rescaled_value>1:
        raise ValueError(f'Cosmological parameter ({key}={value}) outside the interpolation range: {key}_range = {input_bounds[key]}')
    return rescaled_value

class BoostPredictor:
    def __init__(self):
        print ("Loading model and related data")
        self.model = joblib.load(resource_stream('nDGPemu','/cache/nDGPemu_LC_k5_woSN_PCA3_z2.joblib'))
        self.table_mean = np.load(resource_stream('nDGPemu','/cache/TableMean.npy'), allow_pickle=True)
        self.k_vals = np.load(resource_stream('nDGPemu','/cache/k_vals.npy'), allow_pickle=True)
        with resource_stream('nDGPemu','/cache/pca.pkl') as f:
            self.pca = pk.load(f)
    
    def predict(self, H0rc, z, cosmo_params, k_out=None, ext=2):
        '''
        Computes the nDGP boost factor (P_nDGP/P_GR).

        Input
            H0rc: float; value of the nDGP parameter H_0*r_c
            z: float; redshift value.
            cosmo_params: dict of cosmological parameters keys and corresponding values. Must contain Om, ns, As, h, Ob.
                          Notice that Om is the sum of CDM and baryon densities. 
                          The cosmology is assumed to be flat LCDM, i.e. O_Lambda = 1-Om 
            k_out: 1D-array, defualt None; custom wavenumber array for which the boost factor is estimated.
                   If None, the algorithm uses the training values of the wavenumber.
            ext : int or str, optional;
                Controls the extrapolation mode for scipy.interpolate.InterpolatedUnivariateSpline;

                if ext=0 or 'extrapolate', return the extrapolated value.
                if ext=1 or 'zeros', return 0
                if ext=2 or 'raise', raise a ValueError
                if ext=3 of 'const', return the boundary value.
                    The default value is 2.
        Output
            Bk: 1D-array; boost factor values
        '''
        if z<input_bounds['z'][0] or z>input_bounds['z'][1]:
            raise ValueError(f'Redshift value (z={z}) outside the interpolation range: z_range = {input_bounds["z"]}')
        a = 1/(1+z)
        if H0rc>input_bounds['H0rc'][1] or H0rc<input_bounds['H0rc'][0]:
            raise ValueError(f'nDGP parameter (H0rc={H0rc}) outside the interpolation range: H0rc_range = {input_bounds["H0rc"]}', )
        Wrc = input_bounds['H0rc'][0]/H0rc
        if any(key not in cosmo_params for key in required_params):
            missing_keys = [key for key in required_params if key not in cosmo_params]
            raise KeyError(f'The following keys are missing from the cosmo_params dictonary: {missing_keys}.')
        cosmo_list = [rescale_param(cosmo_params,key) for key in required_params]
        input_arr = np.concatenate([[Wrc], cosmo_list,[a]]).reshape(1, -1)
        raw_predics = self.model.predict(input_arr)[0]
        predictions = 10**(-3*(self.pca.inverse_transform(raw_predics)+self.table_mean))+1
        if k_out is not None:
            if ext==2:
                if k_out.min()<self.k_vals.min() or k_out.max()>self.k_vals.max() :
                    raise ValueError(f'Requested k_value is outside of the interpolation range: k_range = {[self.k_vals.min(),self.k_vals.max()]}\n'+
                                    'Set an extrapolation rule with the "ext" keyword to avoid this warning.' )
            return InterpolatedUnivariateSpline(self.k_vals, predictions, ext=ext, k=1)(k_out)
        else:
            return predictions
    