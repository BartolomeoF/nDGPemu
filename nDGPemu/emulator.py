# Import required packages/functions --- The emulator requires sklearn installed ---
import joblib
import pickle as pk
import numpy as np
import pathlib
import os
from scipy.interpolate import InterpolatedUnivariateSpline

package_path = os.fsdecode(pathlib.Path(os.path.dirname(__file__)).parent.absolute())

# Bounds of the interpolation range
input_bounds = {'H0rc':[0.2,20],
                'Om':[0.28,0.36],
                'ns':[0.92,1],
                'As':[1.7e-9,2.5e-9],
                'h':[0.61,0.73],
                'Ob':[0.04,0.06],
                'a':[1/3,1]}

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
        raise UserWarning(f'Cosmological parameter outside the interpolation range: {key} = {value}')
    return rescaled_value

class BoostPredictor:
    def __init__(self):
        print ("Loading model and related data")
        self.model = joblib.load(f'{package_path}/cache/nDGPemu_LC_k5_woSN_PCA3_z2.joblib')
        self.table_mean = np.load(f'{package_path}/cache/TableMean.npy', allow_pickle=True)
        self.k_vals = np.load(f'{package_path}/cache/k_vals.npy', allow_pickle=True)
        with open(f'{package_path}/cache/pca.pkl','rb') as f:
            self.pca = pk.load(f)
    
    def predict(self, H0rc, z, cosmo_params, k_out=None):
        '''
        Computes the nDGP boost factor (P_nDGP/P_GR).

        Input
            H0rc: float; value of the nDGP parameter H_0*r_c
            z: float; redshift value.
            cosmo_params: dict of cosmological parameters keys and corresponding values. Must contain Om, ns, As, h, Ob.

        Output
            k: 1D-array; wavenumber correspoding to the boost factor values
            Bk: 1D-array; boost factor values
        '''
        a = 1/(1+z)
        if a<input_bounds['a'][0] or a>input_bounds['a'][1]:
            raise UserWarning(f'Redshift value outside the interpolation range: z = {input_bounds["z"]}')
        if H0rc>input_bounds['H0rc'][1] or H0rc<input_bounds['H0rc'][0]:
            raise UserWarning(f'nDGP parameter outside the interpolation range: H0rc = {input_bounds["H0rc"]}', )
        Wrc = input_bounds['H0rc'][0]/H0rc
        cosmo_list = [rescale_param(cosmo_params,key) for key in ['Om', 'ns', 'As', 'h', 'Ob']]
        input_arr = np.concatenate([[Wrc], cosmo_list,[a]]).reshape(1, -1)
        raw_predics = self.model.predict(input_arr)[0]
        predictions = 10**(-3*(self.pca.inverse_transform(raw_predics)+self.table_mean))+1
        if k_out is not None:
            return InterpolatedUnivariateSpline(self.k_vals, predictions, ext=2)(k_out)
        else:
            return predictions
    