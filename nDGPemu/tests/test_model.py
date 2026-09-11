import os
import numpy as np

# The cache directory sits alongside the tests/ folder inside the package.
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')

def test_predict(model):
    # set reference parameters
    cosmo_params = {'Om':0.3089,
                'ns':0.9667,
                'As':2.066e-9,
                'h':0.6774,
                'Ob':0.0486}
    H0rc = 1
    z = 1

    Bk = model.predict(H0rc,z,cosmo_params)
    Bk_ref = np.load(os.path.join(CACHE_DIR, 'Test_Bk.npy'), allow_pickle=True)

    assert all(abs(Bk-Bk_ref)<1e-7) , f"Test failed: the model could not reproduce the reference boost factor."

    print("All tests passed successfully.")