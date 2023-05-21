import os
import time
import numpy as np
from sonnmf.main import sonnmf
from sonnmf.numba.main import sonnmf as sonnmf_numba
data_filepath = '../datasets/jasper_small_6.npz'
ini_filepath = '../saved_models/jasper_small_6/r{}_ini.npz'
save_filepath = '../saved_models/jasper_small_6/r{}_l{}_g{}_it{}.npz'
image_filepath = '../images/jasper_small_6/r{}_l{}_g{}_it{}.jpg'
M = np.load(data_filepath)['X']
M = M.astype(np.float64)
m, n = M.shape
r = 20


if os.path.exists(ini_filepath.format(r)):
    data = np.load(ini_filepath.format(r))
    ini_W = data['ini_W']
    ini_H = data['ini_H']
else:
    ini_W = np.random.rand(m, r)
    ini_H = np.random.rand(r, n)
    with open(ini_filepath.format(r), 'wb') as fout:
        np.savez_compressed(fout, ini_W=ini_W, ini_H=ini_H)

st = time.time()
W, H, fscores, gscores, hscores, total_scores = sonnmf(M, ini_W.copy(), ini_H.copy(), lam=1, gamma=1, itermax=100, 
                                                       W_update_iters=10, early_stop=False, verbose=True)
print(1, time.time() - st)


W, H, fscores, gscores, hscores, total_scores = sonnmf_numba(M, ini_W.copy(), ini_H.copy(), lam=1, gamma=1, itermax=1, 
                                                       W_update_iters=1, early_stop=False, verbose=True)

st = time.time()
W, H, fscores, gscores, hscores, total_scores = sonnmf_numba(M, ini_W.copy(), ini_H.copy(), lam=1, gamma=1, itermax=100, 
                                                       W_update_iters=10, early_stop=False, verbose=True)
print(2, time.time() - st)