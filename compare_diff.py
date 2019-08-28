import numpy as np
import os

root_paddle = 'res_paddle'
root_tf = 'res_tf'

#paddle
bc_paddle = np.load(os.path.join(root_paddle, 'bary_coor_paddle.npy'))
ti_paddle = np.load(os.path.join(root_paddle, 'tri_ids_paddle.npy'))
zb_paddle = np.load(os.path.join(root_paddle, 'z_b_paddle.npy'))

#tf
bc_tf = np.load(os.path.join(root_tf, 'bary_coor.npy'))
ti_tf = np.load(os.path.join(root_tf, 'tri_ids.npy'))
zb_tf = np.load(os.path.join(root_tf, 'zb.npy'))

#max_diff
max_bc_diff = max(bc_paddle - bc_tf)
max_ti_diff = max(ti_paddle - ti_tf)
max_zb_diff = max(zb_paddle - zb_tf)

print('max_bc_diff', max_bc_diff)
print('max_ti_diff', max_ti_diff)
print('max_zb_diff', max_zb_diff)