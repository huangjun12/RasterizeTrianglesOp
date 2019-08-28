import paddle
import paddle.fluid as fluid
import numpy as np

v_data = np.array(
    [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
    dtype=np.float32)

print('v_shape', v_data.shape)
t_data = np.array([[0, 1, 2]], dtype=np.int32)
print('t_shape', t_data.shape)

vertices = fluid.layers.data(name="vertices", shape=[-1, 4], dtype="float32")
triangles = fluid.layers.data(name="triangles", shape=[-1, 3], dtype="int32")
barycentric_coordinates, triangle_ids, z_buffer = fluid.layers.rasterize_triangles(
    vertices, triangles, image_height=4, image_width=4)

#gradient
#grd_value = np.random.random((4, 4, 3)).astype('float32')
grd_value = 1000 * np.ones((4,4,3)).astype('float32')
grd_var = fluid.layers.data(name='grd_var', shape=(4, 4, 3), append_batch_size=False, dtype='float32')
vertices.stop_gradient = False
grd = fluid.gradients([barycentric_coordinates], vertices, target_gradients= grd_var)


place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

fetch_list = [barycentric_coordinates.name, triangle_ids.name, z_buffer.name, grd[0].name]

#profiler.start_profiler('All')
bc, ti, zb, grd_out = exe.run(feed = {'vertices':v_data, 'triangles':t_data}, fetch_list=fetch_list)
#profiler.stop_profiler('total', '/tmp/profile')

#np.save('res_paddle/bary_coor_paddle.npy', np.array(bc))
#np.save('res_paddle/tri_ids_paddle.npy', np.array(ti))
#np.save('res_paddle/z_b_paddle.npy', np.array(zb))
print('grd_out:', np.array(grd_out))