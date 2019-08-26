import paddle
import paddle.fluid as fluid
import numpy as np

clip_init = np.array(
    [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
    dtype=np.float32)
clip_init = clip_init * np.reshape(
    np.array((1.0, 1.0, 1.0), dtype=np.float32), [3, 1])

v_data = clip_init
v_shape = v_data.shape
t_data = np.array([[0, 1, 2]], dtype=np.int32)
t_shape = t_data.shape

vertices = fluid.layers.data(name="vertices", shape=v_shape, dtype="float32")
triangles = fluid.layers.data(name="triangles", shape=t_shape, dtype="int32")
barycentric_coordinates, triangle_ids, z_buffer = fluid.layers.rasterize_triangles(
    vertices, triangles, image_height=480, image_width=640)

place = fluid.CPUPlace()
exe = fluid.Executor(place=place)
exe.run(fluid.default_startup_program())

fetch_list = [barycentric_coordinates.name, triangle_ids.name, z_buffer.name]

#profiler.start_profiler('All')
bc, ti, zb = exe.run(feed = {'vertices':v_data, 'triangles':t_data}, fetch_list=fetch_list)
#profiler.stop_profiler('total', '/tmp/profile')

np.save('res_paddle/bary_coor_paddle.npy', np.array(bc))
np.save('res_paddle/tri_ids_paddle.npy', np.array(ti))
np.save('res_paddle/z_b_paddle.npy', np.array(zb))
