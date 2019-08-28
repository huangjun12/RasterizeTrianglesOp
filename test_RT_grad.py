import tensorflow as tf
import numpy as np

import rasterize_triangles

image_width = 4
image_height = 4

clip_init = np.array(
    [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
    dtype=np.float32)

clip_coordinates = tf.constant(clip_init)
triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

rendered_coordinates, triangle_ids, z_buffer = (
    rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
        clip_coordinates, triangles, image_width, image_height))

grd_var = tf.convert_to_tensor((1000*np.ones((4,4,3))).astype('float32'))
gradients = tf.gradients([rendered_coordinates], clip_coordinates, grad_ys = grd_var)


with tf.Session() as sess:
    image = rendered_coordinates.eval()
    triangle_ids = triangle_ids.eval()
    z_buffer = z_buffer.eval()
    grd = sess.run(gradients)
    # sess.run(rendered_coordinates)

#save bary_coor
#np.save('res/bary_coor.npy', image)
#np.save('res/tri_ids.npy', triangle_ids)
#np.save('res/zb.npy', z_buffer)