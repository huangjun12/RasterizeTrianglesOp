import tensorflow as tf
import numpy as np

import rasterize_triangles

image_width = 640
image_height = 480

clip_init = np.array(
    [[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]],
    dtype=np.float32)
clip_init = clip_init * np.reshape(
    np.array((1.0, 1.0, 1.0), dtype=np.float32), [3, 1])

clip_coordinates = tf.constant(clip_init)
triangles = tf.constant([[0, 1, 2]], dtype=tf.int32)

rendered_coordinates, triangle_ids, z_buffer = (
    rasterize_triangles.rasterize_triangles_module.rasterize_triangles(
        clip_coordinates, triangles, image_width, image_height))

#rendered_coordinates = tf.concat(
#    [rendered_coordinates,
#     tf.ones([image_height, image_width, 1])], axis=2)  #load png has extra axis with value 1

with tf.Session() as sess:
    image = rendered_coordinates.eval()
    triangle_ids = triangle_ids.eval()
    z_buffer = z_buffer.eval()
    # sess.run(rendered_coordinates)

#save bary_coor
np.save('res/bary_coor.npy', image)
np.save('res/tri_ids.npy', triangle_ids)
np.save('res/zb.npy', z_buffer)


result_image = np.clip(image, 0., 1.).copy(order='C')

# get compared image
#baseline_path = 'test_data/Simple_Triangle.png'
#baseline_bytes = open(baseline_path, 'rb').read()
#baseline_image = sess.run(tf.image.decode_png(baseline_bytes))
#baseline_image = baseline_image.astype(float) / 255.0