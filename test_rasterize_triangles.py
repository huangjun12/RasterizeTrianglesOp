import unittest
from op_test import OpTest

import numpy as np
import math

def ComputeUnnormalizedMatrixInverse(m11, m12, m13, m21, m22, m23, m31, m32, m33):
    res = np.zeros((9))
    res[0] = m22 * m33 - m32 * m23
    res[1] = m13 * m32 - m33 * m12
    res[2] = m12 * m23 - m22 * m13
    res[3] = m23 * m31 - m33 * m21
    res[4] = m11 * m33 - m31 * m13
    res[5] = m13 * m21 - m23 * m11
    res[6] = m21 * m32 - m31 * m22
    res[7] = m12 * m31 - m32 * m11
    res[8] = m11 * m22 - m21 * m12

    det = m11 * res[0] + m12 * res[3] + m13 * res[6]

    if det < 0.0:
        for i in range(9):
            res[i] = -res[i]
    return res

def ComputeEdgeFunctions(pixel_x, pixel_y, m_inv):
    res = np.zeros((3))
    for i in range(3):
        a = m_inv[3 * i + 0]
        b = m_inv[3 * i + 1]
        c = m_inv[3 * i + 2]

        res[i] = a * pixel_x + b * pixel_y + c
    return res

def PixelIsInsideTriangle(value):
    return (value[0] >= 0 and value[1] >= 0 and value[2] >= 0) and (value[0] > 0 or value[1] > 0 or value[2] > 0)

def ComputeRasterizeTriangles(vertices, triangles, image_height, image_width):
    barycentric_coordinates = np.zeros((image_height, image_width, 3), dtype=np.float32).flatten()
    triangle_ids = np.zeros((image_height, image_width), dtype=np.int32).flatten()
    z_buffer = np.ones((image_height, image_width), dtype=np.float32).flatten()

    vertices = vertices.flatten('F')
    triangles = triangles.flatten('F')
    triangle_count = triangles.shape[0] / 3
    half_image_width = 0.5 * image_width
    half_image_height = 0.5 * image_height

    for triangle_id in range(triangle_count):
        vx0_id = 4 * triangles[3 * triangle_id];
        vx1_id = 4 * triangles[3 * triangle_id + 1];
        vx2_id = 4 * triangles[3 * triangle_id + 2];

        vw0 = vertices[vx0_id + 3]
        vw1 = vertices[vx1_id + 3] 
        vw2 = vertices[vx2_id + 3]
        
        if (vw0 < 0 and vw1 < 0 and vw2 < 0):
            continue

        vx0 = vertices[vx0_id];
        vy0 = vertices[vx0_id + 1];
        vx1 = vertices[vx1_id];
        vy1 = vertices[vx1_id + 1];
        vx2 = vertices[vx2_id];
        vy2 = vertices[vx2_id + 1];

        unnormalized_matrix_inverse = ComputeUnnormalizedMatrixInverse(vx0, vx1, vx2, vy0, vy1, vy2, vw0, vw1, vw2);

        #Initialize the bounding box to the entire screen
        left = 0
        right = image_width
        bottom = 0
        top = image_height

        if (vw0 > 0 and vw1 > 0 and vw2 > 0):
            pixel_x0 = (vx0 / vw0 + 1.0) * half_image_width
            pixel_x1 = (vx1 / vw1 + 1.0) * half_image_width
            pixel_x2 = (vx2 / vw2 + 1.0) * half_image_width
            pixel_y0 = (vy0 / vw0 + 1.0) * half_image_height
            pixel_y1 = (vy1 / vw1 + 1.0) * half_image_height
            pixel_y2 = (vy2 / vw2 + 1.0) * half_image_height

            left = min(max(int(math.floor(min(pixel_x0, pixel_x1, pixel_x2))), 0), image_width)
            right = min(max(int(math.ceil(max(pixel_x0, pixel_x1, pixel_x2))), 0), image_width)
            bottom = min(max(int(math.floor(min(pixel_y0, pixel_y1, pixel_y2))), 0), image_height)
            top = min(max(int(math.ceil(max(pixel_y0, pixel_y1, pixel_y2))), 0), image_height)

        #Iterate over each pixel in the bounding box.
        for iy in range(bottom, top):
            for ix in range(left, right):
                pixel_x = ((ix + 0.5) / half_image_width) - 1.0
                pixel_y = ((iy + 0.5) / half_image_height) - 1.0
                pixel_idx = iy * image_width + ix

                edge_w = ComputeEdgeFunctions(pixel_x, pixel_y, unnormalized_matrix_inverse)

                if not PixelIsInsideTriangle(edge_w):
                    continue

                edge_w_sum = edge_w[0] + edge_w[1] + edge_w[2]
                bc0 = edge_w[0] / edge_w_sum
                bc1 = edge_w[1] / edge_w_sum
                bc2 = edge_w[2] / edge_w_sum

                vz0 = vertices[vx0_id + 2]
                vz1 = vertices[vx1_id + 2]
                vz2 = vertices[vx2_id + 2]

                clip_z = bc0 * vz0 + bc1 * vz1 + bc2 * vz2
                clip_w = bc0 * vw0 + bc1 * vw1 + bc2 * vw2
                z = clip_z / clip_w
                if z < -1.0 or z > 1.0 or z > z_buffer[pixel_idx]:
                    continue

                triangle_ids[pixel_idx] = triangle_id
                z_buffer[pixel_idx] = z
                barycentric_coordinates[3 * pixel_idx + 0] = bc0
                barycentric_coordinates[3 * pixel_idx + 1] = bc1
                barycentric_coordinates[3 * pixel_idx + 2] = bc2

    barycentric_coordinates = barycentric_coordinates.reshape(image_height, image_width, 3)
    triangle_ids = triangle_ids.reshape(image_height, image_width)
    z_buffer = z_buffer.reshape(image_height, image_width)
    return barycentric_coordinates, triangle_ids, z_buffer

class TestRasterizeTrianglesOp(OpTest):
    def setUp(self):
        self.op_type = "rasterize_triangles"
        num_triangles = 20
        v_data = 2 * np.random.rand(12 * num_triangles) - 1
        vertices = v_data.reshape(3 * num_triangles, 4).astype('float32')
        t_data = np.array([i for i in range(3 * num_triangles)])
        triangles = t_data.reshape(num_triangles, 3).astype('int32')

        image_height = 480
        image_width = 640
        barycentric_coordinates, triangle_ids, z_buffer = ComputeRasterizeTriangles(vertices, triangles, image_height, image_width)
        self.inputs = {
            'Vertices': vertices,
            'Triangles': triangles}
        self.attrs = {'image_height': image_height,
                      'image_width': image_width}
        self.outputs = {'BarycentricCoordinates': barycentric_coordinates,
                        'TriangleIds': triangle_ids,
                        'ZBuffer': z_buffer}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['Vertices'], 'BarycentricCoordinates', max_relative_error=0.005)

if __name__ == '__main__':
    unittest.main()