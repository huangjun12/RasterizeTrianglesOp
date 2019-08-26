
# Copyright... 

#import unittest
#from op_test import OpTest

import numpy as np
import math

def ComputeUnnormalizedMatrixInverse(a11, a12, a13, a21, a22, a23, a31, a32, a33):
    res = np.zeros((9))
    res[0] = a22 * a33 - a32 * a23
    res[1] = a13 * a32 - a33 * a12
    res[2] = a12 * a23 - a22 * a13
    res[3] = a23 * a31 - a33 * a21
    res[4] = a11 * a33 - a31 * a13
    res[5] = a13 * a21 - a23 * a11
    res[6] = a21 * a32 - a31 * a22
    res[7] = a12 * a31 - a32 * a11
    res[8] = a11 * a22 - a21 * a12

    det = a11 * res[0] + a12 * res[3] + a13 * res[6]

    if det < 0.0:
        for i in range(9):
            res[i] = -res[i]
    return res

def ComputeEdgeFunctions(px, py, m_inv):
    res = np.zeros((3))
    for i in range(3):
        a = m_inv[3 * i + 0]
        b = m_inv[3 * i + 1]
        c = m_inv[3 * i + 2]

        res[i] = a * px + b * py + c
    return res

def PixelIsInsideTriangle(value):
    return (value[0] >= 0 and value[1] >= 0 and value[2] >= 0) and (value[0] > 0 or value[1] > 0 or value[2] > 0)

def ComputeRasterizeTriangles(vertices, triangles, image_height, image_width):
    # init output
    barycentric_coordinates = np.zeros((image_height, image_width, 3), dtype=np.float32).flatten()
    triangle_ids = np.zeros((image_height, image_width), dtype=np.int32).flatten()
    z_buffer = np.ones((image_height, image_width), dtype=np.float32).flatten()

    vertices = vertices.flatten()
    triangles = triangles.flatten()
    triangle_count = triangles.shape[0] / 3
    half_image_width = 0.5 * image_width
    half_image_height = 0.5 * image_height

    for triangle_id in range(triangle_count):
        #compute for each triangle in mesh
        v0_x_id = 4 * triangles[3 * triangle_id];
        v1_x_id = 4 * triangles[3 * triangle_id + 1];
        v2_x_id = 4 * triangles[3 * triangle_id + 2];

        v0w = vertices[v0_x_id + 3]  #get the last one in each row of vertices, w from xyzw
        v1w = vertices[v1_x_id + 3] 
        v2w = vertices[v2_x_id + 3]
        
        if (v0w < 0 and v1w < 0 and v2w < 0):
            continue

        v0x = vertices[v0_x_id];
        v0y = vertices[v0_x_id + 1];
        v1x = vertices[v1_x_id];
        v1y = vertices[v1_x_id + 1];
        v2x = vertices[v2_x_id];
        v2y = vertices[v2_x_id + 1];

        unnormalized_matrix_inverse = ComputeUnnormalizedMatrixInverse(v0x, v1x, v2x, v0y, v1y, v2y, v0w, v1w, v2w);

        #Initialize the bounding box to the entire screen
        left = 0
        right = image_width
        bottom = 0
        top = image_height

        if (v0w > 0 and v1w > 0 and v2w > 0):
            p0x = (v0x / v0w + 1.0) * half_image_width
            p1x = (v1x / v1w + 1.0) * half_image_width
            p2x = (v2x / v2w + 1.0) * half_image_width
            p0y = (v0y / v0w + 1.0) * half_image_height
            p1y = (v1y / v1w + 1.0) * half_image_height
            p2y = (v2y / v2w + 1.0) * half_image_height

            left = min(max(int(math.floor(min(p0x, p1x, p2x))), 0), image_width)
            right = min(max(int(math.ceil(max(p0x, p1x, p2x))), 0), image_width)
            bottom = min(max(int(math.floor(min(p0y, p1y, p2y))), 0), image_height)
            top = min(max(int(math.ceil(max(p0y, p1y, p2y))), 0), image_height)

        #Iterate over each pixel in the bounding box.
        for iy in range(bottom, top):
            for ix in range(left, right):
                px = ((ix + 0.5) / half_image_width) - 1.0
                py = ((iy + 0.5) / half_image_height) - 1.0
                pixel_idx = int(iy * half_image_width + ix)

                b_over_w = ComputeEdgeFunctions(px, py, unnormalized_matrix_inverse)

                if not PixelIsInsideTriangle(b_over_w):
                    continue

                bs_over_w = b_over_w[0] + b_over_w[1] + b_over_w[2]
                b0 = b_over_w[0] / bs_over_w
                b1 = b_over_w[1] / bs_over_w
                b2 = b_over_w[2] / bs_over_w

                v0z = vertices[v0_x_id + 2]
                v1z = vertices[v1_x_id + 2]
                v2z = vertices[v2_x_id + 2]

                clip_z = b0 * v0z + b1 * v1z + b2 * v2z
                clip_w = b0 * v0w + b1 * v1w + b2 * v2w
                z = clip_z / clip_w
                if z < -1.0 or z > 1.0 or z > z_buffer[pixel_idx]:
                    continue

                triangle_ids[pixel_idx] = triangle_id
                z_buffer[pixel_idx] = z
                barycentric_coordinates[3 * pixel_idx + 0] = b0
                barycentric_coordinates[3 * pixel_idx + 1] = b1
                barycentric_coordinates[3 * pixel_idx + 2] = b2
    print('barycentric_coordinates', barycentric_coordinates)
    print('triangle_ids', triangle_ids)
    print('z_buffer', z_buffer)
    return barycentric_coordinates, triangle_ids, z_buffer

class TestRasterizeTrianglesOp():  #OpTest
    def setUp(self):
        self.op_type = "rasterize_triangles"
        vertices = np.array([[-0.5, -0.5, 0.8, 1.0], [0.0, 0.5, 0.3, 1.0], [0.5, -0.5, 0.3, 1.0]], dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype = np.int32)
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
    c = TestRasterizeTrianglesOp()
    c.setUp()