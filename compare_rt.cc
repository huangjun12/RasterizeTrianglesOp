int main() {
    const int image_height = 3;
    const int image_width = 3;

    const float* vertices = {{-0.5, -0.5, 0.8, 1.0}, {0.0, 0.5, 0.3, 1.0}, {0.5, -0.5, 0.3, 1.0}};
    const int* triangles = {{0, 1, 2}};
    const int triangle_count = 1;
    int* triangle_ids[image_height][image_width][3];
    float* barycentric_coordinates[image_height][image_width] = {0};
    float* z_buffer[image_height][image_width];

    for (i=0; i < image_height; ++i) {
        for (j=0; j < image_width; ++j) {
            z_buffer[i][j] = 1.0
        }
    }

    // Start computing
    const float half_image_width = 0.5 * image_width;
    const float half_image_height = 0.5 * image_height;
    float unnormalized_matrix_inverse[9];
    float edge_w[3];

    for (int triangle_id = 0; triangle_id < triangle_count; ++triangle_id) {
        const int vx0_id = 4 * triangles[3 * triangle_id];
        const int vx1_id = 4 * triangles[3 * triangle_id + 1];
        const int vx2_id = 4 * triangles[3 * triangle_id + 2];

        const float vw0 = vertices[vx0_id + 3];
        const float vw1 = vertices[vx1_id + 3];
        const float vw2 = vertices[vx2_id + 3];
        // if all w < 0, triangle is entirely behind the eye.
        if (vw0 < 0 && vw1 < 0 && vw2 < 0) {
            continue;
        }

        const float vx0 = vertices[vx0_id];
        const float vy0 = vertices[vx0_id + 1];
        const float vx1 = vertices[vx1_id];
        const float vy1 = vertices[vx1_id + 1];
        const float vx2 = vertices[vx2_id];
        const float vy2 = vertices[vx2_id + 1];

        ComputeUnnormalizedMatrixInverse(vx0, vx1, vx2, vy0, vy1, vy2, vw0, vw1,
                                         vw2, unnormalized_matrix_inverse);

        // Project the vertices to pixel coordinates
        int left = 0, right = image_width, bottom = 0, top = image_height;
        if (vw0 > 0 && vw1 > 0 && vw2 > 0) {
            const float pixel_x0 = (vx0 / vw0 + 1.0) * half_image_width;
            const float pixel_x1 = (vx1 / vw1 + 1.0) * half_image_width;
            const float pixel_x2 = (vx2 / vw2 + 1.0) * half_image_width;
            const float pixel_y0 = (vy0 / vw0 + 1.0) * half_image_height;
            const float pixel_y1 = (vy1 / vw1 + 1.0) * half_image_height;
            const float pixel_y2 = (vy2 / vw2 + 1.0) * half_image_height;
            left = ClampedIntegerMin(pixel_x0, pixel_x1, pixel_x2, 0, image_width);
            right = ClampedIntegerMax(pixel_x0, pixel_x1, pixel_x2, 0, image_width);
            bottom = ClampedIntegerMin(pixel_y0, pixel_y1, pixel_y2, 0, image_height);
            top = ClampedIntegerMax(pixel_y0, pixel_y1, pixel_y2, 0, image_height);
        }

        // Iterate over each pixel in the bounding box.
        for (int iy = bottom; iy < top; ++iy) {
            for (int ix = left; ix < right; ++ix) {
                const float pixel_x = ((ix + 0.5) / half_image_width) - 1.0;
                const float pixel_y = ((iy + 0.5) / half_image_height) - 1.0;
                const int pixel_idx = iy * image_width + ix;

                ComputeEdgeFunctions(pixel_x, pixel_y, unnormalized_matrix_inverse, edge_w);
                if (!PixelIsInsideTriangle(edge_w)) {
                    continue;
                }

                const float edge_w_sum = edge_w[0] + edge_w[1] + edge_w[2];
                const float bc0 = edge_w[0] / edge_w_sum;
                const float bc1 = edge_w[1] / edge_w_sum;
                const float bc2 = edge_w[2] / edge_w_sum;

                const float vz0 = vertices[vx0_id + 2];
                const float vz1 = vertices[vx1_id + 2];
                const float vz2 = vertices[vx2_id + 2];
                // Recompute a properly scaled clip-space w value and then divide clip-space z by that.
                const float clip_z = bc0 * vz0 + bc1 * vz1 + bc2 * vz2;
                const float clip_w = bc0 * vw0 + bc1 * vw1 + bc2 * vw2;
                const float z = clip_z / clip_w;

                // Skip the pixel which is farther than the current z-buffer pixel or beyond the near or far clipping plane.
                if (z < -1.0 || z > 1.0 || z > z_buffer[pixel_idx]) {
                    continue;
                }

                triangle_ids[pixel_idx] = triangle_id;
                z_buffer[pixel_idx] = z;
                barycentric_coordinates[3 * pixel_idx + 0] = bc0;
                barycentric_coordinates[3 * pixel_idx + 1] = bc1;
                barycentric_coordinates[3 * pixel_idx + 2] = bc2;
                std::cout<<barycentric_coordinates<<std::endl;
                std::cout<<triangle_ids<<std::endl;
                std::cout<<z_buffer<<std::endl;
            }
        }
    }

    return 0;
}

