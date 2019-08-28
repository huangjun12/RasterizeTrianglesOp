/* Copixel_yright (c) ... */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/eigen.h"

namespace {
// Threshold for a barycentric coordinate triplet's sum.
constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

// The minimum area of a triangle in screen space.
constexpr float kMinimumTriangleArea = 1e-13;

}  //namespace


namespace paddle {
namespace operators{

using Tensor = framework::Tensor;

namespace {

// Takes the minimum, rounds down, and converts to an integer in the range [low, high].
inline int ClampedIntegerMin(float a, float b, float c, int low, int high) {
    return std::min(
            std::max(static_cast<int>(std::floor(std::min(std::min(a, b), c))),
                     low), high);
}

// Takes the maximum, rounds up, and converts to an integer in the range [low, high].
inline int ClampedIntegerMax(float a, float b, float c, int low, int high) {
    return std::min(
            std::max(static_cast<int>(std::ceil(std::max(std::max(a, b), c))), low),
            high);
}

// Computes a 3x3 matrix inverse (http://mathworld.wolfram.com/MatrixInverse.html).
void ComputeUnnormalizedMatrixInverse(const float m11, const float m12,
                                      const float m13, const float m21,
                                      const float m22, const float m23,
                                      const float m31, const float m32,
                                      const float m33, float m_inv[9]) {
    m_inv[0] = m22 * m33 - m32 * m23;
    m_inv[1] = m13 * m32 - m33 * m12;
    m_inv[2] = m12 * m23 - m22 * m13;
    m_inv[3] = m23 * m31 - m33 * m21;
    m_inv[4] = m11 * m33 - m31 * m13;
    m_inv[5] = m13 * m21 - m23 * m11;
    m_inv[6] = m21 * m32 - m31 * m22;
    m_inv[7] = m12 * m31 - m32 * m11;
    m_inv[8] = m11 * m22 - m21 * m12;

    const float det = a11 * m_inv[0] + a12 * m_inv[3] + a13 * m_inv[6];

    if (det < 0.0f) {
        for (int i = 0; i < 9; ++i) {
            m_inv[i] = -m_inv[i];
        }
    }
}

// Computes the edge functions from M^-1 as described by Olano and Greer,
// "Triangle Scan Conversion using 2D Homogeneous Coordinates."
void ComputeEdgeFunctions(const float pixel_x, const float pixel_y, const float m_inv[9],
                          float values[3]) {
    for (int i = 0; i < 3; ++i) {
        const float a = m_inv[3 * i + 0];
        const float b = m_inv[3 * i + 1];
        const float c = m_inv[3 * i + 2];

        values[i] = a * pixel_x + b * pixel_y + c;
    }
}

// Determines whether the point p lies inside a front-facing triangle.
bool PixelIsInsideTriangle(const float edge_values[3]) {
    return (edge_values[0] >= 0 && edge_values[1] >= 0 && edge_values[2] >= 0) &&
           (edge_values[0] > 0 || edge_values[1] > 0 || edge_values[2] > 0);
}

} //namespace

template <typename DeviceContext, typename T>
class RasterizeTrianglesKernel : public framework::OpKernel<T> {
public:
    void Compute(const framework::ExecutionContext& context) const override {
        const Tensor* vertices_tensor = context.Input<Tensor>("Vertices");  //input must stable
        const Tensor* triangles_tensor = context.Input<Tensor>("Triangles");
        Tensor* barycentric_coordinates_tensor = context.Output<Tensor>("BarycentricCoordinates");
        Tensor* triangle_ids_tensor = context.Output<Tensor>("TriangleIds");
        Tensor* z_buffer_tensor = context.Output<Tensor>("ZBuffer");

        barycentric_coordinates_tensor->mutable_data<float>(context.GetPlace());
        triangle_ids_tensor->mutable_data<int>(context.GetPlace());
        z_buffer_tensor->mutable_data<float>(context.GetPlace());

        const int image_height = context.Attr<int>("image_height");
        const int image_width = context.Attr<int>("image_width");

        auto vertices_eigen = framework::EigenVector<float>::Flatten(*vertices_tensor);
        auto triangles_eigen = framework::EigenVector<int>::Flatten(*triangles_tensor);

        auto bc_eigen = framework::EigenVector<float>::Flatten(*barycentric_coordinates_tensor).setZero();
        auto ti_eigen = framework::EigenVector<int>::Flatten(*triangle_ids_tensor).setZero();
        auto zb_eigen = framework::EigenVector<float>::Flatten(*z_buffer_tensor).setConstant(1.0f);

        const float* vertices = vertices_eigen.data();
        const int* triangles = triangles_eigen.data();
        const int triangle_count = triangles_eigen.size() / 3;
        int* triangle_ids = ti_eigen.data();
        float* barycentric_coordinates = bc_eigen.data();
        float* z_buffer = zb_eigen.data();

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
                }
            }
        }
    }

};

template <typename DeviceContext, typename T>
class RasterizeTrianglesGradKernel : public framework::OpKernel<T> {
public:
    void Compute(const framework::ExecutionContext& context) const override {
        const Tensor* vertices_tensor = context.Input<Tensor>("Vertices");
        const Tensor* triangles_tensor = context.Input<Tensor>("Triangles");
        const Tensor* barycentric_coordinates_tensor = context.Input<Tensor>("BarycentricCoordinates");
        const Tensor* triangle_ids_tensor = context.Input<Tensor>("TriangleIds");
        const Tensor* df_barycentric_coordinates_tensor = context.Input<Tensor>(
                framework::GradVarName("BarycentricCoordinates"));
        Tensor* df_vertices_tensor = context.Output<Tensor>(
                framework::GradVarName("Vertices"));

        df_vertices_tensor->mutable_data<float>(context.GetPlace());

        const int image_height = context.Attr<int>("image_height");
        const int image_width = context.Attr<int>("image_width");

        auto vertices_eigen = framework::EigenVector<float>::Flatten(*vertices_tensor);
        auto triangles_eigen = framework::EigenVector<int>::Flatten(*triangles_tensor);
        auto barycentric_coordinates_eigen = framework::EigenVector<float>::Flatten(*barycentric_coordinates_tensor);
        auto triangle_ids_eigen = framework::EigenVector<int>::Flatten(*triangle_ids_tensor);
        auto df_barycentric_coordinates_eigen = framework::EigenVector<float>::Flatten(*df_barycentric_coordinates_tensor);
        auto df_vertices_eigen = framework::EigenVector<float>::Flatten(*df_vertices_tensor);

        const float* vertices_data = vertices_eigen.data();
        const unsigned int vertex_count = vertices_eigen.size() / 4;
        const int* triangles_data = triangles_eigen.data();
        const float* barycentric_coordinates_data = barycentric_coordinates_eigen.data();
        const int* triangle_ids_data = triangle_ids_eigen.data();
        const float* df_barycentric_coordinates_data = df_barycentric_coordinates_eigen.data();
        float* df_vertices_data = df_vertices_eigen.data();

        //Start computing
        std::fill(df_vertices_data, df_vertices_data + vertex_count * 4, 0.0f);

        for (unsigned int pixel_id = 0; pixel_id < image_height * image_width; ++pixel_id) {
            const float bc0 = barycentric_coordinates_data[3 * pixel_id];
            const float bc1 = barycentric_coordinates_data[3 * pixel_id + 1];
            const float bc2 = barycentric_coordinates_data[3 * pixel_id + 2];

        if (bc0 + bc1 + bc2 < kDegenerateBarycentricCoordinatesCutoff) {
            continue;
        }

        const float df_dbc0 = df_barycentric_coordinates_data[3 * pixel_id];
        const float df_dbc1 = df_barycentric_coordinates_data[3 * pixel_id + 1];
        const float df_dbc2 = df_barycentric_coordinates_data[3 * pixel_id + 2];

        const int triangle_at_current_pixel = triangle_ids_data[pixel_id];
        const int* vertices_at_current_pixle = &triangles_data[3 * triangle_at_current_pixel];

        const int v0_id = 4 * vertices_at_current_pixle[0];
        const int v1_id = 4 * vertices_at_current_pixle[1];
        const int v2_id = 4 * vertices_at_current_pixle[2];

        const float x0 = vertices_data[v0_id];
        const float y0 = vertices_data[v0_id + 1];
        const float w0 = vertices_data[v0_id + 3];
        const float x1 = vertices_data[v1_id];
        const float y1 = vertices_data[v1_id + 1];
        const float w1 = vertices_data[v1_id + 3];
        const float x2 = vertices_data[v2_id];
        const float y2 = vertices_data[v2_id + 1];
        const float w2 = vertices_data[v2_id + 3];

        const int ix = pixel_id % image_width;
        const int iy = pixel_id / image_width;
        const float pixel_x = 2 * (ix + 0.5f) / image_width - 1.0f;
        const float pixel_y = 2 * (iy + 0.5f) / image_height - 1.0f;

        // Baricentric gradients wrt each vertex coordinate.
        const float dbc0_dx = pixel_y * (w1 - w2) - (y1 - y2);
        const float dbc1_dx = pixel_y * (w2 - w0) - (y2 - y0);
        const float dbc2_dx = -(dbc0_dx + dbc1_dx);
        const float dbc0_dy = (x1 - x2) - pixel_x * (w1 - w2);
        const float dbc1_dy = (x2 - x0) - pixel_x * (w2 - w0);
        const float dbc2_dy = -(dbc0_dy + dbc1_dy);
        const float dbc0_dw = pixel_x * (y1 - y2) - pixel_y * (x1 - x2);
        const float dbc1_dw = pixel_x * (y2 - y0) - pixel_y * (x2 - x0);
        const float dbc2_dw = -(dbc0_dw + dbc1_dw);

        const float df_dx = df_dbc0 * dbc0_dx + df_dbc1 * dbc1_dx + df_dbc2 * dbc2_dx;
        const float df_dy = df_dbc0 * dbc0_dy + df_dbc1 * dbc1_dy + df_dbc2 * dbc2_dy;
        const float df_dw = df_dbc0 * dbc0_dw + df_dbc1 * dbc1_dw + df_dbc2 * dbc2_dw;

        // Values of edge equations and inverse w at the current pixel.
        const float edge0_w = x2 * dbc0_dx + y2 * dbc0_dy + w2 * dbc0_dw;
        const float edge1_w = x2 * dbc1_dx + y2 * dbc1_dy + w2 * dbc1_dw;
        const float edge2_w = x1 * dbc2_dx + y1 * dbc2_dy + w1 * dbc2_dw;
        const float w_inv = edge0_w + edge1_w + edge2_w;

        const float w_sqr = 1 / (w_inv * w_inv);

        const float edge0 = w_sqr * edge0_w;
        const float edge1 = w_sqr * edge1_w;
        const float edge2 = w_sqr * edge2_w;

        df_vertices_data[v0_id + 0] += edge0 * df_dx;
        df_vertices_data[v0_id + 1] += edge0 * df_dy;
        df_vertices_data[v0_id + 3] += edge0 * df_dw;
        df_vertices_data[v1_id + 0] += edge1 * df_dx;
        df_vertices_data[v1_id + 1] += edge1 * df_dy;
        df_vertices_data[v1_id + 3] += edge1 * df_dw;
        df_vertices_data[v2_id + 0] += edge2 * df_dx;
        df_vertices_data[v2_id + 1] += edge2 * df_dy;
        df_vertices_data[v2_id + 3] += edge2 * df_dw;

        }
    }
};

} // namespace operators
} // namespace paddle

