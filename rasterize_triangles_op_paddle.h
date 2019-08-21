/* Copyright (c) ... */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/eigen.h"
#include "rasterize_triangles_impl.h"

namespace {
// Threshold for a barycentric coordinate triplet's sum, below which the
// coordinates at a pixel are deemed degenerate. Most such degenerate triplets
// in an image will be exactly zero, as this is how pixels outside the mesh
// are rendered.
constexpr float kDegenerateBarycentricCoordinatesCutoff = 0.9f;

// If the area of a triangle is very small in screen space, the corner vertices
// are approaching colinearity, and we should drop the gradient to avoid
// numerical instability (in particular, blowup, as the forward pass computation
// already only has 8 bits of precision).
constexpr float kMinimumTriangleArea = 1e-13;

}  //namespace


namespace paddle {
namespace operators{

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
class RasterizeTrianglesKernel : public framework::OpKernel<T> {
public:
    void Compute(const framework::ExecutionContext& context) const override {
        Tensor* vertices_tensor = context.Input<Tensor>("Vertices");
        Tensor* triangles_tensor = context.Input<Tensor>("Triangles");
        Tensor* barycentric_coordinates_tensor = context.Output<Tensor>("BarycentricCoordinates");
        Tensor* triangle_ids_tensor = context.Output<Tensor>("TriangleIds");
        Tensor* z_buffer_tensor = context.Output<Tensor>("ZBuffer");

        const int image_height = context.Attr<int>("image_height");
        const int image_width = context.Attr<int>("image_width");

        auto vertices_eigen = framework::EigenVector<float>::Flatten(*vertices_tensor);
        auto triangles_eigen = framework::EigenVector<int>::Flatten(*triangles_tensor);

        auto bc_eigen = framework::EigenVector<float>::Flatten(*barycentric_coordinates_tensor).setZero();
        auto ti_eigen = framework::EigenVector<int>::Flatten(*triangle_ids_tensor).setZero();
        auto zb_eigen = framework::EigenVector<float>::Flatten(*z_buffer_tensor).setConstant(1.0f);

        const float* vertices_data = vertices_eigen.data();
        const int* triangles_data = triangles_eigen.data();
        const int triangle_count = triangles_eigen.size() / 3;

        RasterizeTrianglesImpl(vertices_data, triangles_data, triangle_count,
                               image_height, image_width,
                               ti_eigen.data(),
                               bc_eigen.data(),
                               zb_eigen.data());

    }

};

template <typename DeviceContext, typename T>
class RasterizeTrianglesGradKernel : public framework::OpKernel<T> {
public:
    void Compute(const framework::ExecutionContext& ctx) const override {
        Tensor* vertices_tensor = context.Input<Tensor>("Vertices");
        Tensor* triangles_tensor = context.Input<Tensor>("Triangles");
        Tensor* barycentric_coordinates_tensor = context.Input<Tensor>("BarycentricCoordinates");
        Tensor* triangle_ids_tensor = context.Input<Tensor>("TriangleIds");
        Tensor* df_barycentric_coordinates_tensor = context.Input<Tensor>(
                framework::GradVarName("BarycentricCoordinates"));
        Tensor* df_vertices_tensor = context.Output<Tensor>(
                framework::GradVarName("Vertices"));

        const int image_height = context.Attr<int>("image_height");
        const int image_width = context.Attr<int>("image_width");

        auto vertices_eigen = framework::EigenVector<float>::Flatten(*vertices_tensor);
        auto triangles_eigen = framework::EigenVector<int>::Flatten(*triangles_tensor);
        auto barycentric_coordinates_eigen = framework::EigenVector<float>::Flatten(*barycentric_coordinates_tensor);
        auto triangles_ids_eigen = framework::EigenVector<int>::Flatten(*triangles_ids_tensor);
        auto df_barycentric_coordinates_eigen = framework::EigenVector<float>::Flatten(*df_barycentric_coordinates_tensor);
        auto df_vertices_eigen = framework::EigenVector<float>::Flatten(*df_vertices_tensor);

        const float* vertices_data = vertices_eigen.data();
        const unsigned int vertex_count = vertices_eigen.size() / 4;
        const int* triangles_data = triangles_eigen.data();
        const float* barycentric_coordinates_data = barycentric_coordinates_eigen.data();
        const int* triangle_ids_data = triangles_ids_eigen.data();
        const float* df_barycentric_coordinates_data = df_barycentric_coordinates_eigen.data();
        float* df_vertices_data = df_vertices_eigen.data();

        std::fill(df_vertices_data, df_vertices_data + vertex_count *4, 0.0f);

        for (unsigned int pixel_id = 0; pixel_id < image_height * image_width; ++pixel_id) {
            const float b0 = barycentric_coordinates_data[3 * pixel_id];
            const float b1 = barycentric_coordinates_data[3 * pixel_id + 1];
            const float b2 = barycentric_coordinates_data[3 * pixel_id + 2];

        if (b0 + b1 + b2 < kDegenerateBarycentricCoordinatesCutoff) {
            continue;
        }

        const float df_b0 = df_barycentric_coordinates_data[3 * pixel_id];
        const float df_b1 = df_barycentric_coordinates_data[3 * pixel_id + 1];
        const float df_b2 = df_barycentric_coordinates_data[3 * pixel_id + 2];

        const int triangle_at_current_pixel = triangle_ids_data[pixel_id];
        const int* vertices_at_current_pixle = &triangles_data[3 * triangle_at_current_pixel];

        //Extract vertex indices for the current triangle.
        const int v0_id = 4 * vertices_at_current_pixle[0];
        const int v1_id = 4 * vertices_at_current_pixle[1];
        const int v2_id = 4 * vertices_at_current_pixle[2];

        //Extract x,y,w components of the vertices' clip space coordinates.
        const float x0 = vertices_data[v0_id];
        const float y0 = vertices_data[v0_id + 1];
        const float w0 = vertices_data[v0_id + 3];
        const float x1 = vertices_data[v1_id];
        const float y1 = vertices_data[v1_id + 1];
        const float w1 = vertices_data[v1_id + 3];
        const float x2 = vertices_data[v2_id];
        const float y2 = vertices_data[v2_id + 1];
        const float w2 = vertices_data[v2_id + 3];

        //Compute pixel's NDC-s.
        const int ix = pixel_id % image_width;
        const int iy = pixel_id / image_width;
        const float px = 2 * (ix + 0.5f) / image_width - 1.0f;
        const float py = 2 * (iy + 0.5f) / image_height - 1.0f;

        // Baricentric gradients wrt each vertex coordinate share a common factor.
        const float db0_dx = py * (w1 - w2) - (y1 - y2);
        const float db1_dx = py * (w2 - w0) - (y2 - y0);
        const float db2_dx = -(db0_dx + db1_dx);
        const float db0_dy = (x1 - x2) - px * (w1 - w2);
        const float db1_dy = (x2 - x0) - px * (w2 - w0);
        const float db2_dy = -(db0_dy + db1_dy);
        const float db0_dw = px * (y1 - y2) - py * (x1 - x2);
        const float db1_dw = px * (y2 - y0) - py * (x2 - x0);
        const float db2_dw = -(db0_dw + db1_dw);

        // Combine them with chain rule.
        const float df_dx = df_db0 * db0_dx + df_db1 * db1_dx + df_db2 * db2_dx;
        const float df_dy = df_db0 * db0_dy + df_db1 * db1_dy + df_db2 * db2_dy;
        const float df_dw = df_db0 * db0_dw + df_db1 * db1_dw + df_db2 * db2_dw;

        // Values of edge equations and inverse w at the current pixel.
        const float edge0_over_w = x2 * db0_dx + y2 * db0_dy + w2 * db0_dw;
        const float edge1_over_w = x2 * db1_dx + y2 * db1_dy + w2 * db1_dw;
        const float edge2_over_w = x1 * db2_dx + y1 * db2_dy + w1 * db2_dw;
        const float w_inv = edge0_over_w + edge1_over_w + edge2_over_w;

        // All gradients share a common denominator.
        const float w_sqr = 1 / (w_inv * w_inv);

        // Gradients wrt each vertex share a common factor.
        const float edge0 = w_sqr * edge0_over_w;
        const float edge1 = w_sqr * edge1_over_w;
        const float edge2 = w_sqr * edge2_over_w;

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

