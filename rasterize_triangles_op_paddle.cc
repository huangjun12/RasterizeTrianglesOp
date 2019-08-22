/* Copyright... */

#include "paddle/fluid/operators/rasterize_triangles_op.h"


namespace paddle {
namespace operators {

class RasterizeTrianglesOpMaker :  public framework::OpProtoAndCheckerMaker{
public:
    void Make() override{
        AddInput("Vertices",
                "(Tensor), The first input tensor of rasterize triangles op."
                "This is a 2-D tensor with the shape of [vertex_count, 4]."
                "Each row is the 3-D positions of the mesh vertices in clip-space (XYZW).");
        AddInput("Triangles",
                "(Tensor), The second input tensor of rasterize triangles op."
                "This is a 2-D tensor with the shape of [triangle_count, 3]."
                "Each row is a tuple of indices into vertices specifying a triangle to be drawn.");
        AddOutput("BarycentricCoordinates",
                "(Tensor), The first output tensor of rasterize triangles op."
                "3-D tensor with shape [image_height, image_width, 3] containing the rendered"
                "barycentric coordinate triplet per pixel, before perspective correction."
                "The triplet is the zero vector if the pixel is outside the mesh boundary."
                "For valid pixels, the ordering of the coordinates corresponds to the ordering"
                "in triangles.");
        AddOutput("TriangleIds",
                "(Tensor), The second output tensor of rasterize triangles op."
                "2-D tensor with shape [image_height, image_width]. Contains the triangle id value"
                "for each pixel in the output image. For pixels within the mesh, this is the"
                "integer value in the range [0, num_vertices] from triangles."
                "For vertices outside the mesh this is 0; 0 can either indicate belonging to triangle 0,"
                "or being outside the mesh. This ensures all returned triangle ids will validly index"
                "into the vertex array, enabling the use of tf.gather with indices from this tensor."
                "The barycentric coordinates can be used to determine pixel validity instead.");
        AddOutput("ZBuffer",
                "(Tensor), The third output tensor of rasterize triangles op."
                "2-D tensor with shape [image_height, image_width]. Contains the coordinate in"
                "Normalized Device Coordinates for each pixel occupied by a triangle.");
        AddAttr<int>("image_height", "positive int attribute specifying the height of the output image.").GreaterThan(0);
        AddAttr<int>("image_width", "positive int attribute specifying the width of the output image.").GreaterThan(0);
        AddComment(R"DOC(
Rasterize Triangles Operator.

This operator is used for rendering mesh geometry.
)DOC");
    }
};

class RasterizeTrianglesOp : public framework::OperatorWithKernel{
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Vertices"), true, "Input(Vertices) of RasterizeTrianglesOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Triangles"), true,"Input(Triangles) of RasterizeTrianglesOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("BarycentricCoordinates"), true,
              "Output(BarycentricCoordinates) of RasterizeTrianglesOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("TriangleIds"), true,
              "Output(TriangleIds) of RasterizeTrianglesOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("ZBuffer"), true,
              "Output(ZBuffer) of RasterizeTrianglesOp should not be null.");

    auto v_dims = ctx->GetInputDim("Vertices");
    auto t_dims = ctx->GetInputDim("Triangles");

    PADDLE_ENFORCE_EQ(v_dims.size(), 2UL,
            "Input(Vertices) should be 2-D tensor.");
    PADDLE_ENFORCE_EQ(v_dims[-1], 4UL,
            "The second dim of Vertices should be 4, which consists"
            "the 3-D positions of the mesh vertices in clip-space (XYZW), but get u%.",
            v_dims[-1]);

    PADDLE_ENFORCE_EQ(t_dims.size(), 2UL,
                      "Input(Triangles) should be 2-D tensor.");
    PADDLE_ENFORCE_EQ(t_dims[-1], 3UL,
                      "The second dim of Triangles should be 3, which consists the"
                      "indices into vertices specifying a triangle to be drawn, but get u%.",
                      t_dims[-1]);

    int image_height = ctx->Attrs().Get<int>("image_height");
    int image_width = ctx->Attrs().Get<int>("image_width");

    ctx->SetOutputDim("BarycentricCoordinates", {image_height, image_width, static_cast<int>(3)});
    ctx->SetOutputDim("TriangleIds", {image_height, image_width});
    ctx->SetOutputDim("ZBuffer", {image_height, image_width});
  }
};

class RasterizeTrianglesOpGradMaker : public framework::SingleGradOpDescMaker{
public:
    using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

protected:
    std::unique_ptr<framework::OpDesc> Apply() const override {
        std::unique_ptr<framework::OpDesc> retv(new framework::OpDesc());
        retv->SetType("rasterize_triangles_grad");
        retv->SetInput("Vertices", Input("Vertices"));
        retv->SetInput("Triangles", Input("Triangles"));
        retv->SetInput("BarycentricCoordinates", Output("BarycentricCoordinates"));
        retv->SetInput("TriangleIds", Output("TriangleIds"));
        retv->SetInput(framework::GradVarName("BarycentricCoordinates"),
                       OutputGrad("BarycentricCoordinates"));
        retv->SetOutput(framework::GradVarName("Vertices"), InputGrad("Vertices"));
        retv->SetAttrMap(Attrs());
        return retv;
    }

};

class RasterizeTrianglesGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Vertices"), "Input(Vertices) of RasterizeTrianglesGradOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Triangles"), "Input(Triangles) of RasterizeTrianglesGradOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("BarycentricCoordinates"),
                                 "Input(BarycentricCoordinates) of RasterizeTrianglesGradOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput("TriangleIds"),
                                 "Input(TriangleIds) of RasterizeTrianglesGradOp should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("BarycentricCoordinates")),
                                 "Input(BarycentricCoordinates@GRAD) of RasterizeTrianglesGradOp should not be null");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Vertices")),
                                 "Output(Vertices@Grad) of RasterizeTrianglesGradOp should be not null.");

    ctx->SetOutputDim(framework::GradVarName("Vertices"), ctx->GetInputDim("Vertices"));
  }
};

} // namespace operators

} // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(rasterize_triangles, ops::RasterizeTrianglesOp, ops::RasterizeTrianglesOpMaker,
    ops::RasterizeTrianglesOpGradMaker);
REGISTER_OPERATOR(rasterize_triangles_grad, ops::RasterizeTrianglesGradOp);
REGISTER_OP_CPU_KERNEL(rasterize_triangles,
    ops::RasterizeTrianglesKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RasterizeTrianglesKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(rasterize_triangles_grad,
    ops::RasterizeTrianglesGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::RasterizeTrianglesGradKernel<paddle::platform::CPUDeviceContext, double>);




