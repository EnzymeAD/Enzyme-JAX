#ifndef ENZYMEXLA_MPS_ENUMS_TD
#define ENZYMEXLA_MPS_ENUMS_TD

include "Dialect.td"

include "mlir/IR/EnumAttr.td"

class MPS_I32Enum<string name, string summary, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, summary, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mps";
}

class MPS_EnumAttr<EnumAttrInfo enumInfo, string mnemonic>
    : EnumAttr<MPS_Dialect, enumInfo, mnemonic> {
  let assemblyFormat = "`<` $value `>`";
}

// MPSGraphTensorNamedDataLayout
def MPS_TensorNamedDataLayout : MPS_I32Enum<"TensorNamedDataLayout",
    "named tensor data layout", [
  I32EnumAttrCase<"NCHW", 0>,
  I32EnumAttrCase<"NHWC", 1>,
  I32EnumAttrCase<"OIHW", 2>,
  I32EnumAttrCase<"HWIO", 3>,
  I32EnumAttrCase<"CHW", 4>,
  I32EnumAttrCase<"HWC", 5>,
  I32EnumAttrCase<"HW", 6>,
  I32EnumAttrCase<"NCDHW", 7>,
  I32EnumAttrCase<"NDHWC", 8>,
  I32EnumAttrCase<"OIDHW", 9>,
  I32EnumAttrCase<"DHWIO", 10>,
]>;
def MPS_TensorNamedDataLayoutAttr
    : MPS_EnumAttr<MPS_TensorNamedDataLayout, "layout">;

// MPSGraphPaddingStyle
def MPS_PaddingStyle : MPS_I32Enum<"PaddingStyle", "tensor padding style", [
  I32EnumAttrCase<"Explicit", 0, "explicit">,
  I32EnumAttrCase<"TF_VALID", 1>,
  I32EnumAttrCase<"TF_SAME", 2>,
  I32EnumAttrCase<"ExplicitOffset", 3, "explicitOffset">,
  I32EnumAttrCase<"ONNX_SAME_LOWER", 4>,
]>;
def MPS_PaddingStyleAttr : MPS_EnumAttr<MPS_PaddingStyle, "padding_style">;

// MPSGraphPaddingMode
def MPS_PaddingMode : MPS_I32Enum<"PaddingMode", "tensor padding mode", [
  I32EnumAttrCase<"Constant", 0, "constant">,
  I32EnumAttrCase<"Reflect", 1, "reflect">,
  I32EnumAttrCase<"Symmetric", 2, "symmetric">,
  I32EnumAttrCase<"ClampToEdge", 3, "clampToEdge">,
  I32EnumAttrCase<"Zero", 4, "zero">,
  I32EnumAttrCase<"Periodic", 5, "periodic">,
  I32EnumAttrCase<"AntiPeriodic", 6, "antiPeriodic">,
]>;
def MPS_PaddingModeAttr : MPS_EnumAttr<MPS_PaddingMode, "padding_mode">;

// MPSGraphReductionMode
def MPS_ReductionMode : MPS_I32Enum<"ReductionMode", "reduction mode", [
  I32EnumAttrCase<"Min", 0, "min">,
  I32EnumAttrCase<"Max", 1, "max">,
  I32EnumAttrCase<"Sum", 2, "sum">,
  I32EnumAttrCase<"Product", 3, "product">,
  I32EnumAttrCase<"ArgumentMin", 4, "argumentMin">,
  I32EnumAttrCase<"ArgumentMax", 5, "argumentMax">,
]>;
def MPS_ReductionModeAttr : MPS_EnumAttr<MPS_ReductionMode, "reduction_mode">;

// MPSGraphScatterMode
def MPS_ScatterMode : MPS_I32Enum<"ScatterMode", "scatter mode", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"Div", 3, "div">,
  I32EnumAttrCase<"Min", 4, "min">,
  I32EnumAttrCase<"Max", 5, "max">,
  I32EnumAttrCase<"Set", 6, "set">,
]>;
def MPS_ScatterModeAttr : MPS_EnumAttr<MPS_ScatterMode, "scatter_mode">;

// MPSGraphResizeMode
def MPS_ResizeMode : MPS_I32Enum<"ResizeMode", "resize mode", [
  I32EnumAttrCase<"Nearest", 0, "nearest">,
  I32EnumAttrCase<"Bilinear", 1, "bilinear">,
]>;
def MPS_ResizeModeAttr : MPS_EnumAttr<MPS_ResizeMode, "resize_mode">;

// MPSGraphResizeNearestRoundingMode
def MPS_ResizeNearestRoundingMode : MPS_I32Enum<"ResizeNearestRoundingMode",
    "rounding mode for nearest resampling", [
  I32EnumAttrCase<"RoundPreferCeil", 0, "roundPreferCeil">,
  I32EnumAttrCase<"RoundPreferFloor", 1, "roundPreferFloor">,
  I32EnumAttrCase<"Ceil", 2, "ceil">,
  I32EnumAttrCase<"Floor", 3, "floor">,
  I32EnumAttrCase<"RoundToEven", 4, "roundToEven">,
  I32EnumAttrCase<"RoundToOdd", 5, "roundToOdd">,
]>;
def MPS_ResizeNearestRoundingModeAttr
    : MPS_EnumAttr<MPS_ResizeNearestRoundingMode, "nearest_rounding_mode">;

// MPSGraphLossReductionType
def MPS_LossReductionType : MPS_I32Enum<"LossReductionType",
    "loss reduction type", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"Sum", 1, "sum">,
  I32EnumAttrCase<"Mean", 2, "mean">,
]>;
def MPS_LossReductionTypeAttr
    : MPS_EnumAttr<MPS_LossReductionType, "loss_reduction_type">;

// MPSGraphRandomDistribution
def MPS_RandomDistribution : MPS_I32Enum<"RandomDistribution",
    "random distribution", [
  I32EnumAttrCase<"Uniform", 0, "uniform">,
  I32EnumAttrCase<"Normal", 1, "normal">,
  I32EnumAttrCase<"TruncatedNormal", 2, "truncatedNormal">,
]>;
def MPS_RandomDistributionAttr
    : MPS_EnumAttr<MPS_RandomDistribution, "random_distribution">;

// MPSGraphRandomNormalSamplingMethod
def MPS_RandomNormalSamplingMethod : MPS_I32Enum<"RandomNormalSamplingMethod",
    "sampling method for normal distributions", [
  I32EnumAttrCase<"InvCDF", 0, "invCDF">,
  I32EnumAttrCase<"BoxMuller", 1, "boxMuller">,
]>;
def MPS_RandomNormalSamplingMethodAttr
    : MPS_EnumAttr<MPS_RandomNormalSamplingMethod, "sampling_method">;

// MPSGraphFFTScalingMode
def MPS_FFTScalingMode : MPS_I32Enum<"FFTScalingMode", "FFT scaling mode", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"Size", 1, "size">,
  I32EnumAttrCase<"Unitary", 2, "unitary">,
]>;
def MPS_FFTScalingModeAttr : MPS_EnumAttr<MPS_FFTScalingMode, "fft_scaling_mode">;

// MPSGraphNonMaximumSuppressionCoordinateMode
def MPS_NonMaximumSuppressionCoordinateMode
    : MPS_I32Enum<"NonMaximumSuppressionCoordinateMode",
    "box coordinate mode", [
  I32EnumAttrCase<"CornersHeightFirst", 0, "cornersHeightFirst">,
  I32EnumAttrCase<"CornersWidthFirst", 1, "cornersWidthFirst">,
  I32EnumAttrCase<"CentersHeightFirst", 2, "centersHeightFirst">,
  I32EnumAttrCase<"CentersWidthFirst", 3, "centersWidthFirst">,
]>;
def MPS_NonMaximumSuppressionCoordinateModeAttr
    : MPS_EnumAttr<MPS_NonMaximumSuppressionCoordinateMode, "coordinate_mode">;

// MPSGraphRNNActivation
def MPS_RNNActivation : MPS_I32Enum<"RNNActivation", "RNN activation", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"Relu", 1, "relu">,
  I32EnumAttrCase<"Tanh", 2, "tanh">,
  I32EnumAttrCase<"Sigmoid", 3, "sigmoid">,
  I32EnumAttrCase<"HardSigmoid", 4, "hardSigmoid">,
]>;
def MPS_RNNActivationAttr : MPS_EnumAttr<MPS_RNNActivation, "rnn_activation">;

// MPSGraphSparseStorageType
def MPS_SparseStorageType : MPS_I32Enum<"SparseStorageType",
    "sparse storage type", [
  I32EnumAttrCase<"COO", 0>,
  I32EnumAttrCase<"CSC", 1>,
  I32EnumAttrCase<"CSR", 2>,
]>;
def MPS_SparseStorageTypeAttr
    : MPS_EnumAttr<MPS_SparseStorageType, "sparse_storage_type">;

// MPSGraphPoolingReturnIndicesMode
def MPS_PoolingReturnIndicesMode : MPS_I32Enum<"PoolingReturnIndicesMode",
    "layout of the indices returned by max pooling", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"GlobalFlatten1D", 1, "globalFlatten1D">,
  I32EnumAttrCase<"GlobalFlatten2D", 2, "globalFlatten2D">,
  I32EnumAttrCase<"GlobalFlatten3D", 3, "globalFlatten3D">,
  I32EnumAttrCase<"GlobalFlatten4D", 4, "globalFlatten4D">,
  I32EnumAttrCase<"LocalFlatten1D", 5, "localFlatten1D">,
  I32EnumAttrCase<"LocalFlatten2D", 6, "localFlatten2D">,
  I32EnumAttrCase<"LocalFlatten3D", 7, "localFlatten3D">,
  I32EnumAttrCase<"LocalFlatten4D", 8, "localFlatten4D">,
]>;
def MPS_PoolingReturnIndicesModeAttr
    : MPS_EnumAttr<MPS_PoolingReturnIndicesMode, "return_indices_mode">;

#endif // ENZYMEXLA_MPS_ENUMS_TD
