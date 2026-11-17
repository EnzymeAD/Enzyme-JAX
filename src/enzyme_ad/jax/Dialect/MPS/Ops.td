#ifndef ENZYMEXLA_MPS_OPS_TD
#define ENZYMEXLA_MPS_OPS_TD

include "Dialect.td"
include "Enums.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Op classes
//===----------------------------------------------------------------------===//

class MPS_UnaryOp<string mnemonic, string summaryText, list<Trait> traits = []>
    : MPS_Op<mnemonic, !listconcat([Pure, Elementwise,
                                    SameOperandsAndResultType], traits)> {
  let summary = summaryText;
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

// Unary op whose result element type may differ from the operand's.
class MPS_UnaryShapedOp<string mnemonic, string summaryText>
    : MPS_Op<mnemonic, [Pure, Elementwise, SameOperandsAndResultShape]> {
  let summary = summaryText;
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

class MPS_UnaryPredicateOp<string mnemonic, string summaryText>
    : MPS_Op<mnemonic, [Pure, Elementwise, SameOperandsAndResultShape]> {
  let summary = summaryText;
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_BoolTensor:$result);
}

// Binary ops broadcast their operands following NumPy rules.
class MPS_BinaryOp<string mnemonic, string summaryText,
                   list<Trait> traits = []>
    : MPS_Op<mnemonic, !listconcat([Pure, ResultsBroadcastableShape,
                                    SameOperandsAndResultElementType],
                                   traits)> {
  let summary = summaryText;
  let arguments = (ins MPS_Tensor:$primary, MPS_Tensor:$secondary);
  let results = (outs MPS_Tensor:$result);
}

class MPS_ComparisonOp<string mnemonic, string summaryText,
                       list<Trait> traits = []>
    : MPS_Op<mnemonic, !listconcat([Pure, ResultsBroadcastableShape,
                                    SameOperandsElementType], traits)> {
  let summary = summaryText;
  let arguments = (ins MPS_Tensor:$primary, MPS_Tensor:$secondary);
  let results = (outs MPS_BoolTensor:$result);
}

// Reduction over `axes`; reduced dimensions are kept with size 1. Without
// `axes`, all dimensions are reduced.
class MPS_ReductionOp<string mnemonic, string summaryText>
    : MPS_Op<mnemonic, [Pure, SameOperandsAndResultElementType]> {
  let summary = summaryText;
  let description = [{
    Covers the `axis:` and `axes:` overloads. The reduced dimensions are kept
    with size 1. If `axes` is absent, all dimensions are reduced.
  }];
  let arguments = (ins MPS_Tensor:$tensor,
                       OptionalAttr<DenseI64ArrayAttr>:$axes);
  let results = (outs MPS_Tensor:$result);
}

class MPS_CumulativeOp<string mnemonic, string summaryText>
    : MPS_Op<mnemonic, [Pure, SameOperandsAndResultType]> {
  let summary = summaryText;
  let description = [{
    `exclusive` excludes the current element from the accumulation; `reverse`
    accumulates from the end of `axis`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis,
                       DefaultValuedAttr<BoolAttr, "false">:$exclusive,
                       DefaultValuedAttr<BoolAttr, "false">:$reverse);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Descriptors
//===----------------------------------------------------------------------===//

// MPSGraphConvolution2DOpDescriptor
defvar Convolution2DDescriptor = (ins
  I64Attr:$stride_in_x, I64Attr:$stride_in_y,
  I64Attr:$dilation_rate_in_x, I64Attr:$dilation_rate_in_y,
  I64Attr:$groups,
  I64Attr:$padding_left, I64Attr:$padding_right,
  I64Attr:$padding_top, I64Attr:$padding_bottom,
  MPS_PaddingStyleAttr:$padding_style,
  MPS_TensorNamedDataLayoutAttr:$data_layout,
  MPS_TensorNamedDataLayoutAttr:$weights_layout);

// MPSGraphConvolution3DOpDescriptor
defvar Convolution3DDescriptor = (ins
  I64Attr:$stride_in_x, I64Attr:$stride_in_y, I64Attr:$stride_in_z,
  I64Attr:$dilation_rate_in_x, I64Attr:$dilation_rate_in_y,
  I64Attr:$dilation_rate_in_z,
  I64Attr:$groups,
  I64Attr:$padding_left, I64Attr:$padding_right,
  I64Attr:$padding_top, I64Attr:$padding_bottom,
  I64Attr:$padding_front, I64Attr:$padding_back,
  MPS_PaddingStyleAttr:$padding_style,
  MPS_TensorNamedDataLayoutAttr:$data_layout,
  MPS_TensorNamedDataLayoutAttr:$weights_layout);

// MPSGraphDepthwiseConvolution2DOpDescriptor
defvar DepthwiseConvolution2DDescriptor = (ins
  I64Attr:$stride_in_x, I64Attr:$stride_in_y,
  I64Attr:$dilation_rate_in_x, I64Attr:$dilation_rate_in_y,
  I64Attr:$padding_left, I64Attr:$padding_right,
  I64Attr:$padding_top, I64Attr:$padding_bottom,
  MPS_PaddingStyleAttr:$padding_style,
  MPS_TensorNamedDataLayoutAttr:$data_layout,
  MPS_TensorNamedDataLayoutAttr:$weights_layout);

// MPSGraphDepthwiseConvolution3DOpDescriptor
defvar DepthwiseConvolution3DDescriptor = (ins
  DenseI64ArrayAttr:$strides,
  DenseI64ArrayAttr:$dilation_rates,
  DenseI64ArrayAttr:$padding_values,
  MPS_PaddingStyleAttr:$padding_style,
  I64Attr:$channel_dimension_index);

// MPSGraphPooling2DOpDescriptor. `returnIndicesMode` is an attribute of the
// ops that produce or consume indices; `returnIndicesDataType` is the element
// type of the indices tensor.
defvar Pooling2DDescriptor = (ins
  I64Attr:$kernel_width, I64Attr:$kernel_height,
  I64Attr:$stride_in_x, I64Attr:$stride_in_y,
  I64Attr:$dilation_rate_in_x, I64Attr:$dilation_rate_in_y,
  I64Attr:$padding_left, I64Attr:$padding_right,
  I64Attr:$padding_top, I64Attr:$padding_bottom,
  MPS_PaddingStyleAttr:$padding_style,
  MPS_TensorNamedDataLayoutAttr:$data_layout,
  BoolAttr:$ceil_mode,
  BoolAttr:$include_zero_pad_to_average);

// MPSGraphPooling4DOpDescriptor. See Pooling2DDescriptor for the indices
// fields.
defvar Pooling4DDescriptor = (ins
  DenseI64ArrayAttr:$kernel_sizes,
  DenseI64ArrayAttr:$strides,
  DenseI64ArrayAttr:$dilation_rates,
  DenseI64ArrayAttr:$padding_values,
  MPS_PaddingStyleAttr:$padding_style,
  BoolAttr:$ceil_mode,
  BoolAttr:$include_zero_pad_to_average);

// MPSGraphImToColOpDescriptor
defvar ImToColDescriptor = (ins
  I64Attr:$kernel_width, I64Attr:$kernel_height,
  I64Attr:$stride_in_x, I64Attr:$stride_in_y,
  I64Attr:$dilation_rate_in_x, I64Attr:$dilation_rate_in_y,
  I64Attr:$padding_left, I64Attr:$padding_right,
  I64Attr:$padding_top, I64Attr:$padding_bottom,
  MPS_TensorNamedDataLayoutAttr:$data_layout);

// MPSGraphFFTDescriptor
defvar FFTDescriptor = (ins
  BoolAttr:$inverse,
  MPS_FFTScalingModeAttr:$scaling_mode,
  BoolAttr:$round_to_odd_hermitean);

// MPSGraphRandomOpDescriptor. `dataType` is the result element type.
defvar RandomOpDescriptor = (ins
  MPS_RandomDistributionAttr:$distribution,
  F32Attr:$min, F32Attr:$max,
  I64Attr:$min_integer, I64Attr:$max_integer,
  F32Attr:$mean, F32Attr:$standard_deviation,
  MPS_RandomNormalSamplingMethodAttr:$sampling_method);

// MPSGraphStencilOpDescriptor
defvar StencilOpDescriptor = (ins
  MPS_ReductionModeAttr:$reduction_mode,
  DenseI64ArrayAttr:$offsets,
  DenseI64ArrayAttr:$strides,
  DenseI64ArrayAttr:$dilation_rates,
  DenseI64ArrayAttr:$explicit_padding,
  MPS_PaddingModeAttr:$boundary_mode,
  MPS_PaddingStyleAttr:$padding_style,
  F32Attr:$padding_constant);

// MPSGraphSingleGateRNNDescriptor
defvar SingleGateRNNDescriptor = (ins
  BoolAttr:$reverse, BoolAttr:$bidirectional, BoolAttr:$training,
  MPS_RNNActivationAttr:$activation);

// MPSGraphLSTMDescriptor
defvar LSTMDescriptor = (ins
  BoolAttr:$reverse, BoolAttr:$bidirectional, BoolAttr:$produce_cell,
  BoolAttr:$training, BoolAttr:$forget_gate_last,
  MPS_RNNActivationAttr:$input_gate_activation,
  MPS_RNNActivationAttr:$forget_gate_activation,
  MPS_RNNActivationAttr:$cell_gate_activation,
  MPS_RNNActivationAttr:$output_gate_activation,
  MPS_RNNActivationAttr:$activation);

// MPSGraphGRUDescriptor
defvar GRUDescriptor = (ins
  BoolAttr:$reverse, BoolAttr:$bidirectional, BoolAttr:$training,
  BoolAttr:$reset_gate_first, BoolAttr:$reset_after, BoolAttr:$flip_z,
  MPS_RNNActivationAttr:$update_gate_activation,
  MPS_RNNActivationAttr:$reset_gate_activation,
  MPS_RNNActivationAttr:$output_gate_activation);

//===----------------------------------------------------------------------===//
// Memory ops (MPSGraphMemoryOps.h)
//===----------------------------------------------------------------------===//

def MPS_ConstantOp : MPS_Op<"constant",
    [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "constant tensor";
  let description = [{
    Covers `constant` and `complexConstant`: scalar splats, shaped splats and
    raw data.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs MPS_Tensor:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

def MPS_VariableOp : MPS_Op<"variable", [AllTypesMatch<["value", "result"]>]> {
  let summary = "variable initialized with constant data";
  let description = [{
    The result identifies the variable resource; `mps.read` and `mps.assign`
    access its current value.
  }];
  let arguments = (ins ElementsAttr:$value);
  let results = (outs Res<MPS_Tensor, "", [MemAlloc]>:$result);
  let assemblyFormat = "attr-dict $value";
}

def MPS_VariableFromTensorOp : MPS_Op<"variable_from_tensor"> {
  let summary = "variable initialized with the value of a tensor";
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs Res<MPS_Tensor, "", [MemAlloc]>:$result);
}

def MPS_ReadOp : MPS_Op<"read"> {
  let summary = "reads the value of a variable at this point of execution";
  let arguments = (ins Arg<MPS_Tensor, "", [MemRead]>:$variable);
  let results = (outs MPS_Tensor:$result);
}

def MPS_AssignOp : MPS_Op<"assign"> {
  let summary = "writes a tensor to a variable at this point of execution";
  let arguments = (ins Arg<MPS_Tensor, "", [MemWrite]>:$variable,
                       MPS_Tensor:$tensor);
}

//===----------------------------------------------------------------------===//
// Arithmetic ops (MPSGraphArithmeticOps.h)
//===----------------------------------------------------------------------===//

def MPS_IdentityOp : MPS_UnaryOp<"identity", "copy of the input tensor">;
def MPS_ExponentOp : MPS_UnaryOp<"exponent", "elementwise e^x">;
def MPS_ExponentBase2Op : MPS_UnaryOp<"exponent_base2", "elementwise 2^x">;
def MPS_ExponentBase10Op : MPS_UnaryOp<"exponent_base10", "elementwise 10^x">;
def MPS_LogarithmOp : MPS_UnaryOp<"logarithm", "elementwise natural logarithm">;
def MPS_LogarithmBase2Op
    : MPS_UnaryOp<"logarithm_base2", "elementwise base-2 logarithm">;
def MPS_LogarithmBase10Op
    : MPS_UnaryOp<"logarithm_base10", "elementwise base-10 logarithm">;
def MPS_SquareOp : MPS_UnaryOp<"square", "elementwise square">;
def MPS_SquareRootOp : MPS_UnaryOp<"square_root", "elementwise square root">;
def MPS_ReciprocalSquareRootOp : MPS_UnaryOp<"reciprocal_square_root",
    "elementwise reciprocal square root"> {
  let description = [{
    Also covers the deprecated `reverseSquareRoot`.
  }];
}
def MPS_ReciprocalOp : MPS_UnaryOp<"reciprocal", "elementwise reciprocal">;
def MPS_NegativeOp : MPS_UnaryOp<"negative", "elementwise negation">;
def MPS_SignOp : MPS_UnaryOp<"sign", "elementwise sign">;
def MPS_CeilOp : MPS_UnaryOp<"ceil", "elementwise ceiling">;
def MPS_FloorOp : MPS_UnaryOp<"floor", "elementwise floor">;
def MPS_RoundOp : MPS_UnaryOp<"round", "elementwise round">;
def MPS_RintOp
    : MPS_UnaryOp<"rint", "elementwise round to nearest, ties to even">;
def MPS_TruncateOp
    : MPS_UnaryOp<"truncate", "elementwise round toward zero">;
def MPS_SinOp : MPS_UnaryOp<"sin", "elementwise sine">;
def MPS_CosOp : MPS_UnaryOp<"cos", "elementwise cosine">;
def MPS_TanOp : MPS_UnaryOp<"tan", "elementwise tangent">;
def MPS_SinhOp : MPS_UnaryOp<"sinh", "elementwise hyperbolic sine">;
def MPS_CoshOp : MPS_UnaryOp<"cosh", "elementwise hyperbolic cosine">;
def MPS_TanhOp : MPS_UnaryOp<"tanh", "elementwise hyperbolic tangent">;
def MPS_AsinOp : MPS_UnaryOp<"asin", "elementwise arcsine">;
def MPS_AcosOp : MPS_UnaryOp<"acos", "elementwise arccosine">;
def MPS_AtanOp : MPS_UnaryOp<"atan", "elementwise arctangent">;
def MPS_AsinhOp : MPS_UnaryOp<"asinh", "elementwise hyperbolic arcsine">;
def MPS_AcoshOp : MPS_UnaryOp<"acosh", "elementwise hyperbolic arccosine">;
def MPS_AtanhOp : MPS_UnaryOp<"atanh", "elementwise hyperbolic arctangent">;
def MPS_ErfOp : MPS_UnaryOp<"erf", "elementwise error function">;
def MPS_NotOp : MPS_UnaryOp<"not", "elementwise logical NOT">;
def MPS_ConjugateOp
    : MPS_UnaryOp<"conjugate", "elementwise complex conjugate">;
def MPS_BitwiseNotOp : MPS_UnaryOp<"bitwise_not", "elementwise bitwise NOT">;

def MPS_AbsoluteOp
    : MPS_UnaryShapedOp<"absolute", "elementwise absolute value">;
def MPS_AbsoluteSquareOp
    : MPS_UnaryShapedOp<"absolute_square", "elementwise absolute square">;
def MPS_RealPartOp : MPS_UnaryShapedOp<"real_part",
    "real part of a complex tensor">;
def MPS_ImaginaryPartOp : MPS_UnaryShapedOp<"imaginary_part",
    "imaginary part of a complex tensor">;
def MPS_BitwisePopulationCountOp : MPS_UnaryShapedOp<
    "bitwise_population_count", "elementwise count of set bits">;

def MPS_SignbitOp : MPS_UnaryPredicateOp<"signbit", "elementwise sign bit">;
def MPS_IsInfiniteOp
    : MPS_UnaryPredicateOp<"is_infinite", "elementwise check for infinity">;
def MPS_IsFiniteOp
    : MPS_UnaryPredicateOp<"is_finite", "elementwise check for finiteness">;
def MPS_IsNaNOp : MPS_UnaryPredicateOp<"is_nan", "elementwise check for NaN">;

def MPS_PlanarOp : MPS_Op<"planar", [Pure]> {
  let summary = "real tensor with separate real and imaginary planes";
  let description = [{
    Returns a real-valued tensor from a complex-valued tensor with the real
    and imaginary planes separated (`planarTensor`).
  }];
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_AdditionOp : MPS_BinaryOp<"addition", "elementwise addition",
                                  [Commutative]>;
def MPS_SubtractionOp
    : MPS_BinaryOp<"subtraction", "elementwise subtraction">;
def MPS_MultiplicationOp : MPS_BinaryOp<"multiplication",
    "elementwise multiplication", [Commutative]>;
def MPS_DivisionOp : MPS_BinaryOp<"division", "elementwise division">;
def MPS_DivisionNoNaNOp : MPS_BinaryOp<"division_no_nan",
    "elementwise division returning 0 when the divisor is 0">;
def MPS_ModuloOp : MPS_BinaryOp<"modulo", "elementwise modulo">;
def MPS_FloorModuloOp
    : MPS_BinaryOp<"floor_modulo", "elementwise floor modulo">;
def MPS_PowerOp : MPS_BinaryOp<"power", "elementwise power">;
def MPS_MinimumOp
    : MPS_BinaryOp<"minimum", "elementwise minimum", [Commutative]>;
def MPS_MaximumOp
    : MPS_BinaryOp<"maximum", "elementwise maximum", [Commutative]>;
def MPS_MinimumWithNaNPropagationOp
    : MPS_BinaryOp<"minimum_with_nan_propagation",
                   "elementwise minimum propagating NaN", [Commutative]>;
def MPS_MaximumWithNaNPropagationOp
    : MPS_BinaryOp<"maximum_with_nan_propagation",
                   "elementwise maximum propagating NaN", [Commutative]>;
def MPS_Atan2Op : MPS_BinaryOp<"atan2", "elementwise arctangent of y/x">;
def MPS_BitwiseAndOp
    : MPS_BinaryOp<"bitwise_and", "elementwise bitwise AND", [Commutative]>;
def MPS_BitwiseOrOp
    : MPS_BinaryOp<"bitwise_or", "elementwise bitwise OR", [Commutative]>;
def MPS_BitwiseXorOp
    : MPS_BinaryOp<"bitwise_xor", "elementwise bitwise XOR", [Commutative]>;
def MPS_BitwiseLeftShiftOp
    : MPS_BinaryOp<"bitwise_left_shift", "elementwise left shift">;
def MPS_BitwiseRightShiftOp
    : MPS_BinaryOp<"bitwise_right_shift", "elementwise right shift">;
def MPS_LogicalAndOp
    : MPS_BinaryOp<"logical_and", "elementwise logical AND", [Commutative]>;
def MPS_LogicalOrOp
    : MPS_BinaryOp<"logical_or", "elementwise logical OR", [Commutative]>;
def MPS_LogicalNandOp
    : MPS_BinaryOp<"logical_nand", "elementwise logical NAND", [Commutative]>;
def MPS_LogicalNorOp
    : MPS_BinaryOp<"logical_nor", "elementwise logical NOR", [Commutative]>;
def MPS_LogicalXorOp
    : MPS_BinaryOp<"logical_xor", "elementwise logical XOR", [Commutative]>;
def MPS_LogicalXnorOp
    : MPS_BinaryOp<"logical_xnor", "elementwise logical XNOR", [Commutative]>;

def MPS_EqualOp
    : MPS_ComparisonOp<"equal", "elementwise equality", [Commutative]>;
def MPS_NotEqualOp
    : MPS_ComparisonOp<"not_equal", "elementwise inequality", [Commutative]>;
def MPS_LessThanOp : MPS_ComparisonOp<"less_than", "elementwise x < y">;
def MPS_LessThanOrEqualToOp
    : MPS_ComparisonOp<"less_than_or_equal_to", "elementwise x <= y">;
def MPS_GreaterThanOp
    : MPS_ComparisonOp<"greater_than", "elementwise x > y">;
def MPS_GreaterThanOrEqualToOp
    : MPS_ComparisonOp<"greater_than_or_equal_to", "elementwise x >= y">;

def MPS_ComplexOp : MPS_Op<"complex",
    [Pure, ResultsBroadcastableShape, SameOperandsElementType]> {
  let summary = "complex tensor from real and imaginary parts";
  let arguments = (ins MPS_Tensor:$real_tensor, MPS_Tensor:$imaginary_tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SelectOp : MPS_Op<"select", [Pure, ResultsBroadcastableShape]> {
  let summary = "elementwise select";
  let description = [{
    Selects from `true_tensor` where `predicate` is true and from
    `false_tensor` otherwise.
  }];
  let arguments = (ins MPS_Tensor:$predicate, MPS_Tensor:$true_tensor,
                       MPS_Tensor:$false_tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ClampOp : MPS_Op<"clamp",
    [Pure, ResultsBroadcastableShape, SameOperandsAndResultElementType]> {
  let summary = "elementwise clamp between `min` and `max`";
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$min, MPS_Tensor:$max);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Activation ops (MPSGraphActivationOps.h)
//===----------------------------------------------------------------------===//

def MPS_ReLUOp : MPS_UnaryOp<"relu", "rectified linear unit">;
def MPS_SigmoidOp : MPS_UnaryOp<"sigmoid", "sigmoid activation">;

def MPS_ReLUGradientOp : MPS_Op<"relu_gradient", [Pure]> {
  let summary = "gradient of `mps.relu`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SigmoidGradientOp : MPS_Op<"sigmoid_gradient", [Pure]> {
  let summary = "gradient of `mps.sigmoid`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SoftMaxOp : MPS_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "softmax along `axis`";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SoftMaxGradientOp : MPS_Op<"softmax_gradient", [Pure]> {
  let summary = "gradient of `mps.softmax`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source,
                       I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_LeakyReLUOp : MPS_Op<"leaky_relu", [Pure]> {
  let summary = "leaky rectified linear unit";
  let description = [{
    Covers the `alpha:` overload with a scalar `alpha` tensor.
  }];
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$alpha);
  let results = (outs MPS_Tensor:$result);
}

def MPS_LeakyReLUGradientOp : MPS_Op<"leaky_relu_gradient", [Pure]> {
  let summary = "gradient of `mps.leaky_relu`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source,
                       MPS_Tensor:$alpha);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Call ops (MPSGraphCallOps.h)
//===----------------------------------------------------------------------===//

def MPS_CallOp : MPS_Op<"call"> {
  let summary = "invokes another executable";
  let description = [{
    `symbol_name` identifies the executable in
    `MPSGraphCompilationDescriptor.callables`. The result types are the
    expected output types of the executable.
  }];
  let arguments = (ins FlatSymbolRefAttr:$symbol_name,
                       Variadic<MPS_Tensor>:$input_tensors);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = [{
    $symbol_name `(` $input_tensors `)` attr-dict `:`
    functional-type($input_tensors, results)
  }];
}

//===----------------------------------------------------------------------===//
// Control flow ops (MPSGraphControlFlowOps.h)
//===----------------------------------------------------------------------===//

def MPS_IfOp : MPS_Op<"if", [RecursiveMemoryEffects]> {
  let summary = "if-then-else";
  let description = [{
    `predicate` is a single-element tensor. Each region is terminated by
    `mps.yield` with values matching the result types. `else_region` may only
    be empty if the op has no results.
  }];
  let arguments = (ins MPS_Tensor:$predicate);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                        MaxSizedRegion<1>:$else_region);
  let assemblyFormat = [{
    $predicate attr-dict `:` type($predicate) (`->` type($results)^)?
    $then_region (`else` $else_region^)?
  }];
}

def MPS_ForOp : MPS_Op<"for", [RecursiveMemoryEffects]> {
  let summary = "for loop over the half-open range [lower_bound, upper_bound)";
  let description = [{
    `lower_bound`, `upper_bound` and `step` are scalar tensors; `step` must be
    positive. The body block takes the index tensor followed by the iteration
    arguments and yields the next iteration arguments with `mps.yield`. The
    results are the iteration arguments after the last iteration.

    The `numberOfIterations:` overload is `lower_bound = 0`, `step = 1`.
  }];
  let arguments = (ins MPS_Tensor:$lower_bound, MPS_Tensor:$upper_bound,
                       MPS_Tensor:$step,
                       Variadic<MPS_Tensor>:$initial_body_arguments);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `(` $lower_bound `,` $upper_bound `,` $step `)`
    (`iter_args` `(` $initial_body_arguments^ `)`)? attr-dict `:`
    functional-type(operands, results) $body
  }];
}

def MPS_WhileOp : MPS_Op<"while", [RecursiveMemoryEffects]> {
  let summary = "while loop";
  let description = [{
    `before` receives the loop inputs (`initial_inputs` on the first
    iteration) and terminates with `mps.condition`. If the predicate is true,
    `after` runs with the forwarded values and yields the next loop inputs with
    `mps.yield`; otherwise the forwarded values are the results.
  }];
  let arguments = (ins Variadic<MPS_Tensor>:$initial_inputs);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);
  let assemblyFormat = [{
    `(` $initial_inputs `)` attr-dict `:`
    functional-type($initial_inputs, results) $before `do` $after
  }];
}

def MPS_ConditionOp : MPS_Op<"condition",
    [Pure, Terminator, HasParent<"WhileOp">]> {
  let summary = "terminator of the `before` region of `mps.while`";
  let arguments = (ins MPS_Tensor:$predicate,
                       Variadic<MPS_Tensor>:$results);
  let assemblyFormat = [{
    `(` $predicate `:` type($predicate) `)` attr-dict
    ($results^ `:` type($results))?
  }];
}

def MPS_YieldOp : MPS_Op<"yield",
    [Pure, Terminator, ParentOneOf<["IfOp", "ForOp", "WhileOp"]>]> {
  let summary = "terminator of `mps.if`, `mps.for` and `mps.while` regions";
  let arguments = (ins Variadic<MPS_Tensor>:$results);
  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

//===----------------------------------------------------------------------===//
// Convolution ops (MPSGraphConvolutionOps.h)
//===----------------------------------------------------------------------===//

def MPS_Convolution2DOp : MPS_Op<"convolution_2d", [Pure]> {
  let summary = "2D convolution";
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$weights),
                       Convolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_Convolution2DDataGradientOp
    : MPS_Op<"convolution_2d_data_gradient", [Pure]> {
  let summary = "gradient of `mps.convolution_2d` w.r.t. the source";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$weights),
                       Convolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_Convolution2DWeightsGradientOp
    : MPS_Op<"convolution_2d_weights_gradient", [Pure]> {
  let summary = "gradient of `mps.convolution_2d` w.r.t. the weights";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$source),
                       Convolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_Convolution3DOp : MPS_Op<"convolution_3d", [Pure]> {
  let summary = "3D convolution";
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$weights),
                       Convolution3DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_Convolution3DDataGradientOp
    : MPS_Op<"convolution_3d_data_gradient", [Pure]> {
  let summary = "gradient of `mps.convolution_3d` w.r.t. the source";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$weights),
                       Convolution3DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_Convolution3DWeightsGradientOp
    : MPS_Op<"convolution_3d_weights_gradient", [Pure]> {
  let summary = "gradient of `mps.convolution_3d` w.r.t. the weights";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$source),
                       Convolution3DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Convolution transpose ops (MPSGraphConvolutionTransposeOps.h)
//===----------------------------------------------------------------------===//

def MPS_ConvolutionTranspose2DOp
    : MPS_Op<"convolution_transpose_2d", [Pure]> {
  let summary = "2D convolution transpose";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$weights),
                       Convolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ConvolutionTranspose2DDataGradientOp
    : MPS_Op<"convolution_transpose_2d_data_gradient", [Pure]> {
  let summary = "gradient of `mps.convolution_transpose_2d` w.r.t. the source";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$weights),
                       Convolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ConvolutionTranspose2DWeightsGradientOp
    : MPS_Op<"convolution_transpose_2d_weights_gradient", [Pure]> {
  let summary =
      "gradient of `mps.convolution_transpose_2d` w.r.t. the weights";
  let description = [{
    The attributes describe the forward convolution. The result shape is
    `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$source),
                       Convolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Depthwise convolution ops (MPSGraphDepthwiseConvolutionOps.h)
//===----------------------------------------------------------------------===//

def MPS_DepthwiseConvolution2DOp
    : MPS_Op<"depthwise_convolution_2d", [Pure]> {
  let summary = "2D depthwise convolution";
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$weights),
                       DepthwiseConvolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DepthwiseConvolution2DDataGradientOp
    : MPS_Op<"depthwise_convolution_2d_data_gradient", [Pure]> {
  let summary = "gradient of `mps.depthwise_convolution_2d` w.r.t. the source";
  let description = [{
    The result shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$weights),
                       DepthwiseConvolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DepthwiseConvolution2DWeightsGradientOp
    : MPS_Op<"depthwise_convolution_2d_weights_gradient", [Pure]> {
  let summary =
      "gradient of `mps.depthwise_convolution_2d` w.r.t. the weights";
  let description = [{
    The result shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$source),
                       DepthwiseConvolution2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DepthwiseConvolution3DOp
    : MPS_Op<"depthwise_convolution_3d", [Pure]> {
  let summary = "3D depthwise convolution";
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$weights),
                       DepthwiseConvolution3DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DepthwiseConvolution3DDataGradientOp
    : MPS_Op<"depthwise_convolution_3d_data_gradient", [Pure]> {
  let summary = "gradient of `mps.depthwise_convolution_3d` w.r.t. the source";
  let description = [{
    The result shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$weights),
                       DepthwiseConvolution3DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DepthwiseConvolution3DWeightsGradientOp
    : MPS_Op<"depthwise_convolution_3d_weights_gradient", [Pure]> {
  let summary =
      "gradient of `mps.depthwise_convolution_3d` w.r.t. the weights";
  let description = [{
    The result shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$incoming_gradient,
                            MPS_Tensor:$source),
                       DepthwiseConvolution3DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Cumulative ops (MPSGraphCumulativeOps.h)
//===----------------------------------------------------------------------===//

def MPS_CumulativeSumOp : MPS_CumulativeOp<"cumulative_sum",
    "cumulative sum along `axis`">;
def MPS_CumulativeProductOp : MPS_CumulativeOp<"cumulative_product",
    "cumulative product along `axis`">;
def MPS_CumulativeMinimumOp : MPS_CumulativeOp<"cumulative_minimum",
    "cumulative minimum along `axis`">;
def MPS_CumulativeMaximumOp : MPS_CumulativeOp<"cumulative_maximum",
    "cumulative maximum along `axis`">;

//===----------------------------------------------------------------------===//
// Fourier transform ops (MPSGraphFourierTransformOps.h)
//===----------------------------------------------------------------------===//

def MPS_FastFourierTransformOp : MPS_Op<"fast_fourier_transform", [Pure]> {
  let summary = "complex-to-complex FFT over `axes`";
  let arguments = !con((ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$axes),
                       FFTDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_RealToHermiteanFFTOp : MPS_Op<"real_to_hermitean_fft", [Pure]> {
  let summary = "real-to-complex FFT over `axes`";
  let arguments = !con((ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$axes),
                       FFTDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_HermiteanToRealFFTOp : MPS_Op<"hermitean_to_real_fft", [Pure]> {
  let summary = "complex-to-real FFT over `axes`";
  let arguments = !con((ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$axes),
                       FFTDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Gather ops (MPSGraphGatherOps.h)
//===----------------------------------------------------------------------===//

def MPS_GatherOp : MPS_Op<"gather", [Pure]> {
  let summary = "gathers slices of `updates` along `axis`";
  let arguments = (ins MPS_Tensor:$updates, MPS_Tensor:$indices,
                       I64Attr:$axis, I64Attr:$batch_dimensions);
  let results = (outs MPS_Tensor:$result);
}

def MPS_GatherNDOp : MPS_Op<"gather_nd", [Pure]> {
  let summary = "gathers slices of `updates` at N-dimensional indices";
  let arguments = (ins MPS_Tensor:$updates, MPS_Tensor:$indices,
                       I64Attr:$batch_dimensions);
  let results = (outs MPS_Tensor:$result);
}

def MPS_GatherAlongAxisOp : MPS_Op<"gather_along_axis", [Pure]> {
  let summary = "gathers values of `updates` along `axis`";
  let description = [{
    `indices` has the same rank as `updates`.
  }];
  let arguments = (ins MPS_Tensor:$updates, MPS_Tensor:$indices,
                       I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Image to column ops (MPSGraphImToColOps.h)
//===----------------------------------------------------------------------===//

def MPS_ImToColOp : MPS_Op<"im_to_col", [Pure]> {
  let summary = "image to column";
  let arguments = !con((ins MPS_Tensor:$source), ImToColDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ColToImOp : MPS_Op<"col_to_im", [Pure]> {
  let summary = "column to image";
  let description = [{
    The result shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$source), ImToColDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Linear algebra ops (MPSGraphLinearAlgebraOps.h,
// MPSGraphMatrixMultiplicationOps.h, MPSGraphMatrixInverseOps.h)
//===----------------------------------------------------------------------===//

def MPS_BandPartOp : MPS_Op<"band_part", [Pure, SameOperandsAndResultType]> {
  let summary = "band part of a matrix";
  let description = [{
    Keeps `num_lower` sub-diagonals and `num_upper` super-diagonals; -1 keeps
    all of them.
  }];
  let arguments = (ins MPS_Tensor:$input, I64Attr:$num_lower,
                       I64Attr:$num_upper);
  let results = (outs MPS_Tensor:$result);
}

def MPS_MatrixMultiplicationOp : MPS_Op<"matrix_multiplication", [Pure]> {
  let summary = "matrix multiplication with batch broadcasting";
  let arguments = (ins MPS_Tensor:$primary, MPS_Tensor:$secondary);
  let results = (outs MPS_Tensor:$result);
}

def MPS_HammingDistanceOp : MPS_Op<"hamming_distance", [Pure]> {
  let summary = "Hamming distance between vectors in the last dimension";
  let arguments = (ins MPS_Tensor:$primary, MPS_Tensor:$secondary);
  let results = (outs MPS_Tensor:$result);
}

def MPS_InverseOp : MPS_Op<"inverse", [Pure, SameOperandsAndResultType]> {
  let summary = "matrix inverse";
  let arguments = (ins MPS_Tensor:$input);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ScaledDotProductAttentionOp : MPS_Op<"scaled_dot_product_attention",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "scaled dot-product attention";
  let description = [{
    Covers the descriptor overload (`MPSGraphSDPADescriptor`) and the
    `scale:` and `mask:scale:` overloads.
  }];
  let arguments = (ins MPS_Tensor:$query, MPS_Tensor:$key, MPS_Tensor:$value,
                       Optional<MPS_Tensor>:$mask,
                       Optional<MPS_Tensor>:$sinks,
                       F32Attr:$scale,
                       DefaultValuedAttr<BoolAttr, "false">:$is_causal);
  let results = (outs MPS_Tensor:$result);
  let assemblyFormat = ?;
}

//===----------------------------------------------------------------------===//
// Loss ops (MPSGraphLossOps.h)
//===----------------------------------------------------------------------===//

def MPS_SoftMaxCrossEntropyOp : MPS_Op<"softmax_cross_entropy", [Pure]> {
  let summary = "softmax cross-entropy loss";
  let arguments = (ins MPS_Tensor:$source, MPS_Tensor:$labels, I64Attr:$axis,
                       MPS_LossReductionTypeAttr:$reduction_type);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SoftMaxCrossEntropyGradientOp
    : MPS_Op<"softmax_cross_entropy_gradient", [Pure]> {
  let summary = "gradient of `mps.softmax_cross_entropy`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source,
                       MPS_Tensor:$labels, I64Attr:$axis,
                       MPS_LossReductionTypeAttr:$reduction_type);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Non-maximum suppression ops (MPSGraphNonMaximumSuppressionOps.h)
//===----------------------------------------------------------------------===//

def MPS_NonMaximumSuppressionOp : MPS_Op<"non_maximum_suppression", [Pure]> {
  let summary = "non-maximum suppression";
  let description = [{
    `boxes` is `[N,B,4]` f32, `scores` is `[N,B,K]` f32 and the optional
    `class_indices` is `[N,B]` i32.
  }];
  let arguments = (ins MPS_Tensor:$boxes, MPS_Tensor:$scores,
                       Optional<MPS_Tensor>:$class_indices,
                       F32Attr:$iou_threshold, F32Attr:$score_threshold,
                       BoolAttr:$per_class_suppression,
                       MPS_NonMaximumSuppressionCoordinateModeAttr:
                           $coordinate_mode);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Non-zero ops (MPSGraphNonZeroOps.h)
//===----------------------------------------------------------------------===//

def MPS_NonZeroIndicesOp : MPS_Op<"non_zero_indices", [Pure]> {
  let summary = "indices of the non-zero elements";
  let description = [{
    The result is an i32 tensor of shape `[number_of_nonzeros, input_rank]`.
  }];
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Normalization ops (MPSGraphNormalizationOps.h)
//===----------------------------------------------------------------------===//

def MPS_MeanOp : MPS_Op<"mean", [Pure, SameOperandsAndResultElementType]> {
  let summary = "mean over `axes`";
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$axes);
  let results = (outs MPS_Tensor:$result);
}

def MPS_VarianceOp : MPS_Op<"variance",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "variance over `axes`";
  let description = [{
    If `mean` is absent, it is computed from `tensor`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, Optional<MPS_Tensor>:$mean,
                       DenseI64ArrayAttr:$axes);
  let results = (outs MPS_Tensor:$result);
}

def MPS_NormalizeOp : MPS_Op<"normalize",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "batch normalization";
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$mean,
                       MPS_Tensor:$variance, Optional<MPS_Tensor>:$gamma,
                       Optional<MPS_Tensor>:$beta, F32Attr:$epsilon);
  let results = (outs MPS_Tensor:$result);
  let assemblyFormat = ?;
}

def MPS_NormalizationGammaGradientOp
    : MPS_Op<"normalization_gamma_gradient", [Pure]> {
  let summary = "gradient of `mps.normalize` w.r.t. gamma";
  let arguments = (ins MPS_Tensor:$incoming_gradient, MPS_Tensor:$source,
                       MPS_Tensor:$mean, MPS_Tensor:$variance,
                       DenseI64ArrayAttr:$reduction_axes, F32Attr:$epsilon);
  let results = (outs MPS_Tensor:$result);
}

def MPS_NormalizationBetaGradientOp
    : MPS_Op<"normalization_beta_gradient", [Pure]> {
  let summary = "gradient of `mps.normalize` w.r.t. beta";
  let arguments = (ins MPS_Tensor:$incoming_gradient, MPS_Tensor:$source,
                       DenseI64ArrayAttr:$reduction_axes);
  let results = (outs MPS_Tensor:$result);
}

def MPS_NormalizationGradientOp : MPS_Op<"normalization_gradient",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "gradient of `mps.normalize` w.r.t. the source";
  let arguments = (ins MPS_Tensor:$incoming_gradient, MPS_Tensor:$source,
                       MPS_Tensor:$mean, MPS_Tensor:$variance,
                       Optional<MPS_Tensor>:$gamma,
                       Optional<MPS_Tensor>:$gamma_gradient,
                       Optional<MPS_Tensor>:$beta_gradient,
                       DenseI64ArrayAttr:$reduction_axes, F32Attr:$epsilon);
  let results = (outs MPS_Tensor:$result);
  let assemblyFormat = ?;
}

//===----------------------------------------------------------------------===//
// One-hot ops (MPSGraphOneHotOps.h)
//===----------------------------------------------------------------------===//

def MPS_OneHotOp : MPS_Op<"one_hot", [Pure]> {
  let summary = "one-hot encoding";
  let description = [{
    The one-hot dimension is inserted at `axis`; its size in the result type
    is `depth`.
  }];
  let arguments = (ins MPS_Tensor:$indices,
                       DefaultValuedAttr<I64Attr, "-1">:$axis,
                       DefaultValuedAttr<F64Attr, "1.0">:$on_value,
                       DefaultValuedAttr<F64Attr, "0.0">:$off_value);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Optimizer ops (MPSGraphOptimizerOps.h)
//===----------------------------------------------------------------------===//

def MPS_StochasticGradientDescentOp
    : MPS_Op<"stochastic_gradient_descent", [Pure]> {
  let summary = "stochastic gradient descent update";
  let description = [{
    Returns `values - learning_rate * gradient`.
  }];
  let arguments = (ins MPS_Tensor:$learning_rate, MPS_Tensor:$values,
                       MPS_Tensor:$gradient);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ApplyStochasticGradientDescentOp
    : MPS_Op<"apply_stochastic_gradient_descent"> {
  let summary = "stochastic gradient descent update of a variable";
  let arguments = (ins MPS_Tensor:$learning_rate,
                       Arg<MPS_Tensor, "", [MemRead, MemWrite]>:$variable,
                       MPS_Tensor:$gradient);
}

def MPS_AdamOp : MPS_Op<"adam", [Pure]> {
  let summary = "Adam update";
  let description = [{
    `new_maximum_velocity` is present iff `maximum_velocity` is.
  }];
  let arguments = (ins MPS_Tensor:$learning_rate, MPS_Tensor:$beta1,
                       MPS_Tensor:$beta2, MPS_Tensor:$epsilon,
                       MPS_Tensor:$beta1_power, MPS_Tensor:$beta2_power,
                       MPS_Tensor:$values, MPS_Tensor:$momentum,
                       MPS_Tensor:$velocity,
                       Optional<MPS_Tensor>:$maximum_velocity,
                       MPS_Tensor:$gradient);
  let results = (outs MPS_Tensor:$update, MPS_Tensor:$new_momentum,
                      MPS_Tensor:$new_velocity,
                      Optional<MPS_Tensor>:$new_maximum_velocity);
}

def MPS_AdamWithCurrentLearningRateOp
    : MPS_Op<"adam_with_current_learning_rate", [Pure]> {
  let summary = "Adam update with a precomputed learning rate";
  let description = [{
    `new_maximum_velocity` is present iff `maximum_velocity` is.
  }];
  let arguments = (ins MPS_Tensor:$current_learning_rate, MPS_Tensor:$beta1,
                       MPS_Tensor:$beta2, MPS_Tensor:$epsilon,
                       MPS_Tensor:$values, MPS_Tensor:$momentum,
                       MPS_Tensor:$velocity,
                       Optional<MPS_Tensor>:$maximum_velocity,
                       MPS_Tensor:$gradient);
  let results = (outs MPS_Tensor:$update, MPS_Tensor:$new_momentum,
                      MPS_Tensor:$new_velocity,
                      Optional<MPS_Tensor>:$new_maximum_velocity);
}

//===----------------------------------------------------------------------===//
// Pooling ops (MPSGraphPoolingOps.h)
//===----------------------------------------------------------------------===//

def MPS_MaxPooling2DOp : MPS_Op<"max_pooling_2d", [Pure]> {
  let summary = "2D max pooling";
  let arguments = !con((ins MPS_Tensor:$source), Pooling2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_MaxPooling2DReturnIndicesOp
    : MPS_Op<"max_pooling_2d_return_indices", [Pure]> {
  let summary = "2D max pooling returning the indices of the maxima";
  let arguments = !con((ins MPS_Tensor:$source), Pooling2DDescriptor,
                       (ins MPS_PoolingReturnIndicesModeAttr:
                            $return_indices_mode));
  let results = (outs MPS_Tensor:$result, MPS_Tensor:$indices);
}

def MPS_MaxPooling2DGradientOp : MPS_Op<"max_pooling_2d_gradient", [Pure]> {
  let summary = "gradient of `mps.max_pooling_2d`";
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$source),
                       Pooling2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_MaxPooling2DGradientWithIndicesOp
    : MPS_Op<"max_pooling_2d_gradient_with_indices", [Pure]> {
  let summary = "gradient of `mps.max_pooling_2d` from the pooling indices";
  let description = [{
    `indices` is the result of `mps.max_pooling_2d_return_indices`. The result
    shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$indices),
                       Pooling2DDescriptor,
                       (ins MPS_PoolingReturnIndicesModeAttr:
                            $return_indices_mode));
  let results = (outs MPS_Tensor:$result);
}

def MPS_AvgPooling2DOp : MPS_Op<"avg_pooling_2d", [Pure]> {
  let summary = "2D average pooling";
  let arguments = !con((ins MPS_Tensor:$source), Pooling2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_AvgPooling2DGradientOp : MPS_Op<"avg_pooling_2d_gradient", [Pure]> {
  let summary = "gradient of `mps.avg_pooling_2d`";
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$source),
                       Pooling2DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_MaxPooling4DOp : MPS_Op<"max_pooling_4d", [Pure]> {
  let summary = "4D max pooling";
  let arguments = !con((ins MPS_Tensor:$source), Pooling4DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_MaxPooling4DReturnIndicesOp
    : MPS_Op<"max_pooling_4d_return_indices", [Pure]> {
  let summary = "4D max pooling returning the indices of the maxima";
  let arguments = !con((ins MPS_Tensor:$source), Pooling4DDescriptor,
                       (ins MPS_PoolingReturnIndicesModeAttr:
                            $return_indices_mode));
  let results = (outs MPS_Tensor:$result, MPS_Tensor:$indices);
}

def MPS_MaxPooling4DGradientOp : MPS_Op<"max_pooling_4d_gradient", [Pure]> {
  let summary = "gradient of `mps.max_pooling_4d`";
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$source),
                       Pooling4DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_MaxPooling4DGradientWithIndicesOp
    : MPS_Op<"max_pooling_4d_gradient_with_indices", [Pure]> {
  let summary = "gradient of `mps.max_pooling_4d` from the pooling indices";
  let description = [{
    `indices` is the result of `mps.max_pooling_4d_return_indices`. The result
    shape is `outputShape`.
  }];
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$indices),
                       Pooling4DDescriptor,
                       (ins MPS_PoolingReturnIndicesModeAttr:
                            $return_indices_mode));
  let results = (outs MPS_Tensor:$result);
}

def MPS_AvgPooling4DOp : MPS_Op<"avg_pooling_4d", [Pure]> {
  let summary = "4D average pooling";
  let arguments = !con((ins MPS_Tensor:$source), Pooling4DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_AvgPooling4DGradientOp : MPS_Op<"avg_pooling_4d_gradient", [Pure]> {
  let summary = "gradient of `mps.avg_pooling_4d`";
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$source),
                       Pooling4DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_L2NormPooling4DOp : MPS_Op<"l2_norm_pooling_4d", [Pure]> {
  let summary = "4D L2-norm pooling";
  let arguments = !con((ins MPS_Tensor:$source), Pooling4DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_L2NormPooling4DGradientOp
    : MPS_Op<"l2_norm_pooling_4d_gradient", [Pure]> {
  let summary = "gradient of `mps.l2_norm_pooling_4d`";
  let arguments = !con((ins MPS_Tensor:$gradient, MPS_Tensor:$source),
                       Pooling4DDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Quantization ops (MPSGraphQuantizationOps.h)
//===----------------------------------------------------------------------===//

def MPS_QuantizeOp : MPS_Op<"quantize", [Pure]> {
  let summary = "affine quantization";
  let description = [{
    Covers the scalar `scale:zeroPoint:` overload (scalar tensors, no `axis`)
    and the per-axis `scaleTensor:` overloads. The result element type is
    `dataType`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$scale,
                       MPS_Tensor:$zero_point, OptionalAttr<I64Attr>:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DequantizeOp : MPS_Op<"dequantize", [Pure]> {
  let summary = "affine dequantization";
  let description = [{
    Covers the scalar `scale:zeroPoint:` overload (scalar tensors, no `axis`)
    and the `scaleTensor:` overloads. A missing `zero_point` is 0. The result
    element type is `dataType`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$scale,
                       Optional<MPS_Tensor>:$zero_point,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DequantizeLUTOp : MPS_Op<"dequantize_lut", [Pure]> {
  let summary = "lookup-table dequantization";
  let description = [{
    The `LUTTensor:` overloads of `dequantize`. With `axis`, `lut` holds one
    table per element of that axis.
  }];
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$lut,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Random ops (MPSGraphRandomOps.h)
//===----------------------------------------------------------------------===//

def MPS_RandomPhiloxStateOp : MPS_Op<"random_philox_state", [Pure]> {
  let summary = "Philox random state tensor";
  let description = [{
    Initialized either from `seed` or from `counter_low`, `counter_high` and
    `key`.
  }];
  let arguments = (ins OptionalAttr<I64Attr>:$seed,
                       OptionalAttr<I64Attr>:$counter_low,
                       OptionalAttr<I64Attr>:$counter_high,
                       OptionalAttr<I64Attr>:$key);
  let results = (outs MPS_Tensor:$result);
}

def MPS_RandomOp : MPS_Op<"random"> {
  let summary = "random tensor";
  let description = [{
    Covers `randomTensor` and `randomUniformTensor` (uniform in [0, 1)) without
    state. All ops with the same `seed` yield the same stream of values. The
    result type is the shape and `dataType`.
  }];
  let arguments = !con(RandomOpDescriptor, (ins OptionalAttr<I64Attr>:$seed));
  let results = (outs MPS_Tensor:$result);
}

def MPS_RandomWithStateOp : MPS_Op<"random_with_state", [Pure]> {
  let summary = "random tensor from an explicit state";
  let description = [{
    Covers the `stateTensor:` overloads of `randomTensor` and
    `randomUniformTensor`. Returns the random values and the updated state.
  }];
  let arguments = !con((ins MPS_Tensor:$state), RandomOpDescriptor);
  let results = (outs MPS_Tensor:$result, MPS_Tensor:$new_state);
}

def MPS_DropoutOp : MPS_Op<"dropout"> {
  let summary = "dropout";
  let description = [{
    Sets elements to 0 with probability `rate`, a scalar tensor.
  }];
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$rate);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Reduction ops (MPSGraphReductionOps.h)
//===----------------------------------------------------------------------===//

def MPS_ReductionSumOp : MPS_ReductionOp<"reduction_sum", "sum reduction">;
def MPS_ReductionProductOp
    : MPS_ReductionOp<"reduction_product", "product reduction">;
def MPS_ReductionMaximumOp
    : MPS_ReductionOp<"reduction_maximum", "maximum reduction">;
def MPS_ReductionMinimumOp
    : MPS_ReductionOp<"reduction_minimum", "minimum reduction">;
def MPS_ReductionMaximumPropagateNaNOp : MPS_ReductionOp<
    "reduction_maximum_propagate_nan", "maximum reduction propagating NaN">;
def MPS_ReductionMinimumPropagateNaNOp : MPS_ReductionOp<
    "reduction_minimum_propagate_nan", "minimum reduction propagating NaN">;
def MPS_ReductionAndOp
    : MPS_ReductionOp<"reduction_and", "logical AND reduction">;
def MPS_ReductionOrOp
    : MPS_ReductionOp<"reduction_or", "logical OR reduction">;

def MPS_ReductionArgMaximumOp : MPS_Op<"reduction_arg_maximum", [Pure]> {
  let summary = "index of the maximum along `axis`";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ReductionArgMinimumOp : MPS_Op<"reduction_arg_minimum", [Pure]> {
  let summary = "index of the minimum along `axis`";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Resize ops (MPSGraphResizeOps.h)
//===----------------------------------------------------------------------===//

def MPS_ResizeOp : MPS_Op<"resize", [Pure]> {
  let summary = "resize";
  let description = [{
    Covers `resize`, `resizeNearest` and `resizeBilinear` with `size` or
    `sizeTensor`; the result shape is the target size. `nearest_rounding_mode`
    applies to nearest sampling; if absent, `roundPreferCeil` is used.
  }];
  let arguments = (ins MPS_Tensor:$images, MPS_ResizeModeAttr:$mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode,
                       BoolAttr:$center_result, BoolAttr:$align_corners,
                       OptionalAttr<MPS_TensorNamedDataLayoutAttr>:$layout);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ResizeWithScaleOffsetOp : MPS_Op<"resize_with_scale_offset", [Pure]> {
  let summary = "resize with a combined scale and offset tensor";
  let description = [{
    The `scaleOffsetTensor:` overloads. `scale_offset` is
    `[scaleY, scaleX, offsetY, offsetX]`.
  }];
  let arguments = (ins MPS_Tensor:$images, MPS_Tensor:$scale_offset,
                       MPS_ResizeModeAttr:$mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode,
                       MPS_TensorNamedDataLayoutAttr:$layout);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ResizeWithScaleAndOffsetOp
    : MPS_Op<"resize_with_scale_and_offset", [Pure]> {
  let summary = "resize with per-dimension scale and offset tensors";
  let description = [{
    The `scaleTensor:offsetTensor:` overloads.
  }];
  let arguments = (ins MPS_Tensor:$images, MPS_Tensor:$scale,
                       MPS_Tensor:$offset, MPS_ResizeModeAttr:$mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ResizeGradientOp : MPS_Op<"resize_gradient", [Pure]> {
  let summary = "gradient of `mps.resize`";
  let description = [{
    `input` is the forward input; the result has its shape.
  }];
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$input,
                       MPS_ResizeModeAttr:$mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode,
                       BoolAttr:$center_result, BoolAttr:$align_corners,
                       MPS_TensorNamedDataLayoutAttr:$layout);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ResizeGradientWithScaleOffsetOp
    : MPS_Op<"resize_gradient_with_scale_offset", [Pure]> {
  let summary = "gradient of `mps.resize_with_scale_offset`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$input,
                       MPS_Tensor:$scale_offset, MPS_ResizeModeAttr:$mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode,
                       MPS_TensorNamedDataLayoutAttr:$layout);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ResizeGradientWithScaleAndOffsetOp
    : MPS_Op<"resize_gradient_with_scale_and_offset", [Pure]> {
  let summary = "gradient of `mps.resize_with_scale_and_offset`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$input,
                       MPS_Tensor:$scale, MPS_Tensor:$offset,
                       MPS_ResizeModeAttr:$mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Sample grid ops (MPSGraphSampleGridOps.h)
//===----------------------------------------------------------------------===//

def MPS_SampleGridOp : MPS_Op<"sample_grid", [Pure]> {
  let summary = "samples `source` at the given coordinates";
  let description = [{
    Covers the `samplingMode:` and `nearestRoundingMode:` overloads; the
    latter is `sampling_mode = nearest`.
  }];
  let arguments = (ins MPS_Tensor:$source, MPS_Tensor:$coordinates,
                       MPS_TensorNamedDataLayoutAttr:$layout,
                       BoolAttr:$normalize_coordinates,
                       BoolAttr:$relative_coordinates,
                       BoolAttr:$align_corners,
                       MPS_PaddingModeAttr:$padding_mode,
                       MPS_ResizeModeAttr:$sampling_mode,
                       OptionalAttr<MPS_ResizeNearestRoundingModeAttr>:
                           $nearest_rounding_mode,
                       F64Attr:$constant_value);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Scatter ops (MPSGraphScatterNDOps.h)
//===----------------------------------------------------------------------===//

def MPS_ScatterOp : MPS_Op<"scatter", [Pure]> {
  let summary = "scatters `updates` along `axis` into a new tensor";
  let description = [{
    The result shape is `shape`.
  }];
  let arguments = (ins MPS_Tensor:$updates, MPS_Tensor:$indices,
                       I64Attr:$axis, MPS_ScatterModeAttr:$mode);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ScatterWithDataOp : MPS_Op<"scatter_with_data", [Pure]> {
  let summary = "scatters `updates` along `axis` into `data`";
  let arguments = (ins MPS_Tensor:$data, MPS_Tensor:$updates,
                       MPS_Tensor:$indices, I64Attr:$axis,
                       MPS_ScatterModeAttr:$mode);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ScatterNDOp : MPS_Op<"scatter_nd", [Pure]> {
  let summary = "scatters slices of `updates` at N-dimensional indices";
  let description = [{
    The result shape is `shape`.
  }];
  let arguments = (ins MPS_Tensor:$updates, MPS_Tensor:$indices,
                       I64Attr:$batch_dimensions, MPS_ScatterModeAttr:$mode);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ScatterNDWithDataOp : MPS_Op<"scatter_nd_with_data", [Pure]> {
  let summary = "scatters slices of `updates` at N-dimensional indices into "
                "`data`";
  let arguments = (ins MPS_Tensor:$data, MPS_Tensor:$updates,
                       MPS_Tensor:$indices, I64Attr:$batch_dimensions,
                       MPS_ScatterModeAttr:$mode);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ScatterAlongAxisOp : MPS_Op<"scatter_along_axis", [Pure]> {
  let summary = "scatters values of `updates` along `axis` into a new tensor";
  let description = [{
    The result shape is `shape`.
  }];
  let arguments = (ins MPS_Tensor:$updates, MPS_Tensor:$indices,
                       I64Attr:$axis, MPS_ScatterModeAttr:$mode);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ScatterAlongAxisWithDataOp
    : MPS_Op<"scatter_along_axis_with_data", [Pure]> {
  let summary = "scatters values of `updates` along `axis` into `data`";
  let arguments = (ins MPS_Tensor:$data, MPS_Tensor:$updates,
                       MPS_Tensor:$indices, I64Attr:$axis,
                       MPS_ScatterModeAttr:$mode);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Sort ops (MPSGraphSortOps.h)
//===----------------------------------------------------------------------===//

def MPS_SortOp : MPS_Op<"sort", [Pure, SameOperandsAndResultType]> {
  let summary = "sorts along `axis`";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis,
                       DefaultValuedAttr<BoolAttr, "false">:$descending);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ArgSortOp : MPS_Op<"arg_sort", [Pure, SameOperandsAndResultShape]> {
  let summary = "indices that sort along `axis`";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis,
                       DefaultValuedAttr<BoolAttr, "false">:$descending);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Sparse ops (MPSGraphSparseOps.h)
//===----------------------------------------------------------------------===//

def MPS_SparseTensorOp : MPS_Op<"sparse_tensor", [Pure]> {
  let summary = "tensor from sparse values and indices";
  let description = [{
    Covers the descriptor and `sparseTensorWithType:` overloads. The result
    type is the dense `shape` and `dataType`.
  }];
  let arguments = (ins MPS_Tensor:$values, MPS_Tensor:$indices0,
                       MPS_Tensor:$indices1,
                       MPS_SparseStorageTypeAttr:$storage_type);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Stencil ops (MPSGraphStencilOps.h)
//===----------------------------------------------------------------------===//

def MPS_StencilOp : MPS_Op<"stencil", [Pure]> {
  let summary = "weighted reduction over a window of the last 4 dimensions";
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$weights),
                       StencilOpDescriptor);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Tensor shape ops (MPSGraphTensorShapeOps.h)
//===----------------------------------------------------------------------===//

def MPS_ReshapeOp : MPS_Op<"reshape", [Pure, SameOperandsAndResultElementType]> {
  let summary = "reshape to the result shape";
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_TransposeOp : MPS_Op<"transpose",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "permutes dimensions";
  let description = [{
    Also covers `transposeTensor(_:dimension:withDimension:)`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$permutation);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SliceOp : MPS_Op<"slice", [Pure, SameOperandsAndResultElementType]> {
  let summary = "strided slice";
  let description = [{
    Bit `i` of `start_mask` (`end_mask`) ignores `starts[i]` (`ends[i]`) and
    uses the full range; bit `i` of `squeeze_mask` removes dimension `i`.
    Also covers `sliceTensor(_:dimension:start:length:)`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$starts,
                       DenseI64ArrayAttr:$ends, DenseI64ArrayAttr:$strides,
                       DefaultValuedAttr<I32Attr, "0">:$start_mask,
                       DefaultValuedAttr<I32Attr, "0">:$end_mask,
                       DefaultValuedAttr<I32Attr, "0">:$squeeze_mask);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DynamicSliceOp : MPS_Op<"dynamic_slice",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "strided slice with tensor bounds";
  let description = [{
    `mps.slice` with `start`, `end` and `stride` as 1D tensors. The
    `start:sizeTensor:` overload is `end = start + size`, `stride = 1`.
  }];
  let arguments = (ins MPS_Tensor:$tensor, MPS_Tensor:$start,
                       MPS_Tensor:$end, MPS_Tensor:$stride,
                       DefaultValuedAttr<I32Attr, "0">:$start_mask,
                       DefaultValuedAttr<I32Attr, "0">:$end_mask,
                       DefaultValuedAttr<I32Attr, "0">:$squeeze_mask);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SliceGradientOp : MPS_Op<"slice_gradient", [Pure]> {
  let summary = "gradient of `mps.slice`";
  let description = [{
    The result shape is the forward input shape (`fwdInShapeTensor`).
  }];
  let arguments = (ins MPS_Tensor:$input_gradient, DenseI64ArrayAttr:$starts,
                       DenseI64ArrayAttr:$ends, DenseI64ArrayAttr:$strides,
                       DefaultValuedAttr<I32Attr, "0">:$start_mask,
                       DefaultValuedAttr<I32Attr, "0">:$end_mask,
                       DefaultValuedAttr<I32Attr, "0">:$squeeze_mask);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DynamicSliceGradientOp : MPS_Op<"dynamic_slice_gradient", [Pure]> {
  let summary = "gradient of `mps.dynamic_slice`";
  let description = [{
    The result shape is the forward input shape (`fwdInShapeTensor`).
  }];
  let arguments = (ins MPS_Tensor:$input_gradient, MPS_Tensor:$start,
                       MPS_Tensor:$end, MPS_Tensor:$stride,
                       DefaultValuedAttr<I32Attr, "0">:$start_mask,
                       DefaultValuedAttr<I32Attr, "0">:$end_mask,
                       DefaultValuedAttr<I32Attr, "0">:$squeeze_mask);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SliceUpdateOp : MPS_Op<"slice_update", [Pure]> {
  let summary = "writes `update` into a strided slice of `data`";
  let arguments = (ins MPS_Tensor:$data, MPS_Tensor:$update,
                       DenseI64ArrayAttr:$starts, DenseI64ArrayAttr:$ends,
                       DenseI64ArrayAttr:$strides,
                       DefaultValuedAttr<I32Attr, "0">:$start_mask,
                       DefaultValuedAttr<I32Attr, "0">:$end_mask,
                       DefaultValuedAttr<I32Attr, "0">:$squeeze_mask);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DynamicSliceUpdateOp : MPS_Op<"dynamic_slice_update", [Pure]> {
  let summary = "`mps.slice_update` with tensor bounds";
  let arguments = (ins MPS_Tensor:$data, MPS_Tensor:$update,
                       MPS_Tensor:$starts, MPS_Tensor:$ends,
                       MPS_Tensor:$strides,
                       DefaultValuedAttr<I32Attr, "0">:$start_mask,
                       DefaultValuedAttr<I32Attr, "0">:$end_mask,
                       DefaultValuedAttr<I32Attr, "0">:$squeeze_mask);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ConcatOp : MPS_Op<"concat", [Pure]> {
  let summary = "concatenates along `dimension`";
  let description = [{
    With `interleave`, elements of the inputs alternate along `dimension`.
    Also covers `concatTensor(_:with:dimension:)`.
  }];
  let arguments = (ins Variadic<MPS_Tensor>:$tensors, I64Attr:$dimension,
                       DefaultValuedAttr<BoolAttr, "false">:$interleave);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SplitOp : MPS_Op<"split", [Pure]> {
  let summary = "splits along `axis`";
  let description = [{
    The split sizes are the result sizes along `axis`. Covers the
    `numSplits:` and `splitSizes:` overloads.
  }];
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis);
  let results = (outs Variadic<MPS_Tensor>:$results);
}

def MPS_StackOp : MPS_Op<"stack", [Pure]> {
  let summary = "stacks along a new dimension `axis`";
  let arguments = (ins Variadic<MPS_Tensor>:$tensors, I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_TileOp : MPS_Op<"tile", [Pure, SameOperandsAndResultElementType]> {
  let summary = "tiles `multiplier` times along each dimension";
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$multiplier);
  let results = (outs MPS_Tensor:$result);
}

def MPS_TileGradientOp : MPS_Op<"tile_gradient", [Pure]> {
  let summary = "gradient of `mps.tile`";
  let arguments = (ins MPS_Tensor:$incoming_gradient, MPS_Tensor:$source,
                       DenseI64ArrayAttr:$multiplier);
  let results = (outs MPS_Tensor:$result);
}

def MPS_PadOp : MPS_Op<"pad", [Pure, SameOperandsAndResultElementType]> {
  let summary = "pads each dimension";
  let arguments = (ins MPS_Tensor:$tensor,
                       MPS_PaddingModeAttr:$padding_mode,
                       DenseI64ArrayAttr:$left_padding,
                       DenseI64ArrayAttr:$right_padding,
                       F64Attr:$constant_value);
  let results = (outs MPS_Tensor:$result);
}

def MPS_PadGradientOp : MPS_Op<"pad_gradient", [Pure]> {
  let summary = "gradient of `mps.pad`";
  let arguments = (ins MPS_Tensor:$incoming_gradient, MPS_Tensor:$source,
                       MPS_PaddingModeAttr:$padding_mode,
                       DenseI64ArrayAttr:$left_padding,
                       DenseI64ArrayAttr:$right_padding);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SpaceToDepth2DOp : MPS_Op<"space_to_depth_2d", [Pure]> {
  let summary = "moves 2D spatial blocks into the depth dimension";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$width_axis,
                       I64Attr:$height_axis, I64Attr:$depth_axis,
                       I64Attr:$block_size, BoolAttr:$use_pixel_shuffle_order);
  let results = (outs MPS_Tensor:$result);
}

def MPS_DepthToSpace2DOp : MPS_Op<"depth_to_space_2d", [Pure]> {
  let summary = "moves the depth dimension into 2D spatial blocks";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$width_axis,
                       I64Attr:$height_axis, I64Attr:$depth_axis,
                       I64Attr:$block_size, BoolAttr:$use_pixel_shuffle_order);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SpaceToBatchOp : MPS_Op<"space_to_batch", [Pure]> {
  let summary = "moves spatial blocks into the batch dimension";
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$spatial_axes,
                       I64Attr:$batch_axis,
                       DenseI64ArrayAttr:$block_dimensions,
                       BoolAttr:$use_pixel_shuffle_order);
  let results = (outs MPS_Tensor:$result);
}

def MPS_BatchToSpaceOp : MPS_Op<"batch_to_space", [Pure]> {
  let summary = "moves the batch dimension into spatial blocks";
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$spatial_axes,
                       I64Attr:$batch_axis,
                       DenseI64ArrayAttr:$block_dimensions,
                       BoolAttr:$use_pixel_shuffle_order);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ReverseOp : MPS_Op<"reverse", [Pure, SameOperandsAndResultType]> {
  let summary = "reverses along `axes`";
  let description = [{
    If `axes` is absent, all dimensions are reversed.
  }];
  let arguments = (ins MPS_Tensor:$tensor,
                       OptionalAttr<DenseI64ArrayAttr>:$axes);
  let results = (outs MPS_Tensor:$result);
}

def MPS_Flatten2DOp : MPS_Op<"flatten_2d",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "flattens to rank 2 around `axis`";
  let arguments = (ins MPS_Tensor:$tensor, I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

def MPS_BroadcastOp : MPS_Op<"broadcast",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "broadcasts to the result shape";
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ShapeOfOp : MPS_Op<"shape_of", [Pure]> {
  let summary = "shape of a tensor";
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_CastOp : MPS_Op<"cast", [Pure, SameOperandsAndResultShape]> {
  let summary = "converts to the result element type";
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ReinterpretCastOp : MPS_Op<"reinterpret_cast", [Pure]> {
  let summary = "reinterprets the bits as the result element type";
  let arguments = (ins MPS_Tensor:$tensor);
  let results = (outs MPS_Tensor:$result);
}

def MPS_SqueezeOp : MPS_Op<"squeeze",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "removes size-1 dimensions";
  let description = [{
    Removes the dimensions in `axes`, or all size-1 dimensions if `axes` is
    absent.
  }];
  let arguments = (ins MPS_Tensor:$tensor,
                       OptionalAttr<DenseI64ArrayAttr>:$axes);
  let results = (outs MPS_Tensor:$result);
}

def MPS_ExpandDimsOp : MPS_Op<"expand_dims",
    [Pure, SameOperandsAndResultElementType]> {
  let summary = "inserts size-1 dimensions at `axes`";
  let arguments = (ins MPS_Tensor:$tensor, DenseI64ArrayAttr:$axes);
  let results = (outs MPS_Tensor:$result);
}

def MPS_CoordinateOp : MPS_Op<"coordinate", [Pure]> {
  let summary = "coordinate of each element along `axis`";
  let description = [{
    Each element of the result is its coordinate along `axis`; negative values
    wrap around. The result shape is `shape`.
  }];
  let arguments = (ins I64Attr:$axis);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// Top-k ops (MPSGraphTopKOps.h)
//===----------------------------------------------------------------------===//

def MPS_TopKOp : MPS_Op<"top_k", [Pure]> {
  let summary = "`k` largest values along `axis` and their indices";
  let arguments = (ins MPS_Tensor:$source,
                       DefaultValuedAttr<I64Attr, "-1">:$axis, I64Attr:$k);
  let results = (outs MPS_Tensor:$values, MPS_Tensor:$indices);
}

def MPS_BottomKOp : MPS_Op<"bottom_k", [Pure]> {
  let summary = "`k` smallest values along `axis` and their indices";
  let arguments = (ins MPS_Tensor:$source, I64Attr:$axis, I64Attr:$k);
  let results = (outs MPS_Tensor:$values, MPS_Tensor:$indices);
}

def MPS_TopKGradientOp : MPS_Op<"top_k_gradient", [Pure]> {
  let summary = "gradient of `mps.top_k`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source,
                       DefaultValuedAttr<I64Attr, "-1">:$axis, I64Attr:$k);
  let results = (outs MPS_Tensor:$result);
}

def MPS_BottomKGradientOp : MPS_Op<"bottom_k_gradient", [Pure]> {
  let summary = "gradient of `mps.bottom_k`";
  let arguments = (ins MPS_Tensor:$gradient, MPS_Tensor:$source,
                       I64Attr:$axis, I64Attr:$k);
  let results = (outs MPS_Tensor:$result);
}

//===----------------------------------------------------------------------===//
// RNN ops (MPSGraphRNNOps.h)
//===----------------------------------------------------------------------===//

def MPS_SingleGateRNNOp : MPS_Op<"single_gate_rnn",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "single-gate RNN";
  let description = [{
    Returns `h` and, if `training`, `z`. A missing `input_weight` is the
    identity, missing `bias` and `init_state` are zeros and a missing `mask`
    is ones.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$recurrent_weight,
                            Optional<MPS_Tensor>:$input_weight,
                            Optional<MPS_Tensor>:$bias,
                            Optional<MPS_Tensor>:$init_state,
                            Optional<MPS_Tensor>:$mask),
                       SingleGateRNNDescriptor);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = ?;
}

def MPS_SingleGateRNNGradientsOp : MPS_Op<"single_gate_rnn_gradients",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "gradients of `mps.single_gate_rnn`";
  let description = [{
    Returns the gradients for `source`, `recurrent_weight`, `input_weight`,
    `bias` and `init_state`, in that order, skipping absent inputs.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$recurrent_weight,
                            MPS_Tensor:$source_gradient, MPS_Tensor:$z_state,
                            Optional<MPS_Tensor>:$state_gradient,
                            Optional<MPS_Tensor>:$input_weight,
                            Optional<MPS_Tensor>:$bias,
                            Optional<MPS_Tensor>:$init_state,
                            Optional<MPS_Tensor>:$mask),
                       SingleGateRNNDescriptor);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = ?;
}

def MPS_LSTMOp : MPS_Op<"lstm", [Pure, AttrSizedOperandSegments]> {
  let summary = "LSTM";
  let description = [{
    Returns `h`, then `c` if `produce_cell`, then `z` if `training`. A missing
    `input_weight` is the identity, missing `bias`, `init_state`, `init_cell`
    and `peephole` are zeros and a missing `mask` is ones.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$recurrent_weight,
                            Optional<MPS_Tensor>:$input_weight,
                            Optional<MPS_Tensor>:$bias,
                            Optional<MPS_Tensor>:$init_state,
                            Optional<MPS_Tensor>:$init_cell,
                            Optional<MPS_Tensor>:$mask,
                            Optional<MPS_Tensor>:$peephole),
                       LSTMDescriptor);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = ?;
}

def MPS_LSTMGradientsOp : MPS_Op<"lstm_gradients",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "gradients of `mps.lstm`";
  let description = [{
    Returns the gradients for `source`, `recurrent_weight`, `input_weight`,
    `bias`, `peephole`, `init_state` and `init_cell`, in that order, skipping
    absent inputs.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$recurrent_weight,
                            MPS_Tensor:$source_gradient, MPS_Tensor:$z_state,
                            MPS_Tensor:$cell_output_fwd,
                            Optional<MPS_Tensor>:$state_gradient,
                            Optional<MPS_Tensor>:$cell_gradient,
                            Optional<MPS_Tensor>:$input_weight,
                            Optional<MPS_Tensor>:$bias,
                            Optional<MPS_Tensor>:$init_state,
                            Optional<MPS_Tensor>:$init_cell,
                            Optional<MPS_Tensor>:$mask,
                            Optional<MPS_Tensor>:$peephole),
                       LSTMDescriptor);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = ?;
}

def MPS_GRUOp : MPS_Op<"gru", [Pure, AttrSizedOperandSegments]> {
  let summary = "GRU";
  let description = [{
    Returns `h` and, if `training`, `z`. A missing `input_weight` is the
    identity, missing `bias`, `init_state` and `secondary_bias` are zeros and
    a missing `mask` is ones. `secondary_bias` is only used with
    `reset_after`.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$recurrent_weight,
                            Optional<MPS_Tensor>:$input_weight,
                            Optional<MPS_Tensor>:$bias,
                            Optional<MPS_Tensor>:$init_state,
                            Optional<MPS_Tensor>:$mask,
                            Optional<MPS_Tensor>:$secondary_bias),
                       GRUDescriptor);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = ?;
}

def MPS_GRUGradientsOp : MPS_Op<"gru_gradients",
    [Pure, AttrSizedOperandSegments]> {
  let summary = "gradients of `mps.gru`";
  let description = [{
    Returns the gradients for `source`, `recurrent_weight`, `input_weight`,
    `bias`, `init_state` and `secondary_bias`, in that order, skipping absent
    inputs.
  }];
  let arguments = !con((ins MPS_Tensor:$source, MPS_Tensor:$recurrent_weight,
                            MPS_Tensor:$source_gradient, MPS_Tensor:$z_state,
                            MPS_Tensor:$output_fwd,
                            Optional<MPS_Tensor>:$state_gradient,
                            Optional<MPS_Tensor>:$input_weight,
                            Optional<MPS_Tensor>:$bias,
                            Optional<MPS_Tensor>:$init_state,
                            Optional<MPS_Tensor>:$mask,
                            Optional<MPS_Tensor>:$secondary_bias),
                       GRUDescriptor);
  let results = (outs Variadic<MPS_Tensor>:$results);
  let assemblyFormat = ?;
}

#endif // ENZYMEXLA_MPS_OPS_TD
