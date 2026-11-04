#ifndef ENZYME_AD_JAX_DIALECT_JULIA_ATTRS_TD
#define ENZYME_AD_JAX_DIALECT_JULIA_ATTRS_TD

include "src/enzyme_ad/jax/Dialect/Julia/Dialect.td"
include "mlir/IR/EnumAttr.td"

def Julia_UnaryOpKind : I32EnumAttr<"UnaryOpKind", "Julia scalar unary operator", [
    I32EnumAttrCase<"neg", 0>,
    I32EnumAttrCase<"not_op", 1>,
    I32EnumAttrCase<"abs", 2>,
    I32EnumAttrCase<"exp", 3>,
    I32EnumAttrCase<"expm1", 4>,
    I32EnumAttrCase<"exp2", 5>,
    I32EnumAttrCase<"log", 6>,
    I32EnumAttrCase<"log1p", 7>,
    I32EnumAttrCase<"log2", 8>,
    I32EnumAttrCase<"sqrt", 9>,
    I32EnumAttrCase<"rsqrt", 10>,
    I32EnumAttrCase<"cbrt", 11>,
    I32EnumAttrCase<"sin", 12>,
    I32EnumAttrCase<"cos", 13>,
    I32EnumAttrCase<"tan", 14>,
    I32EnumAttrCase<"asin", 15>,
    I32EnumAttrCase<"acos", 16>,
    I32EnumAttrCase<"atan", 17>,
    I32EnumAttrCase<"sinh", 18>,
    I32EnumAttrCase<"cosh", 19>,
    I32EnumAttrCase<"tanh", 20>,
    I32EnumAttrCase<"asinh", 21>,
    I32EnumAttrCase<"acosh", 22>,
    I32EnumAttrCase<"atanh", 23>,
    I32EnumAttrCase<"floor", 24>,
    I32EnumAttrCase<"ceil", 25>,
    I32EnumAttrCase<"round_nearest_even", 26>,
    I32EnumAttrCase<"round_nearest_afz", 27>,
    I32EnumAttrCase<"sign", 28>,
    I32EnumAttrCase<"is_finite", 29>,
    I32EnumAttrCase<"logistic", 30>,
    I32EnumAttrCase<"real", 31>,
    I32EnumAttrCase<"imag", 32>,
    I32EnumAttrCase<"conj", 33>,
    I32EnumAttrCase<"leading_zeros", 34>,
    I32EnumAttrCase<"popcnt", 35>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::julia";
}

def Julia_UnaryOpKindAttr : EnumAttr<Julia_Dialect, Julia_UnaryOpKind, "unary_op"> {
  let assemblyFormat = "`<` $value `>`";
}

def Julia_BinaryOpKind : I32EnumAttr<"BinaryOpKind", "Julia scalar binary operator", [
    I32EnumAttrCase<"add", 0>,
    I32EnumAttrCase<"sub", 1>,
    I32EnumAttrCase<"mul", 2>,
    I32EnumAttrCase<"div", 3>,
    I32EnumAttrCase<"rem", 4>,
    I32EnumAttrCase<"pow", 5>,
    I32EnumAttrCase<"atan2", 6>,
    I32EnumAttrCase<"max", 7>,
    I32EnumAttrCase<"min", 8>,
    I32EnumAttrCase<"and_op", 9>,
    I32EnumAttrCase<"or_op", 10>,
    I32EnumAttrCase<"xor_op", 11>,
    I32EnumAttrCase<"shl", 12>,
    I32EnumAttrCase<"shr_arith", 13>,
    I32EnumAttrCase<"shr_logical", 14>,
    I32EnumAttrCase<"complex_make", 15>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::julia";
}

def Julia_BinaryOpKindAttr : EnumAttr<Julia_Dialect, Julia_BinaryOpKind, "binary_op"> {
  let assemblyFormat = "`<` $value `>`";
}

def Julia_CmpPred : I32EnumAttr<"CmpPred", "Julia comparison predicate", [
    I32EnumAttrCase<"eq", 0>,
    I32EnumAttrCase<"ne", 1>,
    I32EnumAttrCase<"lt", 2>,
    I32EnumAttrCase<"le", 3>,
    I32EnumAttrCase<"gt", 4>,
    I32EnumAttrCase<"ge", 5>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::julia";
}

def Julia_CmpPredAttr : EnumAttr<Julia_Dialect, Julia_CmpPred, "cmp_pred"> {
  let assemblyFormat = "`<` $value `>`";
}

def Julia_LibmFn : I32EnumAttr<"LibmFn", "Julia Base.Math.libm function", [
    I32EnumAttrCase<"erf", 0>,
    I32EnumAttrCase<"erfc", 1>,
    I32EnumAttrCase<"lgamma", 2>,
    I32EnumAttrCase<"tgamma", 3>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::julia";
}

def Julia_LibmFnAttr : EnumAttr<Julia_Dialect, Julia_LibmFn, "libm_fn"> {
  let assemblyFormat = "`<` $value `>`";
}

def Julia_TransposeMode : I32EnumAttr<"TransposeMode", "Matrix transpose mode", [
    I32EnumAttrCase<"none", 0>,
    I32EnumAttrCase<"transpose", 1>,
    I32EnumAttrCase<"adjoint", 2>
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::enzymexla::julia";
}

def Julia_TransposeModeAttr : EnumAttr<Julia_Dialect, Julia_TransposeMode, "transpose_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // ENZYME_AD_JAX_DIALECT_JULIA_ATTRS_TD
