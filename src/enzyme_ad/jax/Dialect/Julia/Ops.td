#ifndef ENZYME_AD_JAX_DIALECT_JULIA_OPS_TD
#define ENZYME_AD_JAX_DIALECT_JULIA_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "src/enzyme_ad/jax/Dialect/Julia/Dialect.td"
include "src/enzyme_ad/jax/Dialect/Julia/Types.td"
include "src/enzyme_ad/jax/Dialect/Julia/Attrs.td"

//===----------------------------------------------------------------------===//
// Function & Control Flow Operations
//===----------------------------------------------------------------------===//

def Julia_FuncOp : Julia_Op<"func", [
    AffineScope, AutomaticAllocationScope,
    SymbolName, SymbolVisibility, FunctionOpInterface, IsolatedFromAbove,
    OpAsmOpInterface
  ]> {
  let summary = "Julia function definition";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Julia_ReturnOp : Julia_Op<"return", [
    Pure, HasParent<"FuncOp">, ReturnLike, Terminator
  ]> {
  let summary = "Return values from a `julia.func`";
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* empty */ }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def Julia_CallOp : Julia_Op<"call", [CallOpInterface]> {
  let summary = "Call a `julia.func`";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $operands `)` prop-dict attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    operand_range getArgOperands() { return getOperands(); }
    MutableOperandRange getArgOperandsMutable() { return getOperandsMutable(); }
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
    void setCalleeFromCallable(CallInterfaceCallable callee) {
      (*this)->setAttr("callee", cast<SymbolRefAttr>(callee));
    }
  }];
}

def Julia_YieldOp : Julia_Op<"yield", [Pure, ReturnLike, Terminator]> {
  let summary = "Region terminator yielding values";
  let arguments = (ins Variadic<AnyType>:$operands);
  let builders = [OpBuilder<(ins), [{ /* empty */ }]>];
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Julia_CondOp : Julia_Op<"cond", [
    Pure, Terminator, HasParent<"WhileOp">
  ]> {
  let summary = "Condition region terminator for `julia.while`";
  let arguments = (ins I1:$predicate);
  let assemblyFormat = "$predicate attr-dict";
}

def Julia_IfOp : Julia_Op<"if", [RecursiveMemoryEffects]> {
  let summary = "Julia conditional expression / statement";
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$thenRegion,
    MaxSizedRegion<1>:$elseRegion
  );
  let hasCustomAssemblyFormat = 1;
}

def Julia_WhileOp : Julia_Op<"while", [RecursiveMemoryEffects]> {
  let summary = "Julia structured while loop";
  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region
    SizedRegion<1>:$cond,
    SizedRegion<1>:$body
  );
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Scalar Operations
//===----------------------------------------------------------------------===//

def Julia_ConstantOp : Julia_Op<"constant", [Pure, ConstantLike]> {
  let summary = "Scalar literal constant";
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
  let hasFolder = 1;
}

def Julia_UnaryOp : Julia_Op<"unary", [Pure]> {
  let summary = "Scalar unary arithmetic / math operation";
  let arguments = (ins
    Julia_UnaryOpKindAttr:$op,
    AnyType:$input
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$op $input attr-dict `:` functional-type($input, $result)";
}

def Julia_BinaryOp : Julia_Op<"binary", [Pure]> {
  let summary = "Scalar binary arithmetic / bitwise / math operation";
  let arguments = (ins
    Julia_BinaryOpKindAttr:$op,
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$op $lhs `,` $rhs attr-dict `:` functional-type(operands, $result)";
}

def Julia_CmpOp : Julia_Op<"cmp", [Pure]> {
  let summary = "Scalar comparison returning `i1` (`Bool`)";
  let arguments = (ins
    Julia_CmpPredAttr:$predicate,
    AnyType:$lhs,
    AnyType:$rhs
  );
  let results = (outs I1:$result);
  let assemblyFormat = "$predicate $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
}

def Julia_SelectOp : Julia_Op<"select", [Pure]> {
  let summary = "Scalar ternary selection `ifelse(pred, on_true, on_false)`";
  let arguments = (ins
    I1:$pred,
    AnyType:$on_true,
    AnyType:$on_false
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$pred `,` $on_true `,` $on_false attr-dict `:` type($on_true) `,` type($on_false) `->` type($result)";
}

def Julia_ClampOp : Julia_Op<"clamp", [Pure]> {
  let summary = "Scalar clamp `clamp(operand, min, max)`";
  let arguments = (ins
    AnyType:$min,
    AnyType:$operand,
    AnyType:$max
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$min `,` $operand `,` $max attr-dict `:` type($min) `,` type($operand) `,` type($max) `->` type($result)";
}

def Julia_CastOp : Julia_Op<"cast", [Pure]> {
  let summary = "Scalar numeric / index / bool type conversion";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def Julia_BitcastOp : Julia_Op<"bitcast", [Pure]> {
  let summary = "Bitwise reinterpret cast `reinterpret(T, x)`";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

def Julia_MathLibmOp : Julia_Op<"libm", [Pure]> {
  let summary = "Scalar special function call via `Base.Math.libm`";
  let arguments = (ins
    Julia_LibmFnAttr:$fn,
    AnyType:$input
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$fn $input attr-dict `:` functional-type($input, $result)";
}

//===----------------------------------------------------------------------===//
// Indexing, Memory & View Operations
//===----------------------------------------------------------------------===//

def Julia_SizeOp : Julia_Op<"size", [Pure]> {
  let summary = "Query 0-based dimension size `size(arr, dim + 1)`";
  let arguments = (ins
    Julia_BufferLike:$source,
    I64Attr:$dimension
  );
  let results = (outs Index:$result);
  let assemblyFormat = "$source `[` $dimension `]` attr-dict `:` type($source)";
}

def Julia_LoadOp : Julia_Op<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "1-based element load `arr[i1, ..., iN]`";
  let arguments = (ins
    Julia_BufferLike:$source,
    Variadic<Index>:$indices
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = "$source `[` $indices `]` attr-dict `:` type($source) `->` type($result)";
}

def Julia_StoreOp : Julia_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "1-based element store `arr[i1, ..., iN] = val`";
  let arguments = (ins
    AnyType:$value,
    Julia_BufferLike:$dest,
    Variadic<Index>:$indices
  );
  let assemblyFormat = "$value `,` $dest `[` $indices `]` attr-dict `:` type($value) `,` type($dest)";
}

def Julia_AllocOp : Julia_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate uninitialized `Array{T, N}(undef, dims...)`";
  let arguments = (ins Variadic<Index>:$dynamic_dims);
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "`(` $dynamic_dims `)` attr-dict `:` type($result)";
}

def Julia_CopyOp : Julia_Op<"copy", [MemoryEffects<[MemRead, MemAlloc]>]> {
  let summary = "Allocate a fresh owned `Array{T, N}` copy of an array or view";
  let arguments = (ins Julia_BufferLike:$source);
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def Julia_MaterializeArrayOp : Julia_Op<"materialize_array", [
    MemoryEffects<[MemRead, MemAlloc]>
  ]> {
  let summary = "Materialize a borrowed `!julia.view` into an owned `!julia.array`";
  let arguments = (ins Julia_BufferLike:$source);
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($result)";
}

def Julia_CopyToOp : Julia_Op<"copy_to!", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "In-place `copyto!(dst, src)`";
  let arguments = (ins
    Julia_BufferLike:$dest,
    Julia_BufferLike:$source
  );
  let assemblyFormat = "$dest `,` $source attr-dict `:` type($dest) `,` type($source)";
}

def Julia_FillInPlaceOp : Julia_Op<"fill!", [MemoryEffects<[MemWrite]>]> {
  let summary = "In-place `fill!(dst, scalar)`";
  let arguments = (ins
    Julia_BufferLike:$dest,
    AnyType:$value
  );
  let assemblyFormat = "$dest `,` $value attr-dict `:` type($dest) `,` type($value)";
}

def Julia_ViewSliceOp : Julia_Op<"view_slice", [Pure]> {
  let summary = "Zero-copy static `@view(src[s1:st1:l1, ...])` (0-based half-open attrs)";
  let arguments = (ins
    Julia_BufferLike:$source,
    DenseI64ArrayAttr:$start_indices,
    DenseI64ArrayAttr:$limit_indices,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs Julia_ViewType:$result);
  let assemblyFormat = [{
    $source $start_indices `to` $limit_indices `step` $strides attr-dict
    `:` type($source) `to` type($result)
  }];
}

def Julia_ViewDynamicSliceOp : Julia_Op<"view_dynamic_slice", [Pure]> {
  let summary = "Zero-copy clamped dynamic `@view(src[...])`";
  let arguments = (ins
    Julia_BufferLike:$source,
    Variadic<AnyType>:$start_indices,
    DenseI64ArrayAttr:$slice_sizes
  );
  let results = (outs Julia_ViewType:$result);
  let assemblyFormat = [{
    $source `[` $start_indices `]` `sizes` $slice_sizes attr-dict
    `:` functional-type(operands, $result)
  }];
}

def Julia_ViewReshapeOp : Julia_Op<"view_reshape", [Pure]> {
  let summary = "Zero-copy `reshape(src, dims...)`";
  let arguments = (ins
    Julia_BufferLike:$source,
    Variadic<Index>:$dynamic_dims
  );
  let results = (outs Julia_ViewType:$result);
  let assemblyFormat = [{
    $source `(` $dynamic_dims `)` attr-dict `:` type($source) `to` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Value-Domain Array Operations (Pre-Bufferization)
//===----------------------------------------------------------------------===//

def Julia_FillOp : Julia_Op<"fill", [Pure]> {
  let summary = "Create an array filled with a scalar value";
  let arguments = (ins
    AnyType:$value,
    Variadic<Index>:$dynamic_dims
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$value `(` $dynamic_dims `)` attr-dict `:` type($value) `->` type($result)";
}

def Julia_ArrayConstantOp : Julia_Op<"array_constant", [Pure]> {
  let summary = "Dense column-major array literal constant";
  let arguments = (ins ElementsAttr:$values);
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$values attr-dict `:` type($result)";
}

def Julia_KernelOp : Julia_Op<"kernel", [Pure]> {
  let summary = "Fused value-domain pointwise / index-mapped loop kernel";
  let description = [{
    Evaluates its single-block region over 1-based Julia column-major indices
    `(%iv0: index, ..., %iv_{R-1}: index)` for `iv_d in 1:shape[d]`, yielding
    one scalar per output array. All result arrays share identical shape.
  }];
  let arguments = (ins Variadic<Index>:$dynamic_dims);
  let results = (outs Variadic<Julia_ArrayType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`(` $dynamic_dims `)` `->` `(` type($results) `)` $body attr-dict";
}

def Julia_BroadcastInDimOp : Julia_Op<"broadcast_in_dim", [Pure]> {
  let summary = "Value-domain broadcast along specified Julia dimensions";
  let arguments = (ins
    AnyType:$operand,
    DenseI64ArrayAttr:$broadcast_dimensions,
    Variadic<Index>:$dynamic_dims
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $operand `dims` `=` $broadcast_dimensions `(` $dynamic_dims `)` attr-dict
    `:` type($operand) `to` type($result)
  }];
}

def Julia_TransposeOp : Julia_Op<"transpose", [Pure]> {
  let summary = "Value-domain dimension permutation `out_dim[d] = in_dim[perm[d]]`";
  let arguments = (ins
    Julia_BufferLike:$operand,
    DenseI64ArrayAttr:$permutation
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$operand `perm` `=` $permutation attr-dict `:` type($operand) `to` type($result)";
}

def Julia_ReverseOp : Julia_Op<"reverse", [Pure]> {
  let summary = "Value-domain reversal along specified Julia dimensions";
  let arguments = (ins
    Julia_BufferLike:$operand,
    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$operand `dims` `=` $dimensions attr-dict `:` type($operand) `to` type($result)";
}

def Julia_ReshapeOp : Julia_Op<"reshape", [Pure]> {
  let summary = "Value-domain column-major reshape (preserving linear element order)";
  let arguments = (ins
    Julia_BufferLike:$operand,
    Variadic<Index>:$dynamic_dims
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$operand `(` $dynamic_dims `)` attr-dict `:` type($operand) `to` type($result)";
}

def Julia_SliceOp : Julia_Op<"slice", [Pure]> {
  let summary = "Value-domain static slice (0-based half-open attrs)";
  let arguments = (ins
    Julia_BufferLike:$operand,
    DenseI64ArrayAttr:$start_indices,
    DenseI64ArrayAttr:$limit_indices,
    DenseI64ArrayAttr:$strides
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $operand $start_indices `to` $limit_indices `step` $strides attr-dict
    `:` type($operand) `to` type($result)
  }];
}

def Julia_DynamicSliceOp : Julia_Op<"dynamic_slice", [Pure]> {
  let summary = "Value-domain clamped dynamic slice";
  let arguments = (ins
    Julia_BufferLike:$operand,
    Variadic<AnyType>:$start_indices,
    DenseI64ArrayAttr:$slice_sizes
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $operand `[` $start_indices `]` `sizes` $slice_sizes attr-dict
    `:` functional-type(operands, $result)
  }];
}

def Julia_DynamicUpdateSliceOp : Julia_Op<"dynamic_update_slice", [Pure]> {
  let summary = "Value-domain clamped dynamic update slice";
  let arguments = (ins
    Julia_BufferLike:$operand,
    AnyType:$update,
    Variadic<AnyType>:$start_indices
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $operand `,` $update `[` $start_indices `]` attr-dict
    `:` functional-type(operands, $result)
  }];
}

def Julia_ConcatOp : Julia_Op<"concat", [Pure]> {
  let summary = "Value-domain concatenation along a Julia dimension";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$inputs,
    I64Attr:$dimension
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "`(` $inputs `)` `dim` `=` $dimension attr-dict `:` functional-type($inputs, $result)";
}

def Julia_PadOp : Julia_Op<"pad", [Pure]> {
  let summary = "Value-domain edge + interior padding";
  let arguments = (ins
    Julia_BufferLike:$operand,
    AnyType:$padding_value,
    DenseI64ArrayAttr:$edge_padding_low,
    DenseI64ArrayAttr:$edge_padding_high,
    DenseI64ArrayAttr:$interior_padding
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $operand `,` $padding_value `low` `=` $edge_padding_low
    `high` `=` $edge_padding_high `interior` `=` $interior_padding attr-dict
    `:` type($operand) `,` type($padding_value) `to` type($result)
  }];
}

def Julia_MatmulOp : Julia_Op<"matmul", [Pure, AttrSizedOperandSegments]> {
  let summary = "Value-domain BLAS/LinearAlgebra matrix/vector product `alpha * op(A) * op(B) + beta * C`";
  let arguments = (ins
    Julia_BufferLike:$lhs,
    Julia_BufferLike:$rhs,
    Optional<Julia_BufferLike>:$acc,
    Optional<AnyType>:$alpha,
    Optional<AnyType>:$beta,
    Julia_TransposeModeAttr:$lhs_transpose,
    Julia_TransposeModeAttr:$rhs_transpose,
    DenseI64ArrayAttr:$batch_dims_lhs,
    DenseI64ArrayAttr:$batch_dims_rhs,
    DenseI64ArrayAttr:$batch_dims_out
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`acc` `(` $acc^ `:` type($acc) `)`)?
    (`alpha` `(` $alpha^ `:` type($alpha) `)`)?
    (`beta` `(` $beta^ `:` type($beta) `)`)?
    `lhs_trans` `=` $lhs_transpose `rhs_trans` `=` $rhs_transpose
    `batch` `=` `[` $batch_dims_lhs `,` $batch_dims_rhs `,` $batch_dims_out `]`
    attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Julia_DotGeneralOp : Julia_Op<"dot_general", [Pure]> {
  let summary = "Value-domain general tensor contraction";
  let arguments = (ins
    Julia_BufferLike:$lhs,
    Julia_BufferLike:$rhs,
    DenseI64ArrayAttr:$lhs_batching_dims,
    DenseI64ArrayAttr:$rhs_batching_dims,
    DenseI64ArrayAttr:$lhs_contracting_dims,
    DenseI64ArrayAttr:$rhs_contracting_dims,
    DenseI64ArrayAttr:$result_dim_order
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs
    `batch` `=` `[` $lhs_batching_dims `,` $rhs_batching_dims `]`
    `contract` `=` `[` $lhs_contracting_dims `,` $rhs_contracting_dims `]`
    `order` `=` $result_dim_order
    attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Julia_ReduceOp : Julia_Op<"reduce", [Pure, SameVariadicOperandSize]> {
  let summary = "Value-domain (single or variadic) reduction over Julia dimensions";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$init_values,
    DenseI64ArrayAttr:$dimensions
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combiner);
  let assemblyFormat = [{
    `(` $inputs `)` `inits` `(` $init_values `)` `dims` `=` $dimensions
    $combiner attr-dict `:` functional-type(operands, results)
  }];
}

def Julia_ReduceWindowOp : Julia_Op<"reduce_window", [Pure, SameVariadicOperandSize]> {
  let summary = "Value-domain windowed reduction (pooling / stencil)";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$inputs,
    Variadic<AnyType>:$init_values,
    DenseI64ArrayAttr:$window_dimensions,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$base_dilations,
    DenseI64ArrayAttr:$window_dilations,
    DenseI64ArrayAttr:$padding
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$combiner);
  let assemblyFormat = [{
    `(` $inputs `)` `inits` `(` $init_values `)`
    `window` `=` $window_dimensions `strides` `=` $window_strides
    `base_dil` `=` $base_dilations `win_dil` `=` $window_dilations
    `pad` `=` $padding $combiner attr-dict `:` functional-type(operands, results)
  }];
}

def Julia_GatherOp : Julia_Op<"gather", [Pure]> {
  let summary = "Value-domain general tensor gather";
  let arguments = (ins
    Julia_BufferLike:$operand,
    AnyType:$start_indices,
    DenseI64ArrayAttr:$offset_dims,
    DenseI64ArrayAttr:$collapsed_slice_dims,
    DenseI64ArrayAttr:$operand_batching_dims,
    DenseI64ArrayAttr:$start_indices_batching_dims,
    DenseI64ArrayAttr:$start_index_map,
    I64Attr:$index_vector_dim,
    DenseI64ArrayAttr:$slice_sizes
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $operand `,` $start_indices
    `offset_dims` `=` $offset_dims
    `collapsed` `=` $collapsed_slice_dims
    `op_batch` `=` $operand_batching_dims
    `idx_batch` `=` $start_indices_batching_dims
    `start_map` `=` $start_index_map
    `idx_vec_dim` `=` $index_vector_dim
    `sizes` `=` $slice_sizes attr-dict
    `:` type($operand) `,` type($start_indices) `->` type($result)
  }];
}

def Julia_ScatterOp : Julia_Op<"scatter", [Pure, AttrSizedOperandSegments]> {
  let summary = "Value-domain variadic tensor scatter";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$inputs,
    AnyType:$scatter_indices,
    Variadic<AnyType>:$updates,
    DenseI64ArrayAttr:$update_window_dims,
    DenseI64ArrayAttr:$inserted_window_dims,
    DenseI64ArrayAttr:$input_batching_dims,
    DenseI64ArrayAttr:$scatter_indices_batching_dims,
    DenseI64ArrayAttr:$scatter_dims_to_operand_dims,
    I64Attr:$index_vector_dim
  );
  let results = (outs Variadic<Julia_ArrayType>:$results);
  let regions = (region SizedRegion<1>:$update_computation);
  let assemblyFormat = [{
    `inputs` `(` $inputs `)` `indices` `(` $scatter_indices `)` `updates` `(` $updates `)`
    `update_win` `=` $update_window_dims
    `inserted_win` `=` $inserted_window_dims
    `in_batch` `=` $input_batching_dims
    `idx_batch` `=` $scatter_indices_batching_dims
    `scatter_map` `=` $scatter_dims_to_operand_dims
    `idx_vec_dim` `=` $index_vector_dim
    $update_computation attr-dict `:` functional-type(operands, results)
  }];
}

def Julia_SortOp : Julia_Op<"sort", [Pure]> {
  let summary = "Value-domain variadic co-sort along a Julia dimension";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$inputs,
    I64Attr:$dimension,
    BoolAttr:$is_stable
  );
  let results = (outs Variadic<Julia_ArrayType>:$results);
  let regions = (region SizedRegion<1>:$comparator);
  let assemblyFormat = [{
    `(` $inputs `)` `dim` `=` $dimension `stable` `=` $is_stable
    $comparator attr-dict `:` functional-type($inputs, results)
  }];
}

def Julia_ConvOp : Julia_Op<"conv", [Pure]> {
  let summary = "Value-domain general N-D convolution";
  let arguments = (ins
    Julia_BufferLike:$lhs,
    Julia_BufferLike:$rhs,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$lhs_dilation,
    DenseI64ArrayAttr:$rhs_dilation,
    DenseBoolArrayAttr:$window_reversal,
    I64Attr:$input_batch_dimension,
    I64Attr:$input_feature_dimension,
    DenseI64ArrayAttr:$input_spatial_dimensions,
    I64Attr:$kernel_input_feature_dimension,
    I64Attr:$kernel_output_feature_dimension,
    DenseI64ArrayAttr:$kernel_spatial_dimensions,
    I64Attr:$output_batch_dimension,
    I64Attr:$output_feature_dimension,
    DenseI64ArrayAttr:$output_spatial_dimensions,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs
    `strides` `=` $window_strides `pad` `=` $padding
    `lhs_dil` `=` $lhs_dilation `rhs_dil` `=` $rhs_dilation
    `rev` `=` $window_reversal
    `in_dims` `=` `[` $input_batch_dimension `,` $input_feature_dimension `,` $input_spatial_dimensions `]`
    `ker_dims` `=` `[` $kernel_input_feature_dimension `,` $kernel_output_feature_dimension `,` $kernel_spatial_dimensions `]`
    `out_dims` `=` `[` $output_batch_dimension `,` $output_feature_dimension `,` $output_spatial_dimensions `]`
    `groups` `=` `[` $feature_group_count `,` $batch_group_count `]`
    attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def Julia_CholeskyOp : Julia_Op<"cholesky", [Pure]> {
  let summary = "Value-domain Cholesky factorization via `LinearAlgebra.cholesky!`";
  let arguments = (ins
    Julia_BufferLike:$operand,
    BoolAttr:$lower
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = "$operand `lower` `=` $lower attr-dict `:` type($operand) `->` type($result)";
}

def Julia_TriangularSolveOp : Julia_Op<"triangular_solve", [Pure]> {
  let summary = "Value-domain triangular solve via `LinearAlgebra.ldiv!` / `rdiv!`";
  let arguments = (ins
    Julia_BufferLike:$a,
    Julia_BufferLike:$b,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    Julia_TransposeModeAttr:$transpose_a
  );
  let results = (outs Julia_ArrayType:$result);
  let assemblyFormat = [{
    $a `,` $b `left` `=` $left_side `lower` `=` $lower
    `unit` `=` $unit_diagonal `trans` `=` $transpose_a attr-dict
    `:` type($a) `,` type($b) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Buffer-Domain Destination-Passing / In-Place Operations (Post-Bufferization)
//===----------------------------------------------------------------------===//

def Julia_KernelInPlaceOp : Julia_Op<"kernel!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place fused column-major / `@simd` loop kernel writing `outs`";
  let arguments = (ins Variadic<Julia_BufferLike>:$outs);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "`outs` `(` $outs `:` type($outs) `)` $body attr-dict";
}

def Julia_MatmulInPlaceOp : Julia_Op<"matmul!", [
    MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments
  ]> {
  let summary = "In-place `LinearAlgebra.mul!(dst, op(A), op(B), alpha, beta)`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$lhs,
    Julia_BufferLike:$rhs,
    Optional<AnyType>:$alpha,
    Optional<AnyType>:$beta,
    Julia_TransposeModeAttr:$lhs_transpose,
    Julia_TransposeModeAttr:$rhs_transpose,
    DenseI64ArrayAttr:$batch_dims_lhs,
    DenseI64ArrayAttr:$batch_dims_rhs,
    DenseI64ArrayAttr:$batch_dims_out
  );
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs
    (`alpha` `(` $alpha^ `:` type($alpha) `)`)?
    (`beta` `(` $beta^ `:` type($beta) `)`)?
    `lhs_trans` `=` $lhs_transpose `rhs_trans` `=` $rhs_transpose
    `batch` `=` `[` $batch_dims_lhs `,` $batch_dims_rhs `,` $batch_dims_out `]`
    attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
}

def Julia_DotGeneralInPlaceOp : Julia_Op<"dot_general!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place general tensor contraction writing `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$lhs,
    Julia_BufferLike:$rhs,
    DenseI64ArrayAttr:$lhs_batching_dims,
    DenseI64ArrayAttr:$rhs_batching_dims,
    DenseI64ArrayAttr:$lhs_contracting_dims,
    DenseI64ArrayAttr:$rhs_contracting_dims,
    DenseI64ArrayAttr:$result_dim_order
  );
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs
    `batch` `=` `[` $lhs_batching_dims `,` $rhs_batching_dims `]`
    `contract` `=` `[` $lhs_contracting_dims `,` $rhs_contracting_dims `]`
    `order` `=` $result_dim_order
    attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
}

def Julia_ReduceInPlaceOp : Julia_Op<"reduce!", [
    MemoryEffects<[MemRead, MemWrite]>, SameVariadicOperandSize
  ]> {
  let summary = "In-place partial reduction writing `dsts`";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$dsts,
    Variadic<AnyType>:$inputs,
    Variadic<AnyType>:$init_values,
    DenseI64ArrayAttr:$dimensions
  );
  let regions = (region SizedRegion<1>:$combiner);
  let assemblyFormat = [{
    `outs` `(` $dsts `)` `ins` `(` $inputs `)` `inits` `(` $init_values `)`
    `dims` `=` $dimensions $combiner attr-dict
    `:` `(` type($dsts) `)` `,` `(` type($inputs) `)` `,` `(` type($init_values) `)`
  }];
}

def Julia_ReduceWindowInPlaceOp : Julia_Op<"reduce_window!", [
    MemoryEffects<[MemRead, MemWrite]>, SameVariadicOperandSize
  ]> {
  let summary = "In-place windowed reduction writing `dsts`";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$dsts,
    Variadic<Julia_BufferLike>:$inputs,
    Variadic<AnyType>:$init_values,
    DenseI64ArrayAttr:$window_dimensions,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$base_dilations,
    DenseI64ArrayAttr:$window_dilations,
    DenseI64ArrayAttr:$padding
  );
  let regions = (region SizedRegion<1>:$combiner);
  let assemblyFormat = [{
    `outs` `(` $dsts `)` `ins` `(` $inputs `)` `inits` `(` $init_values `)`
    `window` `=` $window_dimensions `strides` `=` $window_strides
    `base_dil` `=` $base_dilations `win_dil` `=` $window_dilations
    `pad` `=` $padding $combiner attr-dict
    `:` `(` type($dsts) `)` `,` `(` type($inputs) `)` `,` `(` type($init_values) `)`
  }];
}

def Julia_PadInPlaceOp : Julia_Op<"pad!", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "In-place edge + interior padding writing `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$operand,
    AnyType:$padding_value,
    DenseI64ArrayAttr:$edge_padding_low,
    DenseI64ArrayAttr:$edge_padding_high,
    DenseI64ArrayAttr:$interior_padding
  );
  let assemblyFormat = [{
    $dst `,` $operand `,` $padding_value
    `low` `=` $edge_padding_low `high` `=` $edge_padding_high
    `interior` `=` $interior_padding attr-dict
    `:` type($dst) `,` type($operand) `,` type($padding_value)
  }];
}

def Julia_DynamicSliceInPlaceOp : Julia_Op<"dynamic_slice!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place clamped dynamic slice writing `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$operand,
    Variadic<AnyType>:$start_indices,
    DenseI64ArrayAttr:$slice_sizes
  );
  let assemblyFormat = [{
    $dst `,` $operand `[` $start_indices `]` `sizes` $slice_sizes attr-dict
    `:` type($dst) `,` type($operand) (`,` type($start_indices)^)?
  }];
}

def Julia_DynamicUpdateSliceInPlaceOp : Julia_Op<"dynamic_update_slice!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place clamped dynamic update slice mutating `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    AnyType:$update,
    Variadic<AnyType>:$start_indices
  );
  let assemblyFormat = [{
    $dst `,` $update `[` $start_indices `]` attr-dict
    `:` type($dst) `,` type($update) (`,` type($start_indices)^)?
  }];
}

def Julia_GatherInPlaceOp : Julia_Op<"gather!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place tensor gather writing `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$operand,
    AnyType:$start_indices,
    DenseI64ArrayAttr:$offset_dims,
    DenseI64ArrayAttr:$collapsed_slice_dims,
    DenseI64ArrayAttr:$operand_batching_dims,
    DenseI64ArrayAttr:$start_indices_batching_dims,
    DenseI64ArrayAttr:$start_index_map,
    I64Attr:$index_vector_dim,
    DenseI64ArrayAttr:$slice_sizes
  );
  let assemblyFormat = [{
    $dst `,` $operand `,` $start_indices
    `offset_dims` `=` $offset_dims
    `collapsed` `=` $collapsed_slice_dims
    `op_batch` `=` $operand_batching_dims
    `idx_batch` `=` $start_indices_batching_dims
    `start_map` `=` $start_index_map
    `idx_vec_dim` `=` $index_vector_dim
    `sizes` `=` $slice_sizes attr-dict
    `:` type($dst) `,` type($operand) `,` type($start_indices)
  }];
}

def Julia_ScatterInPlaceOp : Julia_Op<"scatter!", [
    MemoryEffects<[MemRead, MemWrite]>, AttrSizedOperandSegments
  ]> {
  let summary = "In-place variadic tensor scatter mutating `dsts`";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$dsts,
    AnyType:$scatter_indices,
    Variadic<AnyType>:$updates,
    DenseI64ArrayAttr:$update_window_dims,
    DenseI64ArrayAttr:$inserted_window_dims,
    DenseI64ArrayAttr:$input_batching_dims,
    DenseI64ArrayAttr:$scatter_indices_batching_dims,
    DenseI64ArrayAttr:$scatter_dims_to_operand_dims,
    I64Attr:$index_vector_dim
  );
  let regions = (region SizedRegion<1>:$update_computation);
  let assemblyFormat = [{
    `outs` `(` $dsts `)` `indices` `(` $scatter_indices `)` `updates` `(` $updates `)`
    `update_win` `=` $update_window_dims
    `inserted_win` `=` $inserted_window_dims
    `in_batch` `=` $input_batching_dims
    `idx_batch` `=` $scatter_indices_batching_dims
    `scatter_map` `=` $scatter_dims_to_operand_dims
    `idx_vec_dim` `=` $index_vector_dim
    $update_computation attr-dict
    `:` `(` type($dsts) `)` `,` type($scatter_indices) `,` `(` type($updates) `)`
  }];
}

def Julia_SortInPlaceOp : Julia_Op<"sort!", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "In-place variadic co-sort mutating `dsts` along `dimension`";
  let arguments = (ins
    Variadic<Julia_BufferLike>:$dsts,
    I64Attr:$dimension,
    BoolAttr:$is_stable
  );
  let regions = (region SizedRegion<1>:$comparator);
  let assemblyFormat = [{
    `outs` `(` $dsts `:` type($dsts) `)` `dim` `=` $dimension `stable` `=` $is_stable
    $comparator attr-dict
  }];
}

def Julia_ConvInPlaceOp : Julia_Op<"conv!", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "In-place general N-D convolution writing `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$lhs,
    Julia_BufferLike:$rhs,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$lhs_dilation,
    DenseI64ArrayAttr:$rhs_dilation,
    DenseBoolArrayAttr:$window_reversal,
    I64Attr:$input_batch_dimension,
    I64Attr:$input_feature_dimension,
    DenseI64ArrayAttr:$input_spatial_dimensions,
    I64Attr:$kernel_input_feature_dimension,
    I64Attr:$kernel_output_feature_dimension,
    DenseI64ArrayAttr:$kernel_spatial_dimensions,
    I64Attr:$output_batch_dimension,
    I64Attr:$output_feature_dimension,
    DenseI64ArrayAttr:$output_spatial_dimensions,
    I64Attr:$feature_group_count,
    I64Attr:$batch_group_count
  );
  let assemblyFormat = [{
    $dst `,` $lhs `,` $rhs
    `strides` `=` $window_strides `pad` `=` $padding
    `lhs_dil` `=` $lhs_dilation `rhs_dil` `=` $rhs_dilation
    `rev` `=` $window_reversal
    `in_dims` `=` `[` $input_batch_dimension `,` $input_feature_dimension `,` $input_spatial_dimensions `]`
    `ker_dims` `=` `[` $kernel_input_feature_dimension `,` $kernel_output_feature_dimension `,` $kernel_spatial_dimensions `]`
    `out_dims` `=` `[` $output_batch_dimension `,` $output_feature_dimension `,` $output_spatial_dimensions `]`
    `groups` `=` `[` $feature_group_count `,` $batch_group_count `]`
    attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)
  }];
}

def Julia_CholeskyInPlaceOp : Julia_Op<"cholesky!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place Cholesky factorization mutating `dst`";
  let arguments = (ins
    Julia_BufferLike:$dst,
    BoolAttr:$lower
  );
  let assemblyFormat = "$dst `lower` `=` $lower attr-dict `:` type($dst)";
}

def Julia_TriangularSolveInPlaceOp : Julia_Op<"triangular_solve!", [
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "In-place triangular solve mutating `dst` (pre-populated with `b`)";
  let arguments = (ins
    Julia_BufferLike:$dst,
    Julia_BufferLike:$a,
    BoolAttr:$left_side,
    BoolAttr:$lower,
    BoolAttr:$unit_diagonal,
    Julia_TransposeModeAttr:$transpose_a
  );
  let assemblyFormat = [{
    $dst `,` $a `left` `=` $left_side `lower` `=` $lower
    `unit` `=` $unit_diagonal `trans` `=` $transpose_a attr-dict
    `:` type($dst) `,` type($a)
  }];
}

#endif // ENZYME_AD_JAX_DIALECT_JULIA_OPS_TD
