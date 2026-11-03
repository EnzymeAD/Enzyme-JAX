#ifndef ENZYME_AD_JAX_DIALECT_JULIA_TYPES_TD
#define ENZYME_AD_JAX_DIALECT_JULIA_TYPES_TD

include "src/enzyme_ad/jax/Dialect/Julia/Dialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

def Julia_ArrayType : Julia_Type<"Array", "array", [ShapedTypeInterface]> {
  let summary = "Owned mutable column-major Julia Array{T, N}";
  let description = [{
    Represents a heap-allocated, column-major Julia `Array{T, N}` with static
    or dynamic (`?`) dimensions. Dimension 0 is the contiguous minor-most
    column dimension in memory (`stride == 1`).
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }
    ArrayType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                        Type elementType) const {
      return ArrayType::get(shape.value_or(getShape()), elementType);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Julia_ViewType : Julia_Type<"View", "view", [ShapedTypeInterface]> {
  let summary = "Non-owning borrowed Julia view (SubArray / ReshapedArray)";
  let description = [{
    Represents a zero-copy borrowed view (`@view(arr[...])` or `reshape(arr, ...)`)
    aliasing an underlying `!julia.array` root buffer.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape,
      "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    bool hasRank() const { return true; }
    ViewType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                       Type elementType) const {
      return ViewType::get(shape.value_or(getShape()), elementType);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

def Julia_BufferLike : AnyTypeOf<[Julia_ArrayType, Julia_ViewType],
                                 "Julia array or view type">;

#endif // ENZYME_AD_JAX_DIALECT_JULIA_TYPES_TD
