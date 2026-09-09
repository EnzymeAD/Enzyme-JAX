#ifndef ENZYMEXLA_COMM_DIALECT_COMM_UTILS_TD
#define ENZYMEXLA_COMM_DIALECT_COMM_UTILS_TD

def TensorI32
    : Type<CPred<"::llvm::isa<::mlir::TensorType>($_self) && "
                 "::llvm::cast<::mlir::TensorType>($_self).getShape().size() "
                 "== 0 && "
                 "::llvm::cast<::mlir::TensorType>($_self).getElementType()."
                 "isSignlessInteger(32)">,
           "tensor<i32>", "::mlir::TensorType">,
      BuildableType<"RankedTensorType::get({}, $_builder.getIntegerType(32))">;

#endif // ENZYMEXLA_COMM_DIALECT_COMM_UTILS_TD
