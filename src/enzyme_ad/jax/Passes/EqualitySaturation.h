#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "rust/cxx.h"

namespace tensat {
enum class Type : uint8_t;
enum class Ops : uint8_t;
struct Vector;
struct Tensor;

/**
 * Functions exposed to Rust (Tensat) for getting the cost of new operations.
 */

uint64_t get_cost(Ops op, rust::Vec<tensat::Tensor> operands,
                  rust::Vec<tensat::Vector> other_vector_args,
                  rust::Vec<int64_t> int_args);

mlir::Type newTensorType(mlir::OpBuilder &builder, Tensor tensor); 
mlir::Type tensatTypeToMlirType(mlir::OpBuilder &builder, Type type);

rust::Vec<Tensor> get_shape(Ops op, rust::Vec<tensat::Tensor> operands,
                           rust::Vec<tensat::Vector> other_vector_args,
                           rust::Vec<int64_t> int_args);
} // namespace tensat
