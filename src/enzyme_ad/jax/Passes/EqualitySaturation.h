#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "rust/cxx.h"

namespace tensat {
enum class Type : uint8_t;
enum class Ops : uint8_t;
struct Vector;
struct Node;
struct Matrix;
struct Tensor;
struct Node;
struct CppGraphConverter;

/**
 * Functions exposed to Rust (Tensat) for getting the cost of new operations.
 */

<<<<<<< HEAD
rust::Vec<uint64_t> get_cost(Ops op, rust::Vec<tensat::Tensor> operands,
                             rust::Vec<tensat::Vector> other_vector_args,
                             rust::Vec<int64_t> int_args,
                             rust::Vec<tensat::Matrix> matrix_args);
=======
uint64_t get_cost(Ops op, rust::Vec<tensat::Tensor> operands,
                  rust::Vec<tensat::Vector> other_vector_args,
                  rust::Vec<int64_t> int_args,
                  rust::Vec<tensat::Matrix> matrix_args);

uint64_t get_graph_cost(rust::Vec<tensat::Node> nodes);

mlir::Type newTensorType(mlir::OpBuilder &builder, Tensor tensor); 
mlir::Type tensatTypeToMlirType(mlir::OpBuilder &builder, Type type);
>>>>>>> 7a954d5 (End-to-end cost measurement)

rust::Vec<Tensor> get_shape(Ops op, rust::Vec<tensat::Tensor> operands,
                            rust::Vec<tensat::Vector> other_vector_args,
                            rust::Vec<int64_t> int_args,
                            rust::Vec<tensat::Matrix> matrix_args);

rust::Box<CppGraphConverter>
apply_mlir_rewrite(rust::Vec<tensat::Node> nodes,
                   rust::Vec<tensat::Tensor> roots);

} // namespace tensat
