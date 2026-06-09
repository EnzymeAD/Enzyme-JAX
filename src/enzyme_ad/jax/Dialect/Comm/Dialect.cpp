#include "Dialect.h"
#include "Ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#include "src/enzyme_ad/jax/Dialect/Comm/CommDialect.cpp.inc"

#include "src/enzyme_ad/jax/Dialect/Comm/AttrEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/AttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/Types.cpp.inc"

void mlir::enzymexla::comm::CommDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Comm/AttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/enzyme_ad/jax/Dialect/Comm/Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Comm/MPIOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "src/enzyme_ad/jax/Dialect/Comm/NCCLOps.cpp.inc"
      >();
}
