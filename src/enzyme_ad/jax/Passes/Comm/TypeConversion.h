#ifndef ENZYMEXLA_PASSES_COMM_TYPE_CONVERSION_H
#define ENZYMEXLA_PASSES_COMM_TYPE_CONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::comm {

struct StablehloTypeConverter : public TypeConverter {
public:
  // `requestWidth` is the bit width of `MPI_Request`, which is
  // implementation-defined (e.g. `int` in MPICH, a pointer in Open MPI)
  StablehloTypeConverter(unsigned requestWidth = 64);
};

} // namespace mlir::comm

#endif // ENZYMEXLA_PASSES_COMM_TYPE_CONVERSION_H
