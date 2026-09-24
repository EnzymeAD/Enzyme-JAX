#ifndef ENZYMEXLA_PASSES_COMM_TYPE_CONVERSION_H
#define ENZYMEXLA_PASSES_COMM_TYPE_CONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::comm {

struct StablehloTypeConverter : public TypeConverter {
public:
  StablehloTypeConverter();
};

} // namespace mlir::comm

#endif // ENZYMEXLA_PASSES_COMM_TYPE_CONVERSION_H
