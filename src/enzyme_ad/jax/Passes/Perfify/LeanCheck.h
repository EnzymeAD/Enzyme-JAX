#ifndef PERFIFY_LEAN_CHECK_H
#define PERFIFY_LEAN_CHECK_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir {
namespace enzyme {
namespace perfify {

/// Checks each of `names`, fully qualified Lean names, in the Lean file `text`
/// by running `lake -d <projectDir> env lean --json` on it with the `lake`
/// executable. Returns one entry per name: empty if the theorem is proven
/// using only the standard axioms, else why not. Fails if Lean cannot be run
/// or the imports of `text` cannot be loaded.
llvm::Expected<SmallVector<std::string>>
checkLeanTheorems(StringRef text, ArrayRef<std::string> names, StringRef lake,
                  StringRef projectDir);

} // namespace perfify
} // namespace enzyme
} // namespace mlir

#endif
