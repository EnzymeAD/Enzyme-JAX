#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ExecutionEngine/Orc/CoreContainers.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"

namespace enzymexla {

struct CallInfo {
  void (*run)(void *, void *, void **);
  void *(*init)();
};

bool init_jit();
CallInfo CompileHostModule(std::string &key, mlir::ModuleOp modOp,
                           bool compileInit, bool dump_final_module,
                           int lib_counter);
llvm::Error map_symbol(const char *name, void *symbol);
llvm::Expected<void *> lookup_symbol(const char *name);

} // namespace enzymexla
