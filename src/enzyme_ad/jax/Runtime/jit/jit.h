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

bool InitJIT();
CallInfo CompileHostModule(std::string &key, mlir::ModuleOp modOp,
                           bool compileInit, bool dump_final_module,
                           int lib_counter);
llvm::Error MapSymbol(const char *name, void *symbol);
llvm::Expected<void *> LookupSymbol(const char *name);

} // namespace enzymexla
