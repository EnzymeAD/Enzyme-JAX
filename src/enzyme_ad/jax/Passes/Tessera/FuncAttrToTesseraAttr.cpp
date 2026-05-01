//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise the generic tessera_op attribute
// into a proper MLIR Tessera convert attribute on the operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_FUNCATTRTOTESSERAATTRPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

struct FuncAttrToTesseraAttrPass
    : public enzyme::tessera::impl::FuncAttrToTesseraAttrPassBase<
          FuncAttrToTesseraAttrPass> {
  using FuncAttrToTesseraAttrPassBase::FuncAttrToTesseraAttrPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();
    OpBuilder builder(ctx);

    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      StringAttr opAttr;
      bool isPure = false;

      if (auto attr = func->getAttrOfType<StringAttr>("tessera_op")) {
        opAttr = attr;
      } else if (auto attr =
                     func->getAttrOfType<StringAttr>("pure_tessera_op")) {
        opAttr = attr;
        isPure = true;
      }

      if (opAttr) {
        StringRef raw = opAttr.getValue();

        // Parse op name (everything before the '(')
        StringRef opName = raw.take_while([](char c) { return c != '('; });

        // Parse args in parentheses
        StringRef argList = raw.slice(raw.find('(') + 1, raw.find(')'));
        SmallVector<StringRef> argParts;
        argList.split(argParts, ',');

        SmallVector<bool> byRefArgs;

        for (auto arg : argParts) {
          arg = arg.trim();
          if (arg.contains(":byref") || arg.contains(": byref")) {
            byRefArgs.push_back(true);
          } else {
            byRefArgs.push_back(false);
          }
        }

        // Parse sizes of args
        SmallVector<int64_t> sizes;
        StringRef sizeStr = raw.substr(raw.find(')') + 1);
        if (!sizeStr.empty() && sizeStr.consume_front(":")) {
          SmallVector<StringRef> sizeParts;
          sizeStr.split(sizeParts, ',');
          for (auto s : sizeParts) {
            int64_t size;
            s.trim().getAsInteger(10, size);
            sizes.push_back(size);
          }
        }

        // Make sure number of arguments matches number of sizes provided
        if (byRefArgs.size() != sizes.size()) {
          func->emitError("tessera: number of arguments (")
              << byRefArgs.size() << ") does not match number of sizes ("
              << sizes.size() << ")";
          return;
        }

        auto tesseraAttr = enzyme::tessera::ConvertAttr::get(
            builder.getContext(), opName.str(),
            DenseBoolArrayAttr::get(builder.getContext(), byRefArgs),
            DenseI64ArrayAttr::get(builder.getContext(), sizes), isPure);

        func->setAttr("tessera.convert", tesseraAttr);
        func->removeAttr("tessera_op");
        func->removeAttr("pure_tessera_op");
      }
    }
  }
};

} // namespace
