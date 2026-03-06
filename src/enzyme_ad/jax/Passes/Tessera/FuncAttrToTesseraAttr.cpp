//===----------------------------------------------------------------------===//
//
// This file implements a pass to raise the generic tessera_op attribute
// into a proper MLIR Tessera convert attribute on the operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    OpBuilder builder(module.getContext());

    for (auto func : module.getOps<LLVM::LLVMFuncOp>()) {
      if (auto attr = func->getAttrOfType<StringAttr>("tessera_op")) {

        auto tesseraAttr = enzyme::tessera::ConvertAttr::get(
            builder.getContext(), attr.getValue().str());

        func->setAttr("tessera.convert", tesseraAttr);
        func->removeAttr("tessera_op");
      }
    }
  }
};

} // namespace