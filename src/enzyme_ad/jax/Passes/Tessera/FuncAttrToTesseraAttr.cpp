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

using namespace mlir;

namespace {

struct FuncAttrToTesseraAttrPass
    : public PassWrapper<FuncAttrToTesseraAttrPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const final { return "func-attr-to-tessera-attr"; }

  StringRef getDescription() const final {
    return "Convert generic tessera_op func attribute into tessera.convert "
           "attribute.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<enzyme::tessera::TesseraDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());

    for (auto llvmfunc : module.getOps<LLVM::LLVMFuncOp>()) {
      if (auto attr = llvmfunc->getAttrOfType<StringAttr>("tessera_op")) {

        auto tesseraAttr = enzyme::tessera::ConvertAttr::get(
            builder.getContext(), attr.getValue().str());

        llvmfunc->setAttr("tessera.convert", tesseraAttr);
        llvmfunc->removeAttr("tessera_op");
      }
    }

    for (auto func : module.getOps<func::FuncOp>()) {
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

namespace mlir::enzyme::tessera {

std::unique_ptr<mlir::Pass> createFuncAttrToTesseraAttrPass() {
  return std::make_unique<FuncAttrToTesseraAttrPass>();
}

} // namespace mlir::enzyme::tessera
