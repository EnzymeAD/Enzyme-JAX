//===- JITRegionToJITCall.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SetVector.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"

#define DEBUG_TYPE "jit-region-to-jit-call"

namespace mlir::enzyme {
#define GEN_PASS_DEF_JITREGIONTOJITCALLPASS
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace mlir::enzyme

using namespace mlir;

namespace {

struct JITRegionToJITCallPass
    : public mlir::enzyme::impl::JITRegionToJITCallPassBase<
          JITRegionToJITCallPass> {
  using mlir::enzyme::impl::JITRegionToJITCallPassBase<
      JITRegionToJITCallPass>::JITRegionToJITCallPassBase;

  // Runs the gpu_wrappers of `fn` on the stream XLA runs the jit_call on, as
  // lower-kernel does for kernel_calls: convert-parallel-to-gpu1 gives the
  // stream an async.execute waits on to the launches it contains, and
  // lower-jit only compiles a call for the device when it uses that stream.
  static void runOnStream(func::FuncOp fn) {
    SmallVector<enzymexla::GPUWrapperOp> wrappers;
    fn.walk([&](enzymexla::GPUWrapperOp wrapper) {
      if (!wrapper->getParentOfType<async::ExecuteOp>() && wrapper->use_empty())
        wrappers.push_back(wrapper);
    });
    if (wrappers.empty())
      return;

    OpBuilder builder = OpBuilder::atBlockBegin(&fn.getBody().front());
    Value stream = enzymexla::GetStreamOp::create(
        builder, fn.getLoc(), gpu::AsyncTokenType::get(fn.getContext()));
    Value token = enzymexla::StreamToTokenOp::create(
        builder, fn.getLoc(), async::TokenType::get(fn.getContext()), stream);
    for (enzymexla::GPUWrapperOp wrapper : wrappers) {
      builder.setInsertionPoint(wrapper);
      auto execute = async::ExecuteOp::create(
          builder, wrapper.getLoc(), TypeRange(), ValueRange(token),
          ValueRange(), [](OpBuilder &b, Location loc, ValueRange) {
            async::YieldOp::create(b, loc, ValueRange());
          });
      wrapper->moveBefore(execute.getBody()->getTerminator());
    }
  }

  // Replaces `region` by a jit_call of a new function holding its body. The
  // block arguments of the region, its buffers, become the arguments of the
  // function.
  LogicalResult outlineJITRegion(enzymexla::JITRegionOp region,
                                 SymbolTable &symbolTable) {
    Block *body = region.getBodyBlock();

    // The function is isolated from above: the constants the body uses are
    // cloned into it, anything else defined above is not supported.
    llvm::SetVector<Value> captured;
    getUsedValuesDefinedAbove(region.getBody(), captured);
    for (Value value : captured) {
      Operation *def = value.getDefiningOp();
      if (!def || !matchPattern(def, m_Constant()))
        return region.emitError(
                   "cannot outline a jit_region using a non-constant value "
                   "defined above it: ")
               << value;
    }

    auto parent = region->getParentOfType<FunctionOpInterface>();
    std::string name =
        ((parent ? parent.getName() : StringRef("jit_region")) + "_jit_region")
            .str();

    OpBuilder builder(getOperation().getBodyRegion());
    builder.setInsertionPointToEnd(getOperation().getBody());
    auto fn = func::FuncOp::create(
        builder, region.getLoc(), name,
        builder.getFunctionType(body->getArgumentTypes(), {}));
    fn.setPrivate();
    symbolTable.insert(fn);

    fn.getBody().takeBody(region.getBody());
    Block &entry = fn.getBody().front();

    builder.setInsertionPointToStart(&entry);
    IRMapping mapping;
    for (Value value : captured) {
      Operation *clone = builder.clone(*value.getDefiningOp(), mapping);
      replaceAllUsesInRegionWith(
          value, clone->getResult(cast<OpResult>(value).getResultNumber()),
          fn.getBody());
    }

    Operation *yield = entry.getTerminator();
    builder.setInsertionPoint(yield);
    func::ReturnOp::create(builder, yield->getLoc());
    yield->erase();
    runOnStream(fn);

    builder.setInsertionPoint(region);
    auto call = enzymexla::JITCallOp::create(
        builder, region.getLoc(), region.getResultTypes(),
        SymbolRefAttr::get(fn.getSymNameAttr()), region.getInputs(),
        /*backend_config=*/builder.getStringAttr(""),
        region.getOperandLayoutsAttr(), region.getOperandAttrsAttr(),
        region.getResultAttrsAttr(), region.getOutputOperandAliasesAttr(),
        region.getXlaSideEffectFreeAttr());

    region.replaceAllUsesWith(call.getResults());
    region.erase();
    return success();
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    SmallVector<enzymexla::JITRegionOp> regions;
    getOperation().walk(
        [&](enzymexla::JITRegionOp region) { regions.push_back(region); });

    bool failed = false;
    for (enzymexla::JITRegionOp region : regions)
      failed |= mlir::failed(outlineJITRegion(region, symbolTable));

    if (failed)
      signalPassFailure();
  }
};

} // namespace
