//===- SQLRaising.cpp - Raise PostgreSQL calls and SQL strings ------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Adapted from Polygeist's SQLRaising.cpp. Preserve the source ABI directly;
// do not turn result text into integers or pointers into fixed-width handles.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Dialect/SQL/Parser.h"
#include "src/enzyme_ad/jax/Passes/SQL/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>
#include <string>

namespace mlir::sql {
#define GEN_PASS_DEF_SQLRAISINGPASS
#include "src/enzyme_ad/jax/Passes/SQL/Passes.h.inc"
} // namespace mlir::sql

using namespace mlir;
using namespace mlir::sql;

namespace {
bool isHandle(Type type) {
  return isa<IndexType, LLVM::LLVMPointerType, MemRefType, UnrankedMemRefType>(
      type);
}

bool isText(Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  auto memref = dyn_cast<BaseMemRefType>(type);
  return memref && memref.getElementType().isInteger(8);
}

bool isCount(Type type) {
  return isa<IndexType>(type) || type.isSignlessInteger();
}

// Return a known immutable string, following only address-preserving wrappers.
// Unknown/dynamic strings remain sql.unparsed. No load from mutable globals is
// folded. Bound traversal to avoid cycles in malformed input.
std::optional<std::string> constantQuery(Value value) {
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (auto string = value.getDefiningOp<SQLConstantStringOp>()) {
      StringRef bytes = string.getInput();
      size_t end = bytes.find('\0');
      return (end == StringRef::npos ? bytes : bytes.take_front(end)).str();
    }
    if (auto bridge = value.getDefiningOp<enzymexla::Memref2PointerOp>()) {
      value = bridge.getSource();
      continue;
    }
    if (auto bridge = value.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
      value = bridge.getSource();
      continue;
    }
    if (auto gep = value.getDefiningOp<LLVM::GEPOp>()) {
      for (auto index : gep.getIndices()) {
        if (auto dynamic = dyn_cast_if_present<Value>(index)) {
          if (!matchPattern(dynamic, m_Zero()))
            return std::nullopt;
        } else if (!cast<IntegerAttr>(index).getValue().isZero()) {
          return std::nullopt;
        }
      }
      value = gep.getBase();
      continue;
    }
    auto address = value.getDefiningOp<LLVM::AddressOfOp>();
    if (!address)
      return std::nullopt;
    auto global = SymbolTable::lookupNearestSymbolFrom<LLVM::GlobalOp>(
        address, address.getGlobalNameAttr());
    if (!global || !global.getConstant() ||
        (global.getLinkage() != LLVM::Linkage::Private &&
         global.getLinkage() != LLVM::Linkage::Internal) ||
        global->hasAttr("externally_initialized"))
      return std::nullopt;
    auto string = dyn_cast_or_null<StringAttr>(global.getValueAttr());
    if (!string)
      return std::nullopt;
    StringRef bytes = string.getValue();
    size_t end = bytes.find('\0');
    // A PostgreSQL query argument is a C string. Without a terminator the
    // object's bytes do not establish the runtime query, so leave it dynamic.
    if (end == StringRef::npos)
      return std::nullopt;
    return bytes.take_front(end).str();
  }
  return std::nullopt;
}

LogicalResult parseUnparsed(UnparsedOp op, IRRewriter &rewriter) {
  auto text = constantQuery(op.getInput());
  if (!text)
    return success();
  rewriter.setInsertionPoint(op);
  auto query = parseSQL(op.getLoc(), rewriter, *text);
  if (failed(query))
    return failure();
  rewriter.replaceOp(op, *query);
  return success();
}

// Original call properties (including cconv and argument/result attributes),
// dialect, and signature are retained for a future ABI-preserving lowering.
// The callee declaration is left intact as well.
void recordSourceCall(Operation *source, Operation *target, StringRef kind,
                      ValueRange args, OpBuilder &builder) {
  NamedAttrList metadata;
  metadata.append("kind", builder.getStringAttr(kind));
  metadata.append("attributes", source->getAttrDictionary());
  if (auto call = dyn_cast<LLVM::CallOp>(source))
    metadata.append(
        "calling_convention",
        LLVM::CConvAttr::get(builder.getContext(), call.getCConv()));
  metadata.append("type", TypeAttr::get(builder.getFunctionType(
                              args.getTypes(), source->getResultTypes())));
  target->setAttr("sql.source_call",
                  metadata.getDictionary(builder.getContext()));
}

LogicalResult raiseCall(Operation *call, StringRef callee, ValueRange args,
                        StringRef kind, IRRewriter &rewriter) {
  if (callee != "PQexec" && callee != "PQntuples" && callee != "PQgetvalue")
    return success();

  unsigned expected = callee == "PQexec" ? 2 : callee == "PQntuples" ? 1 : 3;
  auto badSignature = [&]() -> LogicalResult {
    return call->emitError() << "unsupported " << callee << " signature";
  };
  if (args.size() != expected || call->getNumResults() != 1 ||
      !isHandle(args[0].getType()))
    return badSignature();
  Type result = call->getResult(0).getType();
  if (callee == "PQexec") {
    if (!isText(args[1].getType()) || !isHandle(result))
      return badSignature();
  } else if (callee == "PQntuples") {
    if (!isCount(result))
      return badSignature();
  } else if (!isCount(args[1].getType()) || !isCount(args[2].getType()) ||
             !isText(result)) {
    return badSignature();
  }

  rewriter.setInsertionPoint(call);
  Operation *replacement;
  if (callee == "PQexec") {
    Value query;
    if (auto text = constantQuery(args[1])) {
      auto parsed = parseSQL(call->getLoc(), rewriter, *text);
      if (failed(parsed))
        return failure();
      query = *parsed;
    } else {
      query = rewriter.create<UnparsedOp>(
          call->getLoc(), ExprType::get(rewriter.getContext()), args[1]);
    }
    replacement =
        rewriter.create<ExecuteOp>(call->getLoc(), result, args[0], query);
  } else if (callee == "PQntuples") {
    replacement =
        rewriter.create<NumResultsOp>(call->getLoc(), result, args[0]);
  } else {
    // libpq takes (handle, row, column); SQL IR retains (handle, column, row).
    // The result is the original text pointer/memref, with no atoi/itoa calls.
    replacement = rewriter.create<GetValueOp>(call->getLoc(), result, args[0],
                                              args[2], args[1]);
  }
  recordSourceCall(call, replacement, kind, args, rewriter);
  rewriter.replaceOp(call, replacement->getResults());
  return success();
}

struct SQLRaisingPass : mlir::sql::impl::SQLRaisingPassBase<SQLRaisingPass> {
  using SQLRaisingPassBase::SQLRaisingPassBase;

  void runOnOperation() override {
    SmallVector<Operation *> worklist;
    getOperation().walk([&](Operation *op) {
      if (isa<LLVM::CallOp, func::CallOp, UnparsedOp>(op))
        worklist.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    for (Operation *op : worklist) {
      LogicalResult result = success();
      if (auto unparsed = dyn_cast<UnparsedOp>(op)) {
        result = parseUnparsed(unparsed, rewriter);
      } else if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        auto callee = call.getCalleeAttr();
        if (!callee)
          continue; // Indirect calls have no established PostgreSQL identity.
        StringRef name = callee.getValue();
        if (name != "PQexec" && name != "PQntuples" && name != "PQgetvalue")
          continue;
        auto fn = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
            call, callee);
        // A local definition may have unrelated semantics. Operand bundles
        // carry SSA values which SQL ops cannot represent yet; keep such calls.
        if (!fn || !fn.isExternal() || fn.getFunctionType().isVarArg() ||
            call.getCalleeOperands().size() != call->getNumOperands())
          continue;
        result =
            raiseCall(call, name, call.getCalleeOperands(), "llvm", rewriter);
      } else if (auto call = dyn_cast<func::CallOp>(op)) {
        auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        if (!fn || !fn.isExternal())
          continue;
        result = raiseCall(call, call.getCallee(), call.getOperands(), "func",
                           rewriter);
      }
      if (failed(result))
        return signalPassFailure();
    }
  }
};
} // namespace
