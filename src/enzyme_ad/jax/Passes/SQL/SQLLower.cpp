//===- SQLLower.cpp - Serialize SQL and restore PostgreSQL calls ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Adapted from Polygeist's SQLLower.cpp. The original declarations and saved
// call metadata define the ABI; never guess pointer widths or parse result text
// as numbers. Work on a clone so any unsupported case leaves the input intact.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "src/enzyme_ad/jax/Dialect/SQL/Parser.h"
#include "src/enzyme_ad/jax/Passes/SQL/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::sql {
#define GEN_PASS_DEF_SQLLOWERPASS
#include "src/enzyme_ad/jax/Passes/SQL/Passes.h.inc"
} // namespace mlir::sql

using namespace mlir;
using namespace mlir::sql;

namespace {
bool isSQL(Operation *op) {
  return op->getName().getDialectNamespace() == "sql";
}

bool isText(Type type) {
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  auto memref = dyn_cast<BaseMemRefType>(type);
  return memref && memref.getElementType().isInteger(8);
}

struct Text {
  std::optional<std::string> constant;
  Value dynamic;
};

class Serializer {
public:
  explicit Serializer(Operation *root) : root(root) {}

  FailureOr<std::string> expression(Value value, unsigned depth = 0) {
    if (depth > 128)
      return error("expression nesting limit exceeded");
    Operation *op = value.getDefiningOp();
    if (!op)
      return error("cannot serialize an SQL expression block argument");
    if (auto integer = dyn_cast<IntOp>(op)) {
      int64_t parsed;
      if (integer.getExpr().getAsInteger(10, parsed))
        return error("invalid signed 64-bit SQL integer literal");
      return integer.getExpr().str();
    }
    if (auto column = dyn_cast<ColumnOp>(op))
      return column.getExpr().str();
    if (auto table = dyn_cast<TableOp>(op))
      return table.getExpr().str();
    if (isa<AllColumnsOp>(op))
      return std::string("*");
    if (isa<ConstantBoolOp>(op))
      return std::string("TRUE");
    if (auto predicate = dyn_cast<CalcBoolOp>(op))
      return predicate.getExpr().empty()
                 ? std::string("TRUE")
                 : "(" + predicate.getExpr().str() + ")";
    if (auto where = dyn_cast<WhereOp>(op))
      return expression(where.getExpr(), depth + 1);

    std::string token;
    if (auto arithmetic = dyn_cast<ArithOp>(op)) {
      token = arithmetic.getOp().str();
      if (token != "+" && token != "-" && token != "*" && token != "/" &&
          token != "%")
        return error("unsupported arithmetic operator");
    } else if (auto comparison = dyn_cast<BoolArithOp>(op)) {
      token = comparison.getOp().str();
      if (token != "=" && token != "!=" && token != "<>" && token != "<" &&
          token != "<=" && token != ">" && token != ">=")
        return error("unsupported comparison operator");
    } else if (isa<AndOp>(op)) {
      token = "AND";
    } else if (isa<OrOp>(op)) {
      token = "OR";
    } else {
      return error("unsupported SQL expression producer");
    }
    auto left = expression(op->getOperand(0), depth + 1);
    auto right = expression(op->getOperand(1), depth + 1);
    if (failed(left) || failed(right))
      return failure();
    // Parenthesize every binary node, including right children of subtraction
    // and division. Equal precedence alone is insufficient to preserve a tree.
    return "(" + *left + " " + token + " " + *right + ")";
  }

  FailureOr<std::string> query(SelectOp select) {
    if (select.getColumns().empty())
      return error("SELECT requires at least one projection");
    std::string text = "SELECT ";
    bool first = true;
    for (Value column : select.getColumns()) {
      auto part = expression(column);
      if (failed(part))
        return failure();
      if (!first)
        text += ", ";
      text += *part;
      first = false;
    }
    auto table = select.getTable().getDefiningOp<TableOp>();
    auto where = select.getWhere().getDefiningOp<WhereOp>();
    if (!table || !where)
      return error("SELECT requires sql.table and sql.where producers");
    if (!table.getExpr().empty())
      text += " FROM " + table.getExpr().str();
    Value predicate = where.getExpr();
    auto raw = predicate.getDefiningOp<CalcBoolOp>();
    if (!predicate.getDefiningOp<ConstantBoolOp>() &&
        !(raw && raw.getExpr().empty())) {
      auto part = expression(predicate);
      if (failed(part))
        return failure();
      text += " WHERE " + *part;
    }
    if (select.getLimit() >= 0)
      text += " LIMIT " + std::to_string(select.getLimit());
    if (failed(validate(text)))
      return failure();
    return text;
  }

  FailureOr<Text> text(Value value, unsigned depth = 0) {
    if (depth > 128) {
      (void)error("string nesting limit exceeded");
      return failure();
    }
    if (auto string = value.getDefiningOp<SQLConstantStringOp>())
      return Text{string.getInput().str(), {}};
    if (auto unparsed = value.getDefiningOp<UnparsedOp>())
      return text(unparsed.getInput(), depth + 1);
    if (auto conversion = value.getDefiningOp<SQLToStringOp>()) {
      Value expr = conversion.getExpr();
      if (expr.getDefiningOp<UnparsedOp>())
        return text(expr, depth + 1);
      if (auto select = expr.getDefiningOp<SelectOp>()) {
        auto result = query(select);
        if (failed(result))
          return failure();
        return Text{*result, {}};
      }
      auto result = expression(expr);
      if (failed(result))
        return failure();
      if (auto table = expr.getDefiningOp<TableOp>())
        if (table.getExpr().empty())
          return Text{std::string{}, {}};
      std::string prefix =
          expr.getDefiningOp<WhereOp>() ? "SELECT 1 WHERE " : "SELECT ";
      if (failed(validate(prefix + *result)))
        return failure();
      return Text{*result, {}};
    }
    if (auto conversion = value.getDefiningOp<SQLBoolToStringOp>()) {
      auto result = expression(conversion.getExpr());
      if (failed(result) || failed(validate("SELECT 1 WHERE " + *result)))
        return failure();
      return Text{*result, {}};
    }
    if (auto concat = value.getDefiningOp<SQLStringConcatOp>()) {
      if (concat.getNumOperands() == 1)
        return text(concat->getOperand(0), depth + 1);
      std::string joined;
      for (Value operand : concat->getOperands()) {
        auto part = text(operand, depth + 1);
        if (failed(part))
          return failure();
        if (!part->constant) {
          (void)error("dynamic string concatenation is not supported");
          return failure();
        }
        joined += *part->constant;
      }
      return Text{joined, {}};
    }
    if (auto *producer = value.getDefiningOp())
      if (isSQL(producer)) {
        (void)error("unsupported SQL string producer");
        return failure();
      }
    if (!isText(value.getType())) {
      (void)error("dynamic strings require a pointer or byte memref");
      return failure();
    }
    return Text{std::nullopt, value};
  }

private:
  Operation *root;
  FailureOr<std::string> error(StringRef message) {
    root->emitError() << "sql-lower: " << message;
    return failure();
  }
  LogicalResult validate(StringRef queryText) {
    // Validate textual leaves too. This checks syntax, not database schemas or
    // expression types. The temporary tree is never inserted in input IR.
    Block block;
    OpBuilder builder(root->getContext());
    builder.setInsertionPointToEnd(&block);
    return failed(parseSQL(root->getLoc(), builder, queryText)) ? failure()
                                                                : success();
  }
};

struct CallInfo {
  FunctionType type;
  Operation *declaration;
  DictionaryAttr attributes;
  LLVM::CConvAttr convention;
};

FailureOr<CallInfo> callInfo(Operation *op, StringRef name) {
  auto fail = [&](StringRef message) -> FailureOr<CallInfo> {
    op->emitError() << "sql-lower: " << message;
    return failure();
  };
  auto symbol = FlatSymbolRefAttr::get(op->getContext(), name);
  Operation *declaration = SymbolTable::lookupNearestSymbolFrom(op, symbol);
  FunctionType declared;
  LLVM::CConvAttr convention;
  StringRef kind;
  if (auto fn = dyn_cast_or_null<LLVM::LLVMFuncOp>(declaration)) {
    if (!fn.isExternal() || fn.getFunctionType().isVarArg() ||
        isa<LLVM::LLVMVoidType>(fn.getFunctionType().getReturnType()))
      return fail("requires a nonvariadic external PostgreSQL declaration");
    declared =
        FunctionType::get(op->getContext(), fn.getFunctionType().getParams(),
                          {fn.getFunctionType().getReturnType()});
    convention = LLVM::CConvAttr::get(op->getContext(), fn.getCConv());
    kind = "llvm";
  } else if (auto fn = dyn_cast_or_null<func::FuncOp>(declaration)) {
    if (!fn.isExternal())
      return fail("requires an external PostgreSQL declaration");
    declared = fn.getFunctionType();
    kind = "func";
  } else {
    return fail(
        "requires an existing PostgreSQL declaration; ABI is not inferred");
  }

  DictionaryAttr attributes;
  if (Attribute saved = op->getAttr("sql.source_call")) {
    auto metadata = dyn_cast<DictionaryAttr>(saved);
    if (!metadata)
      return fail("sql.source_call must be a dictionary");
    auto savedKind = metadata.getAs<StringAttr>("kind");
    auto savedType = metadata.getAs<TypeAttr>("type");
    attributes = metadata.getAs<DictionaryAttr>("attributes");
    if (!savedKind || savedKind.getValue() != kind || !savedType ||
        savedType.getValue() != declared || !attributes ||
        attributes.getAs<FlatSymbolRefAttr>("callee") != symbol)
      return fail("sql.source_call disagrees with the PostgreSQL declaration");
    for (StringRef key : {StringRef("arg_attrs"), StringRef("res_attrs")}) {
      if (Attribute attr = attributes.get(key)) {
        auto array = dyn_cast<ArrayAttr>(attr);
        unsigned count = key == "arg_attrs" ? declared.getNumInputs()
                                            : declared.getNumResults();
        if (!array || (!array.empty() && array.size() != count) ||
            llvm::any_of(array, [](Attribute entry) {
              return !isa<DictionaryAttr>(entry);
            }))
          return fail("invalid saved argument/result attribute array");
      }
    }
    if (kind == "llvm") {
      auto savedCC = metadata.getAs<LLVM::CConvAttr>("calling_convention");
      auto attrCC = attributes.getAs<LLVM::CConvAttr>("CConv");
      if (!savedCC || savedCC != convention || (attrCC && attrCC != savedCC))
        return fail("sql.source_call calling convention mismatch");
    }
  }
  if (declared.getNumResults() != 1 ||
      declared.getResult(0) != op->getResult(0).getType())
    return fail(
        "PostgreSQL result type mismatch; no implicit conversion is made");
  return CallInfo{declared, declaration, attributes, convention};
}

// Materialize NUL-terminated text in exactly the requested address space or
// memref memory space. SymbolTable insertion provides unique stable names.
FailureOr<Value> materialize(Operation *root, StringRef text, Type type,
                             IRRewriter &rewriter,
                             DictionaryAttr callAttributes = {},
                             DictionaryAttr declarationAttributes = {}) {
  auto fail = [&](StringRef message) -> FailureOr<Value> {
    root->emitError() << "sql-lower: " << message;
    return failure();
  };
  auto pointer = dyn_cast<LLVM::LLVMPointerType>(type);
  auto memref = dyn_cast<MemRefType>(type);
  auto unranked = dyn_cast<UnrankedMemRefType>(type);
  if (!isText(type) ||
      (memref && (memref.getRank() != 1 || !memref.getLayout().isIdentity())))
    return fail(
        "string materialization requires a pointer or identity byte memref");

  uint64_t size = text.size() + 1;
  uint64_t alignment = 0;
  SmallVector<DictionaryAttr> requirements;
  if (declarationAttributes)
    requirements.push_back(declarationAttributes);
  if (callAttributes) {
    if (auto args = callAttributes.getAs<ArrayAttr>("arg_attrs")) {
      if (!args.empty() && args.size() != 2)
        return fail("invalid PQexec argument attributes");
      if (!args.empty()) {
        auto attrs = dyn_cast<DictionaryAttr>(args[1]);
        if (!attrs)
          return fail("invalid query argument attributes");
        requirements.push_back(attrs);
      }
    }
  }
  for (DictionaryAttr attrs : requirements) {
    for (NamedAttribute attr : attrs) {
      StringRef name = attr.getName().getValue();
      if (name == "llvm.align" || name == "llvm.dereferenceable" ||
          name == "llvm.dereferenceable_or_null") {
        auto integer = dyn_cast<IntegerAttr>(attr.getValue());
        if (!integer || integer.getValue().isNegative() ||
            integer.getValue().getActiveBits() > 63)
          return fail("invalid query buffer size/alignment attribute");
        uint64_t value = integer.getValue().getZExtValue();
        if (name == "llvm.align") {
          if (!value || (value & (value - 1)))
            return fail("query alignment must be a positive power of two");
          alignment = std::max(alignment, value);
        } else {
          size = std::max(size, value);
        }
      } else if (name != "llvm.nonnull" && name != "llvm.noundef" &&
                 name != "llvm.readonly" && name != "llvm.nocapture" &&
                 name != "llvm.noalias") {
        return fail("unsupported attribute on a rematerialized query argument");
      }
    }
  }
  if (memref && !memref.isDynamicDim(0)) {
    uint64_t extent = memref.getDimSize(0);
    if (size > extent)
      return fail("serialized query does not fit the declared static memref");
    size = extent;
  }
  // Avoid pathological allocations requested by malformed metadata/types.
  if (size > 16 * 1024 * 1024 || alignment > 16 * 1024 * 1024)
    return fail(
        "query buffer exceeds the supported 16 MiB materialization limit");
  std::string bytes = text.str();
  bytes.resize(size, '\0');
  ModuleOp module = root->getParentOfType<ModuleOp>();
  SymbolTable symbols(module);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  if (pointer) {
    auto array = LLVM::LLVMArrayType::get(rewriter.getI8Type(), size);
    auto global = LLVM::GlobalOp::create(rewriter, root->getLoc(), array, true,
                                         LLVM::Linkage::Private, "__sql_query",
                                         rewriter.getStringAttr(bytes),
                                         alignment, pointer.getAddressSpace());
    symbols.insert(global);
    rewriter.setInsertionPoint(root);
    return LLVM::AddressOfOp::create(rewriter, root->getLoc(), global)
        .getResult();
  }
  Attribute space =
      memref ? memref.getMemorySpace() : unranked.getMemorySpace();
  Type element = memref ? memref.getElementType() : unranked.getElementType();
  auto storage = MemRefType::get({static_cast<int64_t>(size)}, element,
                                 MemRefLayoutAttrInterface{}, space);
  auto data = DenseElementsAttr::get<char>(
      RankedTensorType::get(storage.getShape(), element),
      ArrayRef<char>(bytes.data(), bytes.size()));
  auto global = memref::GlobalOp::create(
      rewriter, root->getLoc(), rewriter.getStringAttr("__sql_query"),
      rewriter.getStringAttr("private"), TypeAttr::get(storage), data,
      rewriter.getUnitAttr(),
      alignment ? rewriter.getI64IntegerAttr(alignment) : IntegerAttr{});
  symbols.insert(global);
  rewriter.setInsertionPoint(root);
  Value result = memref::GetGlobalOp::create(rewriter, root->getLoc(), storage,
                                             global.getSymName());
  if (result.getType() != type)
    result = memref::CastOp::create(rewriter, root->getLoc(), type, result);
  return result;
}

LogicalResult lowerCall(Operation *op, IRRewriter &rewriter) {
  StringRef name = isa<ExecuteOp>(op)      ? "PQexec"
                   : isa<NumResultsOp>(op) ? "PQntuples"
                                           : "PQgetvalue";
  auto info = callInfo(op, name);
  if (failed(info))
    return failure();
  unsigned count = isa<ExecuteOp>(op) ? 2 : isa<NumResultsOp>(op) ? 1 : 3;
  if (info->type.getNumInputs() != count)
    return op->emitError("sql-lower: PostgreSQL argument count mismatch");
  if (isa<GetValueOp>(op) && !isText(op->getResult(0).getType()))
    return op->emitError(
        "sql-lower: PQgetvalue requires a text result, not numeric extraction");
  SmallVector<Value> args{op->getOperand(0)};
  if (auto execute = dyn_cast<ExecuteOp>(op)) {
    Serializer serializer(op);
    Text query;
    if (auto select = execute.getCommand().getDefiningOp<SelectOp>()) {
      auto serialized = serializer.query(select);
      if (failed(serialized))
        return failure();
      query.constant = *serialized;
    } else if (execute.getCommand().getDefiningOp<UnparsedOp>()) {
      auto serialized = serializer.text(execute.getCommand());
      if (failed(serialized))
        return failure();
      query = *serialized;
    } else {
      return op->emitError(
          "sql-lower: execute requires sql.select or sql.unparsed");
    }
    if (query.constant) {
      DictionaryAttr declarationAttrs;
      auto argAttrs =
          isa<LLVM::LLVMFuncOp>(info->declaration)
              ? cast<LLVM::LLVMFuncOp>(info->declaration).getArgAttrs()
              : cast<func::FuncOp>(info->declaration).getArgAttrs();
      if (argAttrs && argAttrs->size() > 1) {
        declarationAttrs = dyn_cast<DictionaryAttr>((*argAttrs)[1]);
        if (!declarationAttrs)
          return op->emitError("sql-lower: expected an argument attribute "
                               "dictionary for PQexec");
      }
      auto buffer = materialize(op, *query.constant, info->type.getInput(1),
                                rewriter, info->attributes, declarationAttrs);
      if (failed(buffer))
        return failure();
      args.push_back(*buffer);
    } else {
      args.push_back(query.dynamic);
    }
  } else if (auto get = dyn_cast<GetValueOp>(op)) {
    args.push_back(get.getRow());
    args.push_back(get.getColumn());
  }
  for (auto [value, expected] : llvm::zip(args, info->type.getInputs()))
    if (value.getType() != expected)
      return op->emitError("sql-lower: PostgreSQL argument type mismatch; "
                           "no implicit conversion is made");

  rewriter.setInsertionPoint(op);
  Operation *call;
  if (auto fn = dyn_cast<LLVM::LLVMFuncOp>(info->declaration)) {
    call = LLVM::CallOp::create(rewriter, op->getLoc(), op->getResultTypes(),
                                fn.getSymName(), args);
  } else {
    call = func::CallOp::create(rewriter, op->getLoc(),
                                cast<func::FuncOp>(info->declaration), args);
  }
  if (info->attributes) {
    NamedAttrList attributes(call->getAttrs());
    for (NamedAttribute attr : info->attributes)
      attributes.set(attr.getName(), attr.getValue());
    NamedAttrList properties, discardable;
    auto propertyNames = call->getName().getAttributeNames();
    for (NamedAttribute attr : attributes) {
      if (llvm::is_contained(propertyNames, attr.getName()))
        properties.append(attr.getName().getValue(), attr.getValue());
      else
        discardable.append(attr.getName().getValue(), attr.getValue());
    }
    // Decode inherent properties with diagnostics rather than unchecked
    // attribute setters, which may assert on malformed saved property types.
    if (failed(call->setPropertiesFromAttribute(
            properties.getDictionary(op->getContext()), [&]() {
              return op->emitError("sql-lower: invalid source call properties");
            })))
      return failure();
    call->setDiscardableAttrs(discardable.getDictionary(op->getContext()));
  }
  if (auto llvmCall = dyn_cast<LLVM::CallOp>(call))
    llvmCall.setCConv(info->convention.getCallingConv());
  rewriter.replaceOp(op, call->getResults());
  return success();
}

bool isStringOp(Operation *op) {
  return isa<SQLConstantStringOp, SQLToStringOp, SQLBoolToStringOp,
             SQLStringConcatOp>(op);
}

bool isExpressionOp(Operation *op) {
  return isa<IntOp, ColumnOp, WhereOp, BoolArithOp, CalcBoolOp, ArithOp, AndOp,
             OrOp, TableOp, ConstantBoolOp, SelectOp, AllColumnsOp, UnparsedOp>(
             op) ||
         isStringOp(op);
}

bool containsSQL(Type type) {
  if (isa<ExprType, BoolType>(type))
    return true;
  if (auto function = dyn_cast<FunctionType>(type))
    return llvm::any_of(function.getInputs(), containsSQL) ||
           llvm::any_of(function.getResults(), containsSQL);
  if (auto tuple = dyn_cast<TupleType>(type))
    return llvm::any_of(tuple.getTypes(), containsSQL);
  if (auto shaped = dyn_cast<ShapedType>(type))
    return containsSQL(shaped.getElementType());
  return false;
}

LogicalResult lowerModule(ModuleOp module) {
  IRRewriter rewriter(module.getContext());
  SmallVector<Operation *> calls;
  module.walk([&](Operation *op) {
    if (isa<ExecuteOp, NumResultsOp, GetValueOp>(op))
      calls.push_back(op);
  });
  for (Operation *op : calls)
    if (failed(lowerCall(op, rewriter)))
      return failure();

  struct StringPlan {
    Operation *op;
    Text text;
  };
  SmallVector<StringPlan> strings;
  bool failedPlan = false;
  module.walk([&](Operation *op) {
    if (!isStringOp(op) || failedPlan)
      return;
    if (llvm::none_of(op->getResult(0).getUsers(),
                      [](Operation *user) { return !isSQL(user); }))
      return;
    auto text = Serializer(op).text(op->getResult(0));
    if (failed(text)) {
      failedPlan = true;
      return;
    }
    strings.push_back({op, *text});
  });
  if (failedPlan)
    return failure();
  for (auto &plan : strings) {
    Value value = plan.text.dynamic;
    if (plan.text.constant) {
      auto buffer = materialize(plan.op, *plan.text.constant,
                                plan.op->getResult(0).getType(), rewriter);
      if (failed(buffer))
        return failure();
      value = *buffer;
    }
    if (value.getType() != plan.op->getResult(0).getType())
      return plan.op->emitError("sql-lower: dynamic string type mismatch");
    rewriter.replaceOp(plan.op, value);
  }

  // Remove the consumed expression/string trees, not arbitrary program IR.
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    module.walk([&](Operation *op) {
      if (isExpressionOp(op) && op->use_empty())
        dead.push_back(op);
    });
    for (Operation *op : dead) {
      rewriter.eraseOp(op);
      changed = true;
    }
  }
  auto checkRemaining = [&](Operation *op) -> WalkResult {
    if (isSQL(op)) {
      op->emitError(
          "sql-lower: unsupported escaping SQL expression or operation");
      return WalkResult::interrupt();
    }
    bool escaping = llvm::any_of(op->getOperandTypes(), containsSQL) ||
                    llvm::any_of(op->getResultTypes(), containsSQL);
    if (auto fn = dyn_cast<func::FuncOp>(op)) {
      escaping |= containsSQL(fn.getFunctionType());
    }
    for (Region &region : op->getRegions())
      for (Block &block : region)
        escaping |= llvm::any_of(block.getArgumentTypes(), containsSQL);
    if (escaping) {
      op->emitError("sql-lower: SQL types cannot escape into function "
                    "signatures or non-SQL operations");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  auto result = module.walk<WalkOrder::PreOrder>(checkRemaining);
  return failure(result.wasInterrupted());
}

struct SQLLowerPass : mlir::sql::impl::SQLLowerPassBase<SQLLowerPass> {
  using SQLLowerPassBase::SQLLowerPassBase;
  void runOnOperation() override {
    OwningOpRef<ModuleOp> candidate = cast<ModuleOp>(getOperation()->clone());
    if (failed(lowerModule(*candidate)) || failed(verify(*candidate)))
      return signalPassFailure();
    getOperation().getBodyRegion().takeBody(candidate->getBodyRegion());
  }
};
} // namespace
