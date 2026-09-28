//===----------------------------------------------------------------------===//
//
// This file extracts tessera_op, pure_tessera_op, tessera_optimize,
// tessera_guarantees, tessera_assumes and tessera_preserves global annotations
// and adds tessera_op / pure_tessera_op / tessera.property /
// tessera.preserves attributes and tessera.optimization ops to the module.
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Properties.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace enzyme {
namespace tessera {
#define GEN_PASS_DEF_LIFTTESSERAANNOTATIONSPASS
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h.inc"
} // namespace tessera
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;
using namespace mlir::enzyme::tessera;

namespace {

// What the plugin's tessera::guarantees, tessera::assumes and
// tessera::preserves attributes become. Argument positions count as tessera_op
// argument lists do: `this` first for a member function, and no sret.
//
//   tessera_guarantees=SPD:return    the return value is always SPD
//   tessera_guarantees=SPD:arg2      argument 2 is always SPD on exit
//   tessera_assumes=SPD:arg1         argument 1 is always SPD on entry
//   tessera_preserves=SPD:0,1        the output is SPD if arguments 0 and 1 are
//
// What is always true goes on the definition, as a `tessera.property` on the
// output or parameter it is true of (see Properties.h). A preserves is only
// true given its inputs, so it stays a rule on the function, with inputs
// renumbered to the function's own parameters, sret included:
//
//   tessera.preserves = [{property = "SPD", inputs = [0, 1]}]
constexpr llvm::StringLiteral kGuaranteesPrefix = "tessera_guarantees=";
constexpr llvm::StringLiteral kAssumesPrefix = "tessera_assumes=";
constexpr llvm::StringLiteral kPreservesPrefix = "tessera_preserves=";

bool isPropertyName(StringRef name) {
  return !name.empty() && llvm::all_of(name, [](char c) {
    return llvm::isAlnum(c) || c == '_';
  });
}

/// A result or a parameter of an llvm.func.
struct OutputSlot {
  bool isResult;
  unsigned index;
};

/// How a function's tessera_op says it treats an argument.
enum class Direction { Unmarked, In, Out, InOut };

/// The direction a tessera_op annotation gives each argument position:
/// "name(x:val=in, y:val=out):globals=1".
SmallVector<Direction> directionsOf(StringRef opAnnotation) {
  SmallVector<Direction> directions;
  size_t open = opAnnotation.find('(');
  if (open == StringRef::npos)
    return directions;
  StringRef list = opAnnotation.slice(open + 1, opAnnotation.find(')', open));
  if (list.trim().empty())
    return directions;
  SmallVector<StringRef> parts;
  list.split(parts, ',');
  for (StringRef part : parts) {
    StringRef marker = part.split(':').second.trim();
    directions.push_back(marker == "val=in"      ? Direction::In
                         : marker == "val=out"   ? Direction::Out
                         : marker == "val=inout" ? Direction::InOut
                                                 : Direction::Unmarked);
  }
  return directions;
}

/// What the lift needs to know about a function to place its declarations.
struct FunctionShape {
  // Op handles have no const accessors.
  mutable LLVM::LLVMFuncOp func;
  /// 1 if the function returns through an sret parameter, else 0. Plugin
  /// positions leave the sret out; the function's parameters do not.
  unsigned sretOffset;
  bool returnsValue;
  /// Per position (sret-exclusive), as its tessera_op marks it.
  SmallVector<Direction> directions;

  Direction directionOf(unsigned position) const {
    return position < directions.size() ? directions[position]
                                        : Direction::Unmarked;
  }

  FailureOr<OutputSlot> returnSlot() const {
    if (sretOffset)
      return OutputSlot{false, 0};
    if (returnsValue)
      return OutputSlot{true, 0};
    return failure();
  }

  FailureOr<OutputSlot> argSlot(unsigned position) const {
    unsigned param = position + sretOffset;
    if (param >= func.getNumArguments())
      return failure();
    return OutputSlot{false, param};
  }

  /// Whether a preserves declaration has an output to apply to: the return
  /// value, or else the one argument the function writes.
  bool hasPreservedOutput() const {
    if (succeeded(returnSlot()))
      return true;
    return llvm::count_if(directions, [](Direction d) {
             return d == Direction::Out || d == Direction::InOut;
           }) == 1;
  }
};

FunctionShape shapeOf(LLVM::LLVMFuncOp func, ArrayRef<StringRef> annotations) {
  FunctionShape shape{func, 0, false, {}};
  if (func.getNumArguments() &&
      func.getArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName()))
    shape.sretOffset = 1;
  shape.returnsValue =
      !isa<LLVM::LLVMVoidType>(func.getFunctionType().getReturnType());
  for (StringRef annot : annotations) {
    annot = annot.rtrim('\0');
    if (annot.consume_front("tessera_op=") ||
        annot.consume_front("pure_tessera_op="))
      shape.directions = directionsOf(annot);
  }
  return shape;
}

/// Add a property to the `tessera.property` list on a result or parameter.
void addProperty(LLVM::LLVMFuncOp func, OutputSlot slot, StringRef property) {
  ArrayAttr existing =
      slot.isResult
          ? func.getResultAttrOfType<ArrayAttr>(slot.index, kPropertyAttr)
          : func.getArgAttrOfType<ArrayAttr>(slot.index, kPropertyAttr);
  SmallVector<Attribute> list;
  if (existing)
    llvm::append_range(list, existing.getValue());
  list.push_back(StringAttr::get(func.getContext(), property));
  auto updated = ArrayAttr::get(func.getContext(), list);
  if (slot.isResult)
    func.setResultAttr(slot.index, kPropertyAttr, updated);
  else
    func.setArgAttr(slot.index, kPropertyAttr, updated);
}

/// Place one guarantees ("SPD:return", "SPD:arg2") or assumes ("SPD:arg1")
/// annotation, or explain why it cannot be.
LogicalResult placeFact(const FunctionShape &shape, StringRef text,
                        bool isGuarantee, std::string &why) {
  auto [property, target] = text.split(':');
  if (!isPropertyName(property)) {
    why = "'" + property.str() + "' is not a property name";
    return failure();
  }

  if (target == "return") {
    FailureOr<OutputSlot> slot = shape.returnSlot();
    if (!isGuarantee || failed(slot)) {
      why = isGuarantee ? "the function returns nothing"
                        : "a return value cannot be assumed";
      return failure();
    }
    addProperty(shape.func, *slot, property);
    return success();
  }

  unsigned position;
  if (!target.consume_front("arg") || target.getAsInteger(10, position)) {
    why = "'" + target.str() + "' is neither 'return' nor 'arg<N>'";
    return failure();
  }
  FailureOr<OutputSlot> slot = shape.argSlot(position);
  if (failed(slot)) {
    why = "argument " + std::to_string(position) + " does not exist";
    return failure();
  }

  // On a definition, a parameter's property is read as holding on entry if
  // the function reads the parameter and on exit if it only writes it. One
  // it does both to would be ambiguous, and one it is not known to write
  // cannot come out of the call with anything it did not have going in.
  Direction direction = shape.directionOf(position);
  if (direction == Direction::InOut) {
    why = "argument " + std::to_string(position) +
          " is val=inout, so a property on it could mean on entry or on exit";
    return failure();
  }
  if (isGuarantee && direction != Direction::Out) {
    why = "argument " + std::to_string(position) +
          " is not one the function writes; guaranteeing a property of an "
          "argument needs a tessera_op marking it val=out";
    return failure();
  }
  if (!isGuarantee && direction == Direction::Out) {
    why = "argument " + std::to_string(position) +
          " is val=out, so it has no value on entry to assume anything of";
    return failure();
  }
  addProperty(shape.func, *slot, property);
  return success();
}

/// Parse one preserves annotation ("SPD:0,1") into a rule for the function,
/// or explain why it cannot be.
FailureOr<DictionaryAttr> parsePreserve(const FunctionShape &shape,
                                        StringRef text, std::string &why) {
  auto [property, list] = text.split(':');
  if (!isPropertyName(property) || list.empty()) {
    why = "expected '<property>:<argument>,...'";
    return failure();
  }
  if (!shape.hasPreservedOutput()) {
    why = "the function neither returns a value nor writes exactly one "
          "argument its tessera_op marks val=out or val=inout, so there is no "
          "output for the property to be preserved in";
    return failure();
  }
  Builder b(shape.func.getContext());
  SmallVector<Attribute> inputs;
  SmallVector<StringRef> parts;
  list.split(parts, ',');
  for (StringRef part : parts) {
    unsigned position;
    if (part.trim().getAsInteger(10, position) ||
        failed(shape.argSlot(position))) {
      why = "'" + part.trim().str() + "' is not an argument position";
      return failure();
    }
    inputs.push_back(b.getI64IntegerAttr(position + shape.sretOffset));
  }
  return b.getDictionaryAttr(
      {b.getNamedAttr("property", b.getStringAttr(property)),
       b.getNamedAttr("inputs", b.getArrayAttr(inputs))});
}

struct LiftTesseraAnnotationsPass
    : public enzyme::tessera::impl::LiftTesseraAnnotationsPassBase<
          LiftTesseraAnnotationsPass> {
  using LiftTesseraAnnotationsPassBase::LiftTesseraAnnotationsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // Find string constants in metadata, locate annotations array, and build
    // optimization ops
    LLVM::GlobalOp annotationGlobal = nullptr;
    DenseMap<StringRef, std::string> stringGlobals;
    SmallVector<std::string> optimizationRules;

    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSymName() == "llvm.global.annotations") {
        annotationGlobal = global;
      }
      if (global.getSection() && *global.getSection() == "llvm.metadata") {
        if (auto strAttr =
                dyn_cast_or_null<StringAttr>(global.getValueAttr())) {
          StringRef str = strAttr.getValue();
          stringGlobals[global.getSymName()] = str.str();
          if (str.starts_with("tessera_optimize=")) {
            StringRef rule =
                str.drop_front(StringRef("tessera_optimize=").size());
            if (rule.ends_with('\0'))
              rule = rule.drop_back(1);
            optimizationRules.push_back(rule.str());
          }
        }
      }
    }

    if (!annotationGlobal)
      return;

    Region &region = annotationGlobal.getInitializerRegion();
    if (region.empty())
      return;

    if (!optimizationRules.empty()) {
      OpBuilder builder(ctx);
      Location loc = builder.getUnknownLoc();
      builder.setInsertionPointToEnd(module.getBody());
      auto optimizationsOp = tessera::OptimizationsOp::create(builder, loc);
      Region &body = optimizationsOp.getBody();
      Block *block = builder.createBlock(&body);
      builder.setInsertionPointToStart(block);

      for (const std::string &rule : optimizationRules) {
        tessera::OptimizationOp::create(builder, loc,
                                        builder.getStringAttr(rule));
      }
    }

    DenseMap<Value, StringRef> valueToFunction;
    DenseMap<Value, StringRef> valueToAnnotation;

    // Find addressof operations
    for (Operation &op : region.front()) {
      if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
        StringRef globalName = addrOf.getGlobalName();
        Value result = addrOf.getResult();

        if (module.lookupSymbol<LLVM::LLVMFuncOp>(globalName)) {
          valueToFunction[result] = globalName;
        } else if (stringGlobals.count(globalName)) {
          valueToAnnotation[result] = stringGlobals[globalName];
        }
      }
    }

    DenseMap<Value, StringRef> structToFunction;
    DenseMap<Value, StringRef> structToAnnotation;

    // Follow insertvalue chains to match functions with annotations
    for (Operation &op : region.front()) {
      if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(&op)) {
        Value inserted = insertValue.getValue();
        Value container = insertValue.getContainer();
        Value result = insertValue.getResult();
        auto position = insertValue.getPosition();

        if (position.size() == 1) {
          if (position[0] == 0 && valueToFunction.count(inserted)) {
            structToFunction[result] = valueToFunction[inserted];
          } else if (position[0] == 1 && valueToAnnotation.count(inserted)) {
            structToAnnotation[result] = valueToAnnotation[inserted];
          }

          // Propagate information through the chain
          if (structToFunction.count(container)) {
            structToFunction[result] = structToFunction[container];
          }
          if (structToAnnotation.count(container)) {
            structToAnnotation[result] = structToAnnotation[container];
          }
        }
      }
    }

    // A function can carry more than one annotation -- a tessera op that is
    // also marked tessera_no_rewrite, say -- so keep all of them.
    DenseMap<StringRef, SmallVector<StringRef>> functionToAnnotations;

    for (auto [structValue, funcName] : structToFunction) {
      if (structToAnnotation.count(structValue)) {
        StringRef annotStr = structToAnnotation[structValue];
        functionToAnnotations[funcName].push_back(annotStr);
      }
    }

    // Apply annotations as attributes to functions
    for (auto &[funcName, annotStrs] : functionToAnnotations) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
      if (!func)
        continue;

      // The table is walked in no particular order, and clang can list an
      // annotation once per redeclaration, so settle both before the
      // annotations become ordered lists.
      llvm::sort(annotStrs);
      annotStrs.erase(llvm::unique(annotStrs), annotStrs.end());

      FunctionShape shape = shapeOf(func, annotStrs);
      SmallVector<Attribute> preserves;
      for (StringRef annot : annotStrs) {
        annot = annot.rtrim('\0');
        StringRef kind;
        std::string why;
        LogicalResult placed = success();
        if (annot.consume_front(kGuaranteesPrefix)) {
          kind = "guarantees";
          placed = placeFact(shape, annot, /*isGuarantee=*/true, why);
        } else if (annot.consume_front(kAssumesPrefix)) {
          kind = "assumes";
          placed = placeFact(shape, annot, /*isGuarantee=*/false, why);
        } else if (annot.consume_front(kPreservesPrefix)) {
          kind = "preserves";
          FailureOr<DictionaryAttr> rule = parsePreserve(shape, annot, why);
          if (succeeded(rule))
            preserves.push_back(*rule);
          placed = failure(failed(rule));
        } else {
          continue;
        }
        if (succeeded(placed))
          continue;
        // An annotation that cannot be placed only costs the facts it would
        // have given, so it is left out rather than failing the compile.
        func.emitWarning() << "ignoring tessera " << kind << " annotation '"
                           << annot << "': " << why;
      }
      if (!preserves.empty())
        func->setAttr(kPreservesAttr, ArrayAttr::get(ctx, preserves));

      for (StringRef annot : annotStrs) {
        // Parse "tessera_op=string\0" and "pure_tessera_op=string\0"
        if (annot.consume_front("tessera_op=")) {
          annot = annot.rtrim('\0');
          func->setAttr("tessera_op",
                        StringAttr::get(func->getContext(), annot));
        } else if (annot.consume_front("pure_tessera_op=")) {
          annot = annot.rtrim('\0');
          func->setAttr("pure_tessera_op",
                        StringAttr::get(func->getContext(), annot));
        } else if (annot.rtrim('\0') == "tessera_no_rewrite") {
          // No rule may rewrite anything in this function's body. This is
          // how a replacement's fallback path reaches the op it replaces
          // without being rewritten back into a call to itself.
          //
          // It only protects a body that is still there: the pipeline runs
          // `inline` before this pass, so the function must also be
          // noinline, or its body is copied into callers where nothing marks
          // it.
          func->setAttr("tessera.no_rewrite",
                        UnitAttr::get(func->getContext()));
        }
      }
    }
  }
};
} // namespace
