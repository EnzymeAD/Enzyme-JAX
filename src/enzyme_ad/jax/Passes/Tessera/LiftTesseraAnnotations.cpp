//===----------------------------------------------------------------------===//
//
// Reads the tessera annotations the frontend leaves in the module and turns
// them into the attributes and ops the other tessera passes use. The frontend
// passes annotations on as the user wrote them; all parsing and checking
// happens here.
//
// Annotations on functions, in llvm.global.annotations:
//
//   tessera_op=<marking>             the function is a tessera op, e.g.
//   pure_tessera_op=<marking>        "eigen.inv(x:val=in, y)"
//   tessera_guarantees=<fact>,...    a fact is "SPD(M)", or a bare "SPD" for
//   tessera_assumes=<fact>,...       the return value (guarantees only)
//   tessera_preserves=<property>,<param>,...
//   tessera_readonly=<param>,...
//   tessera_no_rewrite
//   tessera_fact                     a generated function for a fact stated on
//                                    a statement
//   tessera_params=<name>,...        each argument's name (added by the
//                                    frontend)
//   tessera_arg_types=<global>,...   for each argument, a global whose type is
//                                    the type behind the pointer (added by the
//                                    frontend)
//
// A parameter is named as tessera_params names it, or by position. Positions
// count `this` for a member function, and leave out the sret.
//
// Annotations on globals, as strings in llvm.metadata:
//
//   tessera_optimize=<rule>          a rewrite rule
//   tessera_name=<name>=<value>      a C name a rule may use, e.g. "MAT_SPD=5"
//
//===----------------------------------------------------------------------===//

#include "Passes/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "src/enzyme_ad/jax/Dialect/Tessera/Dialect.h"
#include "src/enzyme_ad/jax/Passes/Tessera/Passes.h"
#include "src/enzyme_ad/jax/Passes/Tessera/PropertyPropagation.h"
#include "src/enzyme_ad/jax/Passes/Tessera/RuleAST.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
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

// Annotation prefixes. PropertyPropagation.h describes the attributes the
// property annotations become.
constexpr llvm::StringLiteral kGuaranteesPrefix = "tessera_guarantees=";
constexpr llvm::StringLiteral kAssumesPrefix = "tessera_assumes=";
constexpr llvm::StringLiteral kPreservesPrefix = "tessera_preserves=";
constexpr llvm::StringLiteral kReadonlyPrefix = "tessera_readonly=";
constexpr llvm::StringLiteral kParamsPrefix = "tessera_params=";
constexpr llvm::StringLiteral kArgTypesPrefix = "tessera_arg_types=";
constexpr llvm::StringLiteral kOptimizePrefix = "tessera_optimize=";
constexpr llvm::StringLiteral kNamePrefix = "tessera_name=";

bool isPropertyName(StringRef name) {
  return !name.empty() && llvm::all_of(name, [](char c) {
    return llvm::isAlnum(c) || c == '_';
  });
}

// Split a list the frontend joined with ',', as "SPD(A),symmetric(B)", at
// the commas outside parentheses.
SmallVector<StringRef> splitList(StringRef list) {
  SmallVector<StringRef> items;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0, e = list.size(); i != e; ++i) {
    if (list[i] == '(')
      ++depth;
    else if (list[i] == ')')
      --depth;
    else if (list[i] == ',' && depth == 0) {
      items.push_back(list.slice(start, i).trim());
      start = i + 1;
    }
  }
  items.push_back(list.drop_front(start).trim());
  return items;
}

// Split a fact written "SPD(M)" into the property and the parameter it is
// of. A bare "SPD" has no parameter. Fails if the text is neither form.
bool parseFact(StringRef text, StringRef &property, StringRef &param) {
  size_t open = text.find('(');
  if (open == StringRef::npos) {
    property = text;
    param = "";
    return isPropertyName(property);
  }
  if (!text.ends_with(")"))
    return false;
  property = text.take_front(open).trim();
  param = text.drop_front(open + 1).drop_back().trim();
  return isPropertyName(property) && !param.empty() &&
         !param.contains('(') && !param.contains(')') && !param.contains(',');
}

// A result or a parameter of an llvm.func.
struct OutputSlot {
  bool isResult;
  unsigned index;
};

// How a function's tessera_op says it treats an argument.
enum class Direction { Unmarked, In, Out, InOut };

// The direction a tessera_op marking gives each argument position:
// "name(x:val=in, y:val=out)".
SmallVector<Direction> directionsOf(StringRef marking) {
  SmallVector<Direction> directions;
  size_t open = marking.find('(');
  if (open == StringRef::npos)
    return directions;
  StringRef list = marking.slice(open + 1, marking.find(')', open));
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

// What the lift needs to know about a function to place its declarations.
struct FunctionShape {
  // Op handles have no const accessors.
  mutable LLVM::LLVMFuncOp func;
  // 1 if the function returns through an sret parameter, else 0. Positions
  // don't count the sret; the function's parameters do.
  unsigned sretOffset;
  bool returnsValue;
  // Per position, as its tessera_op marks it.
  SmallVector<Direction> directions;
  // Per position, as tessera_params names it; empty where it does not.
  SmallVector<StringRef> names;
  // Whether the function is a marker for facts stated on a statement.
  bool isMarker = false;

  unsigned numPositions() const { return func.getNumArguments() - sretOffset; }

  FailureOr<OutputSlot> returnSlot() const {
    if (sretOffset)
      return OutputSlot{false, 0};
    if (returnsValue)
      return OutputSlot{true, 0};
    return failure();
  }

  // The position `name` refers to: one tessera_params names so, or a number.
  std::optional<unsigned> positionOf(StringRef name) const {
    unsigned position;
    if (!name.getAsInteger(10, position))
      return position < numPositions() ? std::optional<unsigned>(position)
                                       : std::nullopt;
    for (auto [i, known] : llvm::enumerate(names))
      if (!known.empty() && known == name)
        return i;
    return std::nullopt;
  }

  // How a diagnostic refers to the argument at `position`.
  std::string describe(unsigned position) const {
    if (position < names.size() && !names[position].empty())
      return ("'" + names[position] + "'").str();
    return "argument " + std::to_string(position);
  }

  // Why `name` refers to no argument.
  std::string unknown(StringRef name) const {
    return isMarker ? "'" + name.str() +
                          "' is not a pointer or object variable in scope there"
                    : "'" + name.str() + "' is not a parameter of the function";
  }

  bool isPointer(OutputSlot slot) const {
    return isa<LLVM::LLVMPointerType>(func.getArgumentTypes()[slot.index]);
  }

  // Whether the parameter is a struct passed by value: a pointer to a copy
  // the caller never sees again.
  bool isByValue(OutputSlot slot) const {
    return func.getArgAttr(slot.index, LLVM::LLVMDialect::getByValAttrName()) !=
           nullptr;
  }
};

FunctionShape shapeOf(LLVM::LLVMFuncOp func, ArrayRef<StringRef> annotations) {
  FunctionShape shape{func, 0, false, {}, {}};
  if (func.getNumArguments() &&
      func.getArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName()))
    shape.sretOffset = 1;
  shape.returnsValue =
      !isa<LLVM::LLVMVoidType>(func.getFunctionType().getReturnType());
  for (StringRef annot : annotations) {
    if (annot.consume_front("tessera_op=") ||
        annot.consume_front("pure_tessera_op="))
      shape.directions = directionsOf(annot);
    else if (annot.consume_front(kParamsPrefix))
      annot.split(shape.names, ',');
    else if (annot == "tessera_fact")
      shape.isMarker = true;
  }
  return shape;
}

// Add a property to the list named `attrName` (`tessera.guarantees`,
// `tessera.assumes` or `tessera.establishes`) on a result or parameter.
void addProperty(LLVM::LLVMFuncOp func, OutputSlot slot, StringRef property,
                 StringRef attrName) {
  ArrayAttr existing =
      slot.isResult ? func.getResultAttrOfType<ArrayAttr>(slot.index, attrName)
                    : func.getArgAttrOfType<ArrayAttr>(slot.index, attrName);
  SmallVector<Attribute> list;
  if (existing)
    llvm::append_range(list, existing.getValue());
  list.push_back(StringAttr::get(func.getContext(), property));
  auto updated = ArrayAttr::get(func.getContext(), list);
  if (slot.isResult)
    func.setResultAttr(slot.index, attrName, updated);
  else
    func.setArgAttr(slot.index, attrName, updated);
}

// The parameter `name` refers to, or why there is none.
FailureOr<std::pair<unsigned, OutputSlot>>
paramNamed(const FunctionShape &shape, StringRef name, std::string &why) {
  std::optional<unsigned> position = shape.positionOf(name);
  if (!position) {
    why = shape.unknown(name);
    return failure();
  }
  if (*position >= shape.numPositions()) {
    why = shape.describe(*position) +
          " is not an argument of the function as it was compiled";
    return failure();
  }
  return std::make_pair(*position,
                        OutputSlot{false, *position + shape.sretOffset});
}

// Place one guaranteed ("SPD(M)", "SPD") or assumed ("SPD(M)") fact, or
// explain why it cannot be.
LogicalResult placeFact(const FunctionShape &shape, StringRef text,
                        bool isGuarantee, std::string &why) {
  StringRef property, param;
  if (!parseFact(text, property, param)) {
    why = shape.isMarker
              ? "write a property name and the variable it applies to, such "
                "as \"SPD(A)\""
              : std::string("write a property name and the parameter it "
                            "applies to, such as \"SPD(M)\"") +
                    (isGuarantee ? ", or a bare \"SPD\" for the return value"
                                 : "");
    return failure();
  }

  if (isGuarantee && param == "return")
    param = "";
  if (param.empty()) {
    if (shape.isMarker) {
      why = "a fact stated on a statement names a variable, as \"" +
            property.str() + "(A)\" does";
      return failure();
    }
    if (!isGuarantee) {
      why = "an assumption needs the parameter it applies to; write \"" +
            property.str() + "(M)\"";
      return failure();
    }
    // An older syntax gave the parameter as a separate item. Catch it rather
    // than reading it as a property of the return value.
    if (shape.positionOf(property)) {
      why = "'" + property.str() +
            "' names a parameter, not a property; state a property of it as "
            "\"SPD(" +
            property.str() + ")\"";
      return failure();
    }
    FailureOr<OutputSlot> slot = shape.returnSlot();
    if (failed(slot)) {
      why = "the function returns nothing; name the parameter the fact "
            "applies to, as \"" +
            property.str() + "(M)\"";
      return failure();
    }
    addProperty(shape.func, *slot, property, kGuaranteesAttr);
    return success();
  }

  auto named = paramNamed(shape, param, why);
  if (failed(named))
    return failure();
  auto [position, slot] = *named;
  std::string arg = shape.describe(position);

  // A pointer the tessera_op doesn't lift is a handle: the function may change
  // the object behind it in place. A guarantee on it is about that object.
  Direction direction = position < shape.directions.size()
                            ? shape.directions[position]
                            : Direction::Unmarked;
  if (isGuarantee && direction == Direction::Unmarked &&
      shape.isPointer(slot)) {
    if (shape.isByValue(slot)) {
      why = arg + " is passed by value, so the function cannot give it a "
                  "property";
      return failure();
    }
    addProperty(shape.func, slot, property, kEstablishesAttr);
    return success();
  }

  // Otherwise it's about the value behind the argument. A guarantee needs a
  // value the function writes; an assumption needs one it doesn't.
  if (isGuarantee) {
    if (direction != Direction::Out && direction != Direction::InOut) {
      why = arg + (direction == Direction::In
                       ? " is val=in, so the function only reads the value "
                         "behind it; guaranteeing a property of an argument "
                         "needs a tessera_op marking it val=out or val=inout, "
                         "or a pointer it does not lift"
                       : " is not a pointer, so the function cannot give it a "
                         "property");
      return failure();
    }
    addProperty(shape.func, slot, property, kGuaranteesAttr);
    return success();
  }
  if (direction == Direction::Out || direction == Direction::InOut) {
    why = arg + (direction == Direction::Out
                     ? " is val=out, so it has no value on entry to assume "
                       "anything of"
                     : " is val=inout, so the function changes the value an "
                       "assumption would be of");
    return failure();
  }
  addProperty(shape.func, slot, property, kAssumesAttr);
  return success();
}

// Place one readonly parameter ("A"), or explain why it cannot be.
LogicalResult placeReadonly(const FunctionShape &shape, StringRef name,
                            std::string &why) {
  auto named = paramNamed(shape, name, why);
  if (failed(named))
    return failure();
  auto [position, slot] = *named;
  if (!shape.isPointer(slot) || shape.isByValue(slot)) {
    why = shape.describe(position) +
          (shape.isPointer(slot) ? " is passed by value" : " is not a pointer") +
          ", so there is no object to leave alone";
    return failure();
  }
  shape.func.setArgAttr(slot.index, kReadonlyAttr,
                        UnitAttr::get(shape.func.getContext()));
  return success();
}

// Turn one preserves ("SPD", "A", "B") into a rule, or explain why it can't
// be. All or nothing: dropping an input it couldn't name would claim more
// than was declared.
FailureOr<DictionaryAttr> parsePreserve(const FunctionShape &shape,
                                        ArrayRef<StringRef> items,
                                        std::string &why) {
  if (items.size() < 2) {
    why = "it takes a property name and the parameters it is preserved from";
    return failure();
  }
  StringRef property = items.front();
  if (!isPropertyName(property)) {
    why = "'" + property.str() +
          "' is not a property name; use letters, digits and underscores";
    return failure();
  }
  // A preserves applies to the return value, or else to the one argument the
  // function writes. With neither, the one object named keeps the property
  // through the call.
  bool hasOutput = succeeded(shape.returnSlot()) ||
                   llvm::count_if(shape.directions, [](Direction d) {
                     return d == Direction::Out || d == Direction::InOut;
                   }) == 1;
  if (!hasOutput && items.size() != 2) {
    why = "the function neither returns a value nor writes exactly one "
          "argument its tessera_op marks val=out or val=inout, so there is no "
          "output for the property to be preserved in";
    return failure();
  }
  Builder b(shape.func.getContext());
  SmallVector<Attribute> inputs;
  for (StringRef name : items.drop_front()) {
    auto named = paramNamed(shape, name, why);
    if (failed(named))
      return failure();
    inputs.push_back(b.getI64IntegerAttr(named->second.index));
  }
  return b.getDictionaryAttr(
      {b.getNamedAttr("property", b.getStringAttr(property)),
       b.getNamedAttr("inputs", b.getArrayAttr(inputs))});
}

// Erase the globals that only carried an argument's type. Only llvm.used or
// llvm.compiler.used keep them alive, so remove them from those first. A
// global that anything else uses is kept.
void eraseTypeCarriers(ModuleOp module, ArrayRef<LLVM::GlobalOp> carriers) {
  if (carriers.empty())
    return;
  llvm::DenseSet<StringRef> names;
  for (LLVM::GlobalOp carrier : carriers)
    names.insert(carrier.getSymName());

  for (StringRef listName : {"llvm.used", "llvm.compiler.used"}) {
    auto list = module.lookupSymbol<LLVM::GlobalOp>(listName);
    if (!list || list.getInitializerRegion().empty())
      continue;
    Block &body = list.getInitializerRegion().front();
    // Only a list built the way the importer builds it is rewritten.
    SmallVector<StringRef> kept;
    bool dropped = false, understood = true;
    for (Operation &op : body) {
      if (auto addressOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
        if (names.contains(addressOf.getGlobalName()))
          dropped = true;
        else
          kept.push_back(addressOf.getGlobalName());
      } else if (!isa<LLVM::UndefOp, LLVM::PoisonOp, LLVM::InsertValueOp,
                      LLVM::ReturnOp>(op)) {
        understood = false;
      }
    }
    if (!dropped || !understood)
      continue;
    if (kept.empty()) {
      list.erase();
      continue;
    }

    Location loc = list.getLoc();
    auto ptrType = LLVM::LLVMPointerType::get(module.getContext());
    auto arrayType = LLVM::LLVMArrayType::get(ptrType, kept.size());
    for (Operation &op : llvm::make_early_inc_range(llvm::reverse(body)))
      op.erase();
    list.setGlobalType(arrayType);
    auto builder = OpBuilder::atBlockEnd(&body);
    Value array = LLVM::UndefOp::create(builder, loc, arrayType);
    for (auto [i, symbol] : llvm::enumerate(kept)) {
      Value address = LLVM::AddressOfOp::create(builder, loc, ptrType, symbol);
      array = LLVM::InsertValueOp::create(builder, loc, array, address,
                                          ArrayRef<int64_t>{(int64_t)i});
    }
    LLVM::ReturnOp::create(builder, loc, array);
  }

  for (LLVM::GlobalOp carrier : carriers)
    if (SymbolTable::symbolKnownUseEmpty(carrier, module))
      carrier.erase();
}

// An annotation on a function, and where in the source it was written, as
// "file.cpp:12: ", or nothing if that is not known.
struct Annotation {
  StringRef text;
  std::string where;
};

void warnIgnored(const FunctionShape &shape, const Annotation &annot,
                 StringRef kind, StringRef text, StringRef why) {
  // An annotation that cannot be used only costs what it would have given,
  // so it is left out rather than failing the compile.
  shape.func.emitWarning() << annot.where << "ignoring tessera " << kind
                           << " '" << text << "'"
                           << (shape.isMarker ? " stated on a statement" : "")
                           << ": " << why;
}

struct LiftTesseraAnnotationsPass
    : public enzyme::tessera::impl::LiftTesseraAnnotationsPassBase<
          LiftTesseraAnnotationsPass> {
  using LiftTesseraAnnotationsPassBase::LiftTesseraAnnotationsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    // String globals by name, the rules, and the values of C names rules use.
    LLVM::GlobalOp annotationGlobal = nullptr;
    DenseMap<StringRef, StringRef> stringGlobals;
    SmallVector<StringRef> optimizationRules;
    llvm::StringMap<std::string> ruleNames;

    for (auto global : module.getOps<LLVM::GlobalOp>()) {
      if (global.getSymName() == "llvm.global.annotations")
        annotationGlobal = global;
      if (!global.getSection() || *global.getSection() != "llvm.metadata")
        continue;
      auto strAttr = dyn_cast_or_null<StringAttr>(global.getValueAttr());
      if (!strAttr)
        continue;
      StringRef str = strAttr.getValue().rtrim('\0');
      stringGlobals[global.getSymName()] = str;
      if (str.consume_front(kOptimizePrefix)) {
        optimizationRules.push_back(str);
      } else if (str.consume_front(kNamePrefix)) {
        auto [name, value] = str.split('=');
        ruleNames[name] = value.str();
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

      for (StringRef rule : optimizationRules) {
        tessera::OptimizationOp::create(
            builder, loc,
            builder.getStringAttr(substituteRuleNames(rule, ruleNames, loc)));
      }
    }

    // Each table entry is {function, annotation, file, line, args}, built up
    // one field at a time by insertvalue ops.
    struct Entry {
      StringRef symbol, annotation, file;
      int64_t line = 0;
    };
    DenseMap<Value, StringRef> symbols;
    DenseMap<Value, int64_t> constants;
    DenseMap<Value, Entry> partial;
    SmallVector<Entry> entries;
    for (Operation &op : region.front()) {
      if (auto addrOf = dyn_cast<LLVM::AddressOfOp>(&op)) {
        symbols[addrOf.getResult()] = addrOf.getGlobalName();
      } else if (auto constant = dyn_cast<LLVM::ConstantOp>(&op)) {
        if (auto value = dyn_cast<IntegerAttr>(constant.getValue()))
          constants[constant.getResult()] = value.getInt();
      } else if (auto insertValue = dyn_cast<LLVM::InsertValueOp>(&op)) {
        auto position = insertValue.getPosition();
        Value inserted = insertValue.getValue();
        if (position.size() != 1)
          continue;
        // An entry is complete once it is put in the table.
        if (isa<LLVM::LLVMArrayType>(insertValue.getType())) {
          auto it = partial.find(inserted);
          if (it != partial.end())
            entries.push_back(it->second);
          continue;
        }
        Entry entry = partial.lookup(insertValue.getContainer());
        switch (position[0]) {
        case 0:
          entry.symbol = symbols.lookup(inserted);
          break;
        case 1:
          entry.annotation = stringGlobals.lookup(symbols.lookup(inserted));
          break;
        case 2:
          entry.file = stringGlobals.lookup(symbols.lookup(inserted));
          break;
        case 3:
          entry.line = constants.lookup(inserted);
          break;
        }
        partial[insertValue.getResult()] = entry;
      }
    }

    // A function can carry more than one annotation -- a tessera op that is
    // also marked tessera_no_rewrite, say -- so keep all of them.
    llvm::MapVector<StringRef, SmallVector<Annotation>> functionToAnnotations;
    for (const Entry &entry : entries) {
      if (entry.annotation.empty() ||
          !module.lookupSymbol<LLVM::LLVMFuncOp>(entry.symbol))
        continue;
      std::string where;
      if (!entry.file.empty() && entry.line > 0)
        where = (entry.file + ":" + Twine(entry.line) + ": ").str();
      functionToAnnotations[entry.symbol].push_back(
          {entry.annotation, std::move(where)});
    }

    SmallVector<LLVM::GlobalOp> typeCarriers;
    for (auto &[funcName, annots] : functionToAnnotations) {
      auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(funcName);

      // The table is in no particular order, and clang may repeat an
      // annotation once per redeclaration, so sort and deduplicate.
      llvm::stable_sort(annots, [](const Annotation &a, const Annotation &b) {
        return a.text < b.text;
      });
      annots.erase(llvm::unique(annots,
                                [](const Annotation &a, const Annotation &b) {
                                  return a.text == b.text;
                                }),
                   annots.end());

      SmallVector<StringRef> texts;
      for (const Annotation &annot : annots)
        texts.push_back(annot.text);
      FunctionShape shape = shapeOf(func, texts);

      SmallVector<Attribute> preserves;
      for (const Annotation &annot : annots) {
        StringRef text = annot.text;
        std::string why;
        bool isGuarantee = text.consume_front(kGuaranteesPrefix);
        if (isGuarantee || text.consume_front(kAssumesPrefix)) {
          StringRef kind = isGuarantee ? "guarantees" : "assumes";
          // Each fact stands alone: a bad one is skipped, the rest kept.
          for (StringRef fact : splitList(text))
            if (failed(placeFact(shape, fact, isGuarantee, why)))
              warnIgnored(shape, annot, kind, fact, why);
        } else if (text.consume_front(kReadonlyPrefix)) {
          // Likewise for readonly. Skipping one only means calls are assumed
          // to change that object.
          for (StringRef name : splitList(text))
            if (failed(placeReadonly(shape, name, why)))
              warnIgnored(shape, annot, "readonly", name, why);
        } else if (text.consume_front(kPreservesPrefix)) {
          FailureOr<DictionaryAttr> rule =
              parsePreserve(shape, splitList(text), why);
          if (succeeded(rule))
            preserves.push_back(*rule);
          else
            warnIgnored(shape, annot, "preserves", text, why);
        } else if (text.consume_front(kArgTypesPrefix)) {
          SmallVector<StringRef> carrierNames;
          text.split(carrierNames, ',');
          SmallVector<Attribute> types;
          for (StringRef name : carrierNames) {
            auto carrier = name.empty()
                               ? LLVM::GlobalOp()
                               : module.lookupSymbol<LLVM::GlobalOp>(name);
            if (carrier) {
              types.push_back(TypeAttr::get(carrier.getGlobalType()));
              typeCarriers.push_back(carrier);
            } else {
              types.push_back(UnitAttr::get(ctx));
            }
          }
          func->setAttr(kArgTypesAttr, ArrayAttr::get(ctx, types));
        } else if (text.consume_front("tessera_op=")) {
          func->setAttr("tessera_op", StringAttr::get(ctx, text));
        } else if (text.consume_front("pure_tessera_op=")) {
          func->setAttr("pure_tessera_op", StringAttr::get(ctx, text));
        } else if (text == "tessera_no_rewrite") {
          // No rule may rewrite this function's body, so a replacement's
          // fallback can call the original op without being rewritten into
          // itself. The function must also be noinline: the pipeline inlines
          // before this pass, and an inlined copy isn't protected.
          func->setAttr("tessera.no_rewrite", UnitAttr::get(ctx));
        } else if (text == "tessera_fact") {
          // A generated, empty function for a fact stated on a statement.
          // tessera-to-llvm deletes it once the facts are used.
          func->setAttr(kFactMarkerAttr, UnitAttr::get(ctx));
        }
      }
      if (!preserves.empty())
        func->setAttr(kPreservesAttr, ArrayAttr::get(ctx, preserves));
    }

    // Their types are on the functions now, which is all they were for.
    eraseTypeCarriers(module, llvm::SetVector<LLVM::GlobalOp>(
                                  typeCarriers.begin(), typeCarriers.end())
                                  .getArrayRef());
  }
};
} // namespace
