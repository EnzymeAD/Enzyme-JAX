#include "mlir/Support/LogicalResult.h"
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::comm;

// Split
LogicalResult CommSplit::verify() {
  for (Operation &op : getDeclarations().getOps()) {
    // Check that all ops are allowable as members
    if (!op.hasTrait<SplitMemberOp>()) {
      return op.emitOpError("not allowed as immediate split op member");
    }
  }

  // check that all branches have disjoint device sets
  // check that loop branches only present if split is loopy
  DenseSet<unsigned> used_devices;
  for (CommBranch branch : getBranches()) {
    for (unsigned device : branch.getDeviceIds()) {
      if (used_devices.contains(device)) {
        return branch.emitError(
            "uses device already accounted for in same split");
      }
      used_devices.insert(device);
    }

    if(!getIsLoopy() && branch.isLoop()){
      return branch.emitOpError("loop branches only allowed in loopy splits");
    }
  }
  
  return success();
}

// CommBranch
/**
 * A branch is a loop if any of its looping regions (loop, reentry, exit) are non-empty.
 * If any of these regions are non-empty, all of them must be non-empty.
 */
bool CommBranch::isLoop(){
  return !getLoop().empty();
}

/**
 * Verify that 
 *   - the exit block is used only if the branch is loopy (otherwise code should be placed in the entry block)
 *   - If the branch is loopy, the loop block should end with comm.condition
 *   - Region arguments should match the comm.yield types
 */
LogicalResult CommBranch::verify() {

  // Either all or none of the loop, reentry, exit regions should be empty
  if(getLoop().empty() != getReentry().empty() || getLoop().empty() != getExit().empty()){
    return emitOpError("all or none of loop, reentry, exit regions should be non-empty");
  }
  // Entry region block should exist (enforced by sized_region), 
  // have no region arguments, and should end with sfc.yield (will check types later)
  auto &entry = getEntry().front();
  if (dyn_cast<CommYield>(entry.back()) == nullptr) {
    return emitOpError("entry block should end with comm.yield");
  }
  if (entry.getNumArguments() > 0) {
    return emitOpError("entry block should have no arguments");
  }

  if (isLoop()) {
    // Check comm.condition terminator
    auto terminator = getLoop().front().getTerminator();
    if(auto cond_op = dyn_cast<CommCondition>(terminator)){
      // Check that condition results match the region arguments for reentry, exit
      auto forwarding_types = cond_op.getResults().getTypes();
      auto reentry_arg_types = getReentry().front().getArgumentTypes();
      auto exit_arg_types = getExit().front().getArgumentTypes();
      
      if(forwarding_types != reentry_arg_types){
        return emitOpError("condition results should match reentry region arguments");
      }
      if(forwarding_types != exit_arg_types){
        return emitOpError("condition results should match exit region arguments");
      }
    } else {
      return emitOpError("loop block should end with comm.condition");
    }

    // Check entry, reentry block has comm.yield terminator matching loop region arguments
    auto entry_term = cast<CommYield>(getEntry().front().getTerminator());
    auto reentry_term = dyn_cast<CommYield>(getReentry().front().getTerminator());
    if(!reentry_term){
      return reentry_term.emitOpError("reentry block should end with comm.yield");
    }
    for (CommYield yield_to_check : {entry_term, reentry_term}) {
      if(yield_to_check.getResults().getTypes() != getLoop().getArgumentTypes()){
        return yield_to_check.emitOpError("comm.yield types should match arguments of loop region");
      }
    }

    // Check existence of comm.yield with no arguments in exit region
    if (auto exit_term = dyn_cast<CommYield>(getExit().front().back())){
      if(exit_term.getNumOperands() > 0){
        return exit_term.emitOpError("exit block should end with comm.yield with no arguments");
      }
    } else {
      return emitOpError("exit block should end with comm.yield with no arguments");
    }
  } 
  return success();
}

// CommSend
LogicalResult CommSend::verify(){
  auto op = getToken().getDefiningOp();
  if(!isa<CommSimpleMessage>(op)) return emitOpError("can only send to tokens from simple messages");
  return success();
}

CommSimpleMessage CommSend::getMessage(){
  return dyn_cast<CommSimpleMessage>(getToken().getDefiningOp());
}

// CommSimpleMessage
mlir::Type CommSimpleMessage::getInputType() {
  return getDataType();
}

mlir::Type CommSimpleMessage::getOutputType() {
  return getDataType();
}

// CommMultiplexMessage
LogicalResult CommMultiplexMessage::verify() {
  for (mlir::Value input_token : getInTokens()) { 
    auto input_op = input_token.getDefiningOp();
    if(CommMessage input_msg = dyn_cast<CommMessage>(input_op)){
      // check that the data types of the input message and this message match
      if(input_msg.getOutputType() != getDataType()){
        return emitOpError("includes message with return type different than declared");
      }
    } else {
      // TODO write verification to ensure all tokens are defined by messages only
      return input_op->emitOpError("message tokens should only be defined by message declarations");
    }
  }
  return success();
}

mlir::Type CommMultiplexMessage::getInputType() {
  // cannot send to a multiplex message!
  return NoneType();
}

mlir::Type CommMultiplexMessage::getOutputType() {
  return getDataType();
}

// Custom parsers

/**
 * Want to see either <<empty string>> or:
 *  `loop` $loop `reenter` $reentry `exit` $exit
 */
static ParseResult parseBranchLoopRegions(OpAsmParser &p, Region &loop, Region &reentry, Region &exit){
  auto res = p.parseOptionalKeyword("loop");
  if(res.succeeded()) {
    // Keyword indicates presence of items in IR, which we need to (non-optionally) parse
    res = p.parseRegion(loop);            if(res.failed()) return failure();
    res = p.parseKeyword("reenter");      if(res.failed()) return failure();
    res = p.parseRegion(reentry);         if(res.failed()) return failure();
    res = p.parseKeyword("exit");         if(res.failed()) return failure();
    res = p.parseRegion(exit);            if(res.failed()) return failure();
    return success();
  } else {
    // No keyword indicates no Loop items, have no blocks in region
    return success();
  }
}

static void printBranchLoopRegions(OpAsmPrinter &p, Operation* op, Region &loop, Region &reentry, Region &exit){
  if(cast<CommBranch>(op).isLoop()){
    p.printKeywordOrString("loop");
    p.printRegion(loop);
    p.printKeywordOrString("reenter");
    p.printRegion(reentry);
    p.printKeywordOrString("exit");
    p.printRegion(exit);
  }
}

#define GET_OP_CLASSES
#include "src/enzyme_ad/jax/Dialect/Comm/CommOps.cpp.inc"