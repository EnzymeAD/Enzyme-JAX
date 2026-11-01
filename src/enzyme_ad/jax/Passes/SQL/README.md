# SQL parsing, raising, and lowering

`sql-raising` is a module pass. It raises direct calls to external declarations
of `PQexec`, `PQntuples`, and `PQgetvalue` in either the LLVM or Func dialect.
It also parses constant inputs to existing `sql.unparsed` operations.
`sql-lower` serializes structured queries and restores Func or LLVM PostgreSQL
calls. Neither pass runs queries or participates in the Reactant pipeline yet.

## Parser choice and supported grammar

The parser adapts Polygeist's dependency-free `Parser.cpp`. The hsql-based
`NewParser.cpp` would require a pinned sql-parser repository and Bazel library;
its literal handling and WHERE construction are incomplete as well. Using one
checked parser keeps this milestone independent of that new dependency. A full
SQL frontend remains a possible later replacement behind `parseSQL`.

Supported syntax is one SELECT with integer/column/arithmetic projections (or
`*`), an optional single named FROM table, an optional Boolean WHERE predicate,
and an optional nonnegative integer LIMIT or LIMIT ALL. Arithmetic uses `+`,
`-`, `*`, `/`, and `%`; comparisons use `=`, `!=`, `<>`, `<`, `<=`, `>`, and `>=`.
AND binds more tightly than OR; arithmetic/comparison precedence and parentheses
are preserved. Unary minus is supported on integer literals only. Integer
literals and LIMIT are checked against signed 64-bit bounds.

Keywords are case insensitive, identifiers respect token boundaries, qualified
and double-quoted identifiers retain their SQL spelling, and ordinary line and
block comments are accepted. An absent FROM is an empty `sql.table`; an absent
WHERE wraps the true `sql.constant_bool` predicate. A negative SELECT limit
represents no bound. The parser has a nesting limit and builds IR in a temporary
block so errors leave the caller's IR unchanged.

Aliases, DISTINCT, joins, subqueries, functions, string/float/NULL literals,
predicate projections, NOT, nested comments, ordering/grouping, offsets, and
multiple statements are outside this grammar. Malformed or unsupported known
constant queries diagnose an error and fail the pass; they are not partially
raised or silently accepted. This parser is a bounded SQL frontend, not a full
PostgreSQL grammar or semantic validator.

## ABI and semantics

Handles keep their original LLVM pointer (including address space), memref, or
index type. Counts and row/column operands keep their original integer/index
types. No pointer/integer casts, data-layout guesses, or string/number conversions
are introduced. `PQgetvalue` returns the original text pointer/memref; it does
not imply numeric extraction. Its libpq `(handle, row, column)` order is mapped
to SQL IR's existing `(handle, column, row)` order. All three database operations
have conservative memory effects so they cannot move across result destruction.

Raised operations retain a `sql.source_call` dictionary containing the source
dialect (`kind`), complete original call attribute dictionary (`attributes`),
and original function type (`type`). LLVM calls additionally record their
`calling_convention` explicitly. Argument/result attributes use the original
call ordering. Callee declarations and their attributes are left intact.
Lowering consumes this metadata and reverses the row/column mapping. The
round-trip tests still require execution on Hydra for Milestone 3 validation.

Query bytes are recovered from `sql.str_constant`, or from a NUL-terminated
immutable, non-interposable LLVM string global addressed directly or through an
all-zero GEP. Externally initialized globals are not folded.
EnzymeXLA pointer/memref bridges can be traversed. C-string recovery stops at
the first NUL, never reads past a StringAttr, and does not fold mutable globals,
nonzero offsets, or globals without a terminator. Unknown queries stay
`sql.unparsed` inside a raised `sql.execute`.

Indirect calls, local definitions, variadic LLVM declarations, and LLVM calls
with operand-bundle SSA values remain unchanged. Recognized external calls with
unsupported signatures diagnose an error. Unsupported query encodings and
pointer wrappers remain dynamic. No whole-program identity of an externally
declared libpq symbol is proven.

## Lowering contract

`sql-lower` requires the existing external `PQexec`, `PQntuples`, and
`PQgetvalue` declarations for the operations being lowered. It checks saved
call kinds, types, callees, and calling conventions against those declarations.
Saved inherent properties use MLIR's
[checked property decoder](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html)
so malformed metadata produces diagnostics.
Without metadata, the declaration supplies the ABI and default call properties.
Missing declarations or incompatible signatures are diagnosed; no ABI, handle
conversion, integer extension/truncation, or numeric text extraction is guessed.

Structured SELECTs become NUL-terminated global strings. Every binary expression
is parenthesized to preserve its tree, including right children of subtraction
and division. Empty FROM and absent WHERE are omitted; negative limits remain
unbounded. The bounded parser checks serialized syntax, including textual
leaves, in a temporary block. Dynamic `sql.unparsed` queries pass their original
string operand directly to the reconstructed call when its type matches.

Strings are materialized as LLVM globals in the requested pointer address space,
or memref globals in the requested memory space. Identity rank-one byte memrefs
and unranked byte memrefs are supported; a static extent must fit the string.
Call and declaration query-argument alignment/dereferenceability requirements
determine buffer alignment and padding. Unknown query-argument attributes that
might invalidate rematerialization are rejected. Buffers are limited to 16 MiB.
Symbols use deterministic, collision-safe `__sql_query` names.

Constant string concatenation and SQL expression/Boolean serialization are
supported. A dynamic singleton concatenation is a pass-through; multi-operand
dynamic concatenation is unsupported because no allocation/lifetime runtime has
been specified. Escaping SQL expression values, unsupported producers/operators,
inconsistent metadata, and incompatible string types are errors.

Lowering works on a cloned module. Only after removal of all supported SQL
operations/types and successful MLIR verification is its body installed in the
input module. Failure leaves the original IR unchanged. Cleanup removes the
consumed SQL expression trees; it does not optimize queries or unrelated IR.

## Validation

Run the following on Hydra:

```sh
bazel build //:enzymexlamlir-opt
bazel test --test_output=errors \
  //test/lit_tests:sql/roundtrip.mlir.test \
  //test/lit_tests:sql/support_ops.mlir.test \
  //test/lit_tests:sql/invalid.mlir.test \
  //test/lit_tests:sql/parse.mlir.test \
  //test/lit_tests:sql/parse_invalid.mlir.test \
  //test/lit_tests:sql/raising.mlir.test \
  //test/lit_tests:sql/raising_boundaries.mlir.test \
  //test/lit_tests:sql/raising_invalid.mlir.test \
  //test/lit_tests:sql/lowering.mlir.test \
  //test/lit_tests:sql/lowering_roundtrip.mlir.test \
  //test/lit_tests:sql/lowering_attributes.mlir.test \
  //test/lit_tests:sql/lowering_invalid.mlir.test \
  //test/lit_tests:tessera/roundtrip.mlir.test
```

Tests cover precedence, diagnostics, LLVM/Func raising and lowering, pointer
layouts/address spaces, result widths, row/column order, raw text access, call
metadata, query buffer attributes, dynamic queries, and conservative boundaries.
Successful lowering tests prohibit residual SQL operations/types/metadata and
reparse the output. These are IR-level checks, not PostgreSQL execution tests.
