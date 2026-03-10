#!/bin/sh
# helper script that generates a Graphviz dot file from a given HLO file

# required for hlo-translate, uses gcc if not forced in hydra
export CC=clang

usage() {
    echo "Usage: $0 [-o <output path of dot file>] [-f <matching func.func pattern>] <input hlo file>"
    exit 1
}

OUTPUT=""
PATTERN=""
DEBUG=false
while getopts o:f:d opt; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        f) PATTERN="$OPTARG" ;;
        d) DEBUG=true ;;
        *) usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2
           exit 1
           ;;
    esac
done

shift $((OPTIND - 1))
INPUT="$1"
if [ -z "$INPUT" ]; then
    usage
fi

HLO_TRANSLATE_CMD=./$(bazel cquery --output=starlark --starlark:expr=target.files_to_run.executable.path @xla//xla/hlo/tools:hlo-translate 2>/dev/null)
MLIR_OPT_CMD=./$(bazel cquery --output=starlark --starlark:expr=target.files_to_run.executable.path //:enzymexlamlir-opt 2>/dev/null)

# convert HLO to MLIR
TMP_HLO_TRANSLATE_CMD=$(mktemp)
$HLO_TRANSLATE_CMD --hlo-to-mlir $INPUT -o $TMP_HLO_TRANSLATE_CMD

# filter MLIR to only include the specified `func.func` if provided, otherwise include all
TMP_AWK=$TMP_HLO_TRANSLATE_CMD
if [ -n "$PATTERN" ]; then
    TMP_AWK=$(mktemp)
    awk '/func\.func '"$PATTERN"'/ {p=1} p {print; c+=gsub(/\{/,"{"); c-=gsub(/\}/,"}"); if(c==0) p=0}' $TMP_HLO_TRANSLATE_CMD > $TMP_AWK
fi

# convert MLIR to graphviz dot format
TMP_MLIR_OPT_CMD=$(mktemp)
$MLIR_OPT_CMD --view-op-graph $TMP_AWK 2>$TMP_MLIR_OPT_CMD >/dev/null

if [ -n "$OUTPUT" ]; then
    mv $TMP_MLIR_OPT_CMD $OUTPUT
else
    cat $TMP_MLIR_OPT_CMD
fi

# clean temporary files
if [ "$DEBUG" = true ]; then
    echo "Debug mode enabled, temporary files not removed:"
    echo "  HLO to MLIR output: $TMP_HLO_TRANSLATE_CMD"
    if [ -n "$FUNC_NAME" ]; then
        echo "  Filtered MLIR output: $TMP_AWK"
    fi
    echo "  MLIR to Graphviz output: $TMP_MLIR_OPT_CMD"
else
    rm $TMP_HLO_TRANSLATE_CMD $TMP_AWK $TMP_MLIR_OPT_CMD
fi
