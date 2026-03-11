#!/bin/sh
# helper script that generates a Graphviz dot file from a given HLO file
# NOTE running this script can invalidate the bazel build cache, because it requires compiling the commands with clang

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

hlo_translate() {
    bazel run --action_env=CC=clang --define using_clang=true --run_under="cd $PWD &&" @xla//xla/hlo/tools:hlo-translate -- $@
}

mlir_opt() {
    bazel run --action_env=CC=clang --define using_clang=true --run_under="cd $PWD &&" //:enzymexlamlir-opt -- $@
}

# convert HLO to MLIR
TMP_HLO_TRANSLATE_CMD=$(mktemp)
hlo_translate --hlo-to-mlir $INPUT -o $TMP_HLO_TRANSLATE_CMD

# filter MLIR to only include the specified `func.func` if provided, otherwise include all
TMP_AWK=$TMP_HLO_TRANSLATE_CMD
if [ -n "$PATTERN" ]; then
    TMP_AWK=$(mktemp)
    awk '/func\.func '"$PATTERN"'/ {p=1} p {print; c+=gsub(/\{/,"{"); c-=gsub(/\}/,"}"); if(c==0) p=0}' $TMP_HLO_TRANSLATE_CMD > $TMP_AWK
fi

# convert MLIR to graphviz dot format
TMP_MLIR_OPT_CMD=$(mktemp)
mlir_opt --view-op-graph $TMP_AWK 2>$TMP_MLIR_OPT_CMD >/dev/null

if [ -n "$OUTPUT" ]; then
    cp --interactive $TMP_MLIR_OPT_CMD $OUTPUT
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
