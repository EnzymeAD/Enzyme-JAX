#!/bin/sh
# helper script that generates a Graphviz dot file from a given HLO file

# required for hlo-translate, uses gcc if not forced in hydra
export CC=clang

usage() {
    echo "Usage: $0 [-o <output path of dot file>] [-f <matching func.func name>] <input hlo file>"
    exit 1
}

OUTPUT=""
FUNC_NAME=""

while getopts "o:f:" ; do
    case $opt in
        o) OUTPUT="$OPTARG" ;;
        f) FUNC_NAME="$OPTARG" ;;
        *) usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2
           exit 1
           ;;
    esac
done

shift $((OPTIND - 1))
INPUT="$1"
if [[ -z "$INPUT" ]]; then
    usage
fi


# convert HLO to MLIR
MLIR_CODE=$(bazel run --run_under="cd $PWD &&" @xla//xla/hlo/tools:hlo-translate -- --hlo-to-mlir $INPUT)

# filter MLIR to only include the specified `func.func` if provided, otherwise include all
if [ -n "$FUNC_NAME" ]; then
    MLIR_CODE=$(echo "$MLIR_CODE" | awk '/func\.func private @${FUNC_NAME}/ {p=1} p {print; c+=gsub(/\{/,"{"); c-=gsub(/\}/,"}"); if(c==0) p=0}')
fi

# convert MLIR to graphviz dot format
echo "$MLIR_CODE" | bazel run --run_under="cd $PWD &&" //:enzymexlamir-opt -- --view-op-graph 2>&1 >/dev/null
if [ -n "$OUTPUT" ]; then
    echo $MLIR_CODE > $OUTPUT
else
    echo $MLIR_CODE
fi
