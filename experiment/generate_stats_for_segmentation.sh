#!/bin/bash

thresholds=(10 25 100 500 1000 5000)
platforms=("cpu")
models=("bert" "gpt2" "jaxmd" "kan1" "kan2" "llama" "maxtext" "nasrnn" "resnet" "searchlesschess" )

datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=segmentation_$datetime.txt

export STATS_FILENAME=stats_segmentation_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

run_experiment() {
    local model=$1
    local platform=$2
    local experiment_name=$3
    local command=$4

    eval "$command" >> $filename 2>&1
}

export EQSAT_ONLY=true

for platform in "${platforms[@]}"; do
    for model in "${models[@]}"; do
        for threshold in "${thresholds[@]}"; do
            export EXPERIMENT_NAME="${model}_tau=${threshold}-${platform}_${datetime}"
            eval "export ENZYME_RULES=false MULTI_RULES=false EQSAT_RULES=false"
            export SEGMENTATION_THRESHOLD=$threshold
            export ILP_LIMIT=2
            export SAT_LIMIT=2
            export EQSAT_PLATFORM=$platform

            if [ "$platform" == "gpu" ]; then
                COMMAND="ILP_TIME_LIMIT=${ILP_LIMIT} SATURATION_TIME_LIMIT=${SAT_LIMIT} python test/${model}.py"
            else
                COMMAND="JAX_PLATFORMS=cpu ILP_TIME_LIMIT=${ILP_LIMIT} SATURATION_TIME_LIMIT=${SAT_LIMIT} python test/${model}.py"
            fi

            run_experiment "$model" "$platform" "$EXPERIMENT_NAME" "$COMMAND"
        done

        unset SEGMENTATION_THRESHOLD
        export SEGMENTATION_OFF=true
        export EXPERIMENT_NAME="${model}_no_segmentation-${platform}"

        run_experiment "$model" "$platform" "$EXPERIMENT_NAME" "$COMMAND"
        unset SEGMENTATION_OFF
    done
done
