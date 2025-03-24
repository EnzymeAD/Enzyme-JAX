#!/bin/bash

thresholds=(5 10 25 100 500 1000 5000)
platforms=("cpu" "gpu")
models=("llama" "maxtext" "jaxmd" "kan1" "kan2" "gemma" "bert" "maxtext" "searchlesschess" "nasrnn" "resnet')
datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=segmentation_$datetime.txt
segmentation_size_csv="stats_segmentation_2025-03-23_11:58:22.csv"

export STATS_FILENAME=stats_segmentation_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

echo "Segmentation" > $filename
echo "--------------------------" >> $filename

run_experiment() {
    local model=$1
    local platform=$2
    local experiment_name=$3
    local command=$4

    echo "Running $experiment_name..." | tee -a $filename
    START_TIME=$(date +%s)
    eval "$command" >> $filename 2>&1
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo "$experiment_name: ${DURATION} seconds" >> $filename
    echo "--------------------------------" >> $filename
}

for model in "${models[@]}"; do
    for platform in "${platforms[@]}"; do
        for threshold in "${thresholds[@]}"; do
            export EXPERIMENT_NAME="${model}_tau=${threshold}-${platform}_${datetime}"
            export SEGMENTATION_THRESHOLD=$threshold
            read ILP_LIMIT SAT_LIMIT <<< $(python compute_time_limits.py --csv "$segmentation_size_csv" --model "$model" --tau "$threshold")
            export EQSAT_PLATFORM=$platform
            export EQSAT_ONLY=true

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
