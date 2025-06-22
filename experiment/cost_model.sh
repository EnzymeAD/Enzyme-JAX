#!/bin/bash

configs=(
  "export FUSION_COSTS=true ZERO_COSTS=true"
  "export FUSION_COSTS=false ZERO_COSTS=true"
  "export FUSION_COSTS=true ZERO_COSTS=false"
)
config_names=(
  "baseline"
  "no-fusion"
  "no-zero"
)
platforms=("cpu" "gpu")
models=("bert" "gemma" "gpt2" "jaxmd" "kan1" "kan2" "llama" "maxtext" "nasrnn" "resnet" "searchlesschess" )
datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=cost_model_$datetime.txt
num_repeats=3

export STATS_FILENAME=stats_cost_model_$datetime.csv
touch $STATS_FILENAME
echo "experiment_name,eqsat_time,segments" > $STATS_FILENAME

echo "Cost model ablation" > $filename
echo "--------------------------" >> $filename

for repeat in $(seq 1 $num_repeats); do
    for model in "${models[@]}"; do
        for platform in "${platforms[@]}"; do
            for i in "${!configs[@]}"; do
                config="${configs[$i]}"
                config_name="${config_names[$i]}"
                eval "$config"
                export EQSAT_PLATFORM=$platform
                export ILP_TIME_LIMIT=10
                export SATURATION_TIME_LIMIT=10

                if [ "$platform" == "gpu" ]; then
                    COMMAND="python test/${model}.py"
                else
                    COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
                fi

                export EXPERIMENT_NAME="${model}_cost-model_${config_name}-${platform}_${datetime}_run${repeat}"
                echo "Running $EXPERIMENT_NAME (repeat $repeat/$num_repeats)..." | tee -a $filename

                START_TIME=$(date +%s)
                eval "$COMMAND" >> $filename 2>&1
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))

                echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> $filename
                echo "--------------------------------" >> $filename
            done
        done
    done
done
