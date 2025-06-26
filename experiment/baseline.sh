#!/bin/bash

platforms=("cpu" "gpu")
models=("bert" "gpt2" "jaxmd" "kan1" "kan2" "llama" "maxtext" "nasrnn" "resnet" "searchlesschess" )
num_repeats=9

datetime=$(date '+%Y-%m-%d_%H:%M:%S')
filename=baseline_$datetime.txt

echo "Baseline" > $filename
echo "--------------------------" >> $filename

for repeat in $(seq 1 $num_repeats); do
    for model in "${models[@]}"; do
        for platform in "${platforms[@]}"; do
            export EXPERIMENT_NAME="${model}-${platform}_${datetime}_run${repeat}"
            export EQSAT_PLATFORM=$platform
            export ILP_TIME_LIMIT=10
            export SATURATION_TIME_LIMIT=10
    
            if [ "$platform" == "gpu" ]; then
                COMMAND="python test/${model}.py"
            else
                COMMAND="JAX_PLATFORMS=cpu python test/${model}.py"
            fi
    
            # time the command and redirect output to baseline.txt
            echo "Running $EXPERIMENT_NAME..." | tee -a $filename
            START_TIME=$(date +%s)
            eval "$COMMAND" >> $filename 2>&1
            END_TIME=$(date +%s)
            DURATION=$((END_TIME - START_TIME))
    
            # append the timing to baseline.txt
            echo "$EXPERIMENT_NAME: ${DURATION} seconds" >> $filename
            echo "--------------------------------" >> $filename
        done
    done
done
