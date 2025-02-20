ENV_NAME=Freeway
TRAIN_MODES=(0 1 2 3 4 5 6 7)
for METHOD in FuseNetwMerge 
do
    for TRAIN_MODE in "${TRAIN_MODES[@]}"
    do    
        for ((MODE=0; MODE<=TRAIN_MODE; MODE++))
        do
            # python test_agent.py --load "agents/$ENV_NAME/ModelZoo/${ENV_NAME}_${TRAIN_MODE}_${METHOD}_42" --mode $MODE --csv "data/eval_${ENV_NAME}/eval_results.csv" --train_mode $TRAIN_MODE
            python test_agent.py --load "agents/$ENV_NAME/Randn2.5e-3PoolSize6/${ENV_NAME}_${TRAIN_MODE}_${METHOD}_42" --mode $MODE --csv "data/eval_${ENV_NAME}/eval_results.csv" --train_mode $TRAIN_MODE
        done
    done
done