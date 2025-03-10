TRAIN_MODE=9
SEED=42
for TRAIN_MODE in 0 1 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19
do
for METHOD in simple finetune prognet packnet masknet creus cbpnet fusenet fusenet_merge
do
    for ((TEST_MODE=0; TEST_MODE<=TRAIN_MODE; TEST_MODE++))
    do
        python test_agent.py    --method $METHOD \
                                --load "./agents/ModelZoo/task_${TRAIN_MODE}__${METHOD}__run_sac__${SEED}" \
                                --task-id $TEST_MODE \
                                --train-task $TRAIN_MODE \
                                --csv "data/eval_results.csv" \
                                --seed $SEED
    done
done
done
