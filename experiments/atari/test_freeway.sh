ENV_NAME=Freeway
TRAIN_MODE=7
# for METHOD in PackNet CompoNet FN
for METHOD in TV1
do
    for MODE in {0..7}
    do
        python test_agent.py --load "agents/$ENV_NAME/${ENV_NAME}_${TRAIN_MODE}_${METHOD}" --mode $MODE --csv "data/eval_${ENV_NAME}/eval_results.csv"
    done
done