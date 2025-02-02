SEED=42
ENV="ALE/Freeway-v5"
TRAIN_MODES="--first-mode 4 --last-mode 7"
python run_experiments.py   --env $ENV  \
                            --method_type FuseNetwMerge   \
                            $TRAIN_MODES \
                            --seed $SEED \
                            --alpha_learning_rate 2.5e-3 \
                            --delta_theta_mode T \
                            --pool_size 4 \
                            --fuse_actor \
                            --tag "Randn2.5e-3PoolSize4" \
                            
TRAIN_MODES="--first-mode 5 --last-mode 7"
python run_experiments.py   --env $ENV  \
                            --method_type FuseNetwMerge   \
                            $TRAIN_MODES \
                            --seed $SEED \
                            --alpha_learning_rate 2.5e-3 \
                            --delta_theta_mode T \
                            --pool_size 4 \
                            --fuse_actor \
                            --tag "Randn2.5e-3PoolSize5" \

TRAIN_MODES="--first-mode 6 --last-mode 7"
python run_experiments.py   --env $ENV  \
                            --method_type FuseNetwMerge   \
                            $TRAIN_MODES \
                            --seed $SEED \
                            --alpha_learning_rate 2.5e-3 \
                            --delta_theta_mode T \
                            --pool_size 6 \
                            --fuse_actor \
                            --tag "Randn2.5e-3PoolSize6" \
# python run_experiments.py --env $ENV  --method_type Baseline  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type CompoNet  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
                            # --debug
# python run_experiments.py --env $ENV  --method_type PackNet   $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type ProgNet   $TRAIN_MODES --seed $SEED
