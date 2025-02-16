SEED=42
ENV="ALE/SpaceInvaders-v5"
STEPS=1400000
# TRAIN_MODES="--first-mode 4 --last-mode 9"
# python run_experiments.py   --env $ENV  \
#                             --method_type FuseNetwMerge   \
#                             $TRAIN_MODES \
#                             --seed $SEED \
#                             --alpha_learning_rate 2.5e-3 \
#                             --delta_theta_mode T \
#                             --pool_size 4 \
#                             --fuse_actor \
#                             --tag "Randn2.5e-3PoolSize4" \
                            
# TRAIN_MODES="--first-mode 5 --last-mode 9"
# python run_experiments.py   --env $ENV  \
#                             --method_type FuseNetwMerge   \
#                             $TRAIN_MODES \
#                             --seed $SEED \
#                             --alpha_learning_rate 2.5e-3 \
#                             --delta_theta_mode T \
#                             --pool_size 5 \
#                             --fuse_actor \
#                             --tag "Randn2.5e-3PoolSize5" \

# TRAIN_MODES="--first-mode 6 --last-mode 9"
# python run_experiments.py   --env $ENV  \
#                             --method_type FuseNetwMerge   \
#                             $TRAIN_MODES \
#                             --seed $SEED \
#                             --alpha_learning_rate 2.5e-3 \
#                             --delta_theta_mode T \
#                             --pool_size 6 \
#                             --fuse_actor \
#                             --tag "Randn2.5e-3PoolSize6" \

# TRAIN_MODES="--first-mode 7 --last-mode 9"
# python run_experiments.py   --env $ENV  \
#                             --method_type FuseNetwMerge   \
#                             $TRAIN_MODES \
#                             --seed $SEED \
#                             --alpha_learning_rate 2.5e-3 \
#                             --delta_theta_mode T \
#                             --pool_size 7 \
#                             --fuse_actor \
#                             --tag "Randn2.5e-3PoolSize7" \

TRAIN_MODES="--first-mode 8 --last-mode 9"
python run_experiments.py   --env $ENV  \
                            --method_type FuseNetwMerge   \
                            $TRAIN_MODES \
                            --seed $SEED \
                            --alpha_learning_rate 2.5e-3 \
                            --delta_theta_mode T \
                            --pool_size 8 \
                            --fuse_actor \
                            --tag "Randn2.5e-3PoolSize8" \

# python run_experiments.py --env $ENV  --method_type Baseline  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type CompoNet  --total_timesteps $STEPS $TRAIN_MODES --seed $SEED
                            # --debug
# python run_experiments.py --env $ENV  --method_type PackNet   $TRAIN_MODES --seed $SEED
# python run_experiments.py --env $ENV  --method_type ProgNet   $TRAIN_MODES --seed $SEED
