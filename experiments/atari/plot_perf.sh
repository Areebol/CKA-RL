# python gather_rt_results.py --base_dir $1 --env SpaceInvaders
python gather_rt_results.py --base_dir $1
python process_results.py --data-dir $1 