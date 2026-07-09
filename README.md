
######### results for X-Palm dataset 
python run_benchmark.py --dataset xpalm --eval_protocol open_set 

python run_benchmark.py --dataset xpalm --eval_protocol closed_set 


######### results for CASIA-MS dataset 
python run_benchmark.py --dataset casiams --eval_protocol open_set

python run_benchmark.py --dataset casiams --eval_protocol closed_set


######### results for XJTU-UP dataset 
python run_benchmark.py --dataset xjtu --eval_protocol open_set --n_ids 192 

python run_benchmark.py --dataset xjtu --eval_protocol closed_set --n_ids 192 


