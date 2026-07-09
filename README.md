
```markdown
### Reproducing Benchmarks

You can run the benchmarks for the respective datasets using the commands below:

```bash
# ==========================================
# 1. Results for X-Palm Dataset
# ==========================================
python run_benchmark.py --dataset xpalm --eval_protocol open_set 
python run_benchmark.py --dataset xpalm --eval_protocol closed_set 

# ==========================================
# 2. Results for CASIA-MS Dataset
# ==========================================
python run_benchmark.py --dataset casiams --eval_protocol open_set
python run_benchmark.py --dataset casiams --eval_protocol closed_set

# ==========================================
# 3. Results for XJTU-UP Dataset
# ==========================================
python run_benchmark.py --dataset xjtu --eval_protocol open_set --n_ids 192 
python run_benchmark.py --dataset xjtu --eval_protocol closed_set --n_ids 192 


