## Running token by token:

This will measure and record likelihood and curvature token by token.

```
python run.py --output_name all2 --base_model_name EleutherAI/pythia-2.8B --mask_filling_model_name t5-3b --n_perturbation_list 5 --n_samples 100 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text --tok_by_tok --max_length 150   --revision main
```

If you also went likelihood ratio, do:

```
python run.py --output_name all --base_model_name EleutherAI/pythia-2.8B --mask_filling_model_name t5-3b --n_perturbation_list 5 --n_samples 100 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text --tok_by_tok --max_length 150   --revision main --ref_model facebook/opt-2.7B
```

## Running ONLY MIA without sampling, directly from datasets:
This will compare human written text from two different datasets, as opposed to sample and generate.

```
python run_mia_unified.py --output_name unified_mia --base_model_name EleutherAI/gpt-neo-2.7B --mask_filling_model_name t5-3b --n_perturbation_list 25 --n_samples 2000 --pct_words_masked 0.3 --span_length 2 --cache_dir cache --dataset_member the_pile --dataset_member_key text --dataset_nonmember xsum --ref_model gpt2-xl  --baselines_only --max_length 2000
```

## Result plotting:

All the plots and result analysis are here in this notebook:

```
tok_by_tok.ipynb
```

## Using existing data/meta-data:

unzip the file I sent and place it under results/
then you can use the ipynb from above. 

