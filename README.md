## Running token by token:

This will measure and record likelihood and curvature token by token.

```
python run.py --output_name all2 --base_model_name EleutherAI/pythia-2.8B --mask_filling_model_name t5-3b --n_perturbation_list 5 --n_samples 100 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text --tok_by_tok --max_length 150   --revision main
```

If you also went likelihood ratio, do:

```
python run.py --output_name all --base_model_name EleutherAI/pythia-2.8B --mask_filling_model_name t5-3b --n_perturbation_list 5 --n_samples 100 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text --tok_by_tok --max_length 150   --revision main --ref_model facebook/opt-2.7B
```


## Result plotting:

All the plots and result analysis are here in this notebook:

```
tok_by_tok.ipynb
```

## Using existing data/meta-data:

unzip the file I sent and place it under results/
then you can use the ipynb from above. 