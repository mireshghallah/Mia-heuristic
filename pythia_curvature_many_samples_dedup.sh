
python run.py --output_name all --base_model_name EleutherAI/pythia-160m-deduped --mask_filling_model_name t5-3b --n_perturbation_list 50 --n_samples 5000 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text   --revision main --ref_model facebook/opt-125m
python run.py --output_name all --base_model_name EleutherAI/pythia-2.8B-deduped --mask_filling_model_name t5-3b --n_perturbation_list 50 --n_samples 5000 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text   --revision main --ref_model facebook/opt-2.7B
python run.py --output_name all --base_model_name EleutherAI/pythia-6.9B-deduped --mask_filling_model_name t5-3b --n_perturbation_list 50 --n_samples 5000 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text   --revision main --ref_model facebook/6.7B



#python run.py --output_name pythia --base_model_name EleutherAI/pythia-70m-deduped --mask_filling_model_name t5-3b --n_perturbation_list 25 --n_samples 312 --pct_words_masked 0.3 --span_length 2 --dataset the_pile --dataset_key text   --revision main
