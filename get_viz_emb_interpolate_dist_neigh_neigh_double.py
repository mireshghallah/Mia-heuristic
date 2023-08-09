import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time



# 15 colorblind-friendly colors
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")


def load_base_model(base_model):
    print('MOVING BASE MODEL TO GPU...', end='', flush=True)
    start = time.time()
    try:
        mask_model.cpu()
    except NameError:
        pass
    if args.openai_model is None:
        base_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def load_mask_model():
    print('MOVING MASK MODEL TO GPU...', end='', flush=True)
    start = time.time()

    if args.openai_model is None:
        base_model.cpu()
    if not args.random_fills:
        mask_model.to(DEVICE)
    print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    span_length = 2
    n_spans = 1




    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - args.buffer_size)
        search_end = min(len(tokens), end + args.buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
    
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    if not args.random_fills:
        perturbed_texts = texts
        neighbors = [[text] for text in texts]
        for i in range(args.hops):
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in perturbed_texts]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            perturbed_texts2 = apply_extracted_fills(masked_texts, extracted_fills)
            for j,(orig, per) in enumerate(zip(perturbed_texts,perturbed_texts2)):
                while orig == per:
                    print("EQUAL")
                    masked_texts = [tokenize_and_mask(orig, span_length, pct, ceil_pct)]
                    raw_fills = replace_masks(masked_texts)
                    extracted_fills = extract_fills(raw_fills)
                    per = apply_extracted_fills(masked_texts, extracted_fills)[0]    
                    perturbed_texts2[j] = per
            
            perturbed_texts = perturbed_texts2
            for element, element2 in zip(neighbors,perturbed_texts):
                element.append(element2)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
            raw_fills = replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1
    else:
        if args.random_fills_tokens:
            # tokenize base_tokenizer
            tokens = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
            valid_tokens = tokens.input_ids != base_tokenizer.pad_token_id
            replace_pct = args.pct_words_masked * (args.span_length / (args.span_length + 2 * args.buffer_size))

            # replace replace_pct of input_ids with random tokens
            random_mask = torch.rand(tokens.input_ids.shape, device=DEVICE) < replace_pct
            random_mask &= valid_tokens
            random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            # while any of the random tokens are special tokens, replace them with random non-special tokens
            while any(base_tokenizer.decode(x) in base_tokenizer.all_special_tokens for x in random_tokens):
                random_tokens = torch.randint(0, base_tokenizer.vocab_size, (random_mask.sum(),), device=DEVICE)
            tokens.input_ids[random_mask] = random_tokens
            perturbed_texts = base_tokenizer.batch_decode(tokens.input_ids, skip_special_tokens=True)
        else:
            masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
            perturbed_texts = masked_texts
            # replace each <extra_id_*> with args.span_length random words from FILL_DICTIONARY
            for idx, text in enumerate(perturbed_texts):
                filled_text = text
                for fill_idx in range(count_masks([text])[0]):
                    fill = random.sample(FILL_DICTIONARY, span_length)
                    filled_text = filled_text.replace(f"<extra_id_{fill_idx}>", " ".join(fill))
                assert count_masks([filled_text])[0] == 0, "Failed to replace all masks"
                perturbed_texts[idx] = filled_text

    return neighbors


def perturb_texts(texts, span_length, pct, ceil_pct=False):
    chunk_size = args.chunk_size
    if '11b' in mask_filling_model_name:
        chunk_size //= 2

    outputs = []
    for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
        outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
    return outputs


def drop_last_word(text):
    return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
    if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
        p = drop_last_word(p)

    # sample from the openai model
    kwargs = { "engine": args.openai_model, "max_tokens": 200 }
    if args.do_top_p:
        kwargs['top_p'] = args.top_p
    
    r = openai.Completion.create(prompt=f"{p}", **kwargs)
    return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30):
    # encode each text as a list of token ids
    if args.dataset == 'pubmed':
        texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    else:
        all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
        all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

    if args.openai_model:
        # decode the prefixes back into text
        prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
        pool = ThreadPool(args.batch_size)

        decoded = pool.map(_openai_sample, prefixes)
    else:
        decoded = ['' for _ in range(len(texts))]

        # sample from the model until we get a sample with at least min_words words for each example
        # this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
        tries = 0
        while (m := min(len(x.split()) for x in decoded)) < min_words:
            if tries != 0:
                print()
                print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

            sampling_kwargs = {}
            if args.do_top_p:
                sampling_kwargs['top_p'] = args.top_p
            elif args.do_top_k:
                sampling_kwargs['top_k'] = args.top_k
            min_length = 50 if args.dataset in ['pubmed'] else 150
            outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
            decoded = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            tries += 1

    if args.openai_model:
        global API_TOKEN_COUNTER

        # count total number of tokens with GPT2_TOKENIZER
        total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
        API_TOKEN_COUNTER += total_tokens

    return decoded


def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])[:-1]
    labels = labels.view(-1)[1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean()


# Get the log likelihood of each text under the base_model
def get_ll(text):
    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            return -base_model(**tokenized, labels=labels).loss.item()


def get_emb(text, neigh,base_model,base_tokenizer):
    if args.openai_model:        
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        #with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        tokenized_neigh = base_tokenizer(neigh, return_tensors="pt").to(DEVICE)
        

        #print(len(tokenized.input_ids[0]))


        while True:
            min_len = min(len(tokenized.input_ids[0]), len(tokenized_neigh.input_ids[0]))
            # print("min len is, ", min_len,len(tokenized.input_ids[0]),len(tokenized_neigh.input_ids[0]))

            #print(min_len,len(tokenized.input_ids[0]),len(tokenized_neigh.input_ids[0]))

            text_trim = base_tokenizer.decode(tokenized.input_ids[0][:min_len])
            neigh_trim = base_tokenizer.decode(tokenized_neigh.input_ids[0][:min_len])
            # print(text_trim)
            # print(neigh_trim)

            
            tokenized = base_tokenizer(text_trim, return_tensors="pt").to(DEVICE)
            tokenized_neigh = base_tokenizer(neigh_trim, return_tensors="pt").to(DEVICE)

            if  len(tokenized.input_ids[0]) == len(tokenized_neigh.input_ids[0]):
                break
     
        labels = tokenized.input_ids
        labels_neigh = tokenized_neigh.input_ids


        #print("here")

        if 'gpt' in args.base_model_name.lower() :
            embeddings = base_model.transformer.wte(tokenized.input_ids).detach() #.cpu().numpy().squeeze()
            embeddings_neigh = base_model.transformer.wte(tokenized_neigh.input_ids).detach() #.cpu().numpy().squeeze()

        elif 'pythia' in args.base_model_name.lower():
            embeddings = base_model.gpt_neox.embed_in(tokenized.input_ids).detach() #.cpu().numpy().squeeze()
            embeddings_neigh = base_model.gpt_neox.embed_in(tokenized_neigh.input_ids).detach() #.cpu().numpy().squeeze()

        else:
            embeddings = base_model.model.decoder.embed_tokens(tokenized.input_ids).detach() #.cpu().numpy().squeeze()
            embeddings_neigh = base_model.model.decoder.embed_tokens(tokenized_neigh.input_ids).detach() #.cpu().numpy().squeeze()


            #embeddings = np.mean(embeddings,axis = 0).reshape(-1)
        
       
                        
    return embeddings, embeddings_neigh,labels,labels_neigh


def get_distances_embs(point1,point2):

    distances={}

    distances['l2']= np.float64(np.linalg.norm((point1 - point2),ord=2))
    distances['l1']= np.float64(np.linalg.norm((point1 - point2),ord=1))
    distances['linf']= np.float64(np.linalg.norm((point1 - point2),ord=np.inf))
    distances['cos'] = np.float64(np.dot(point1, point2)/(np.linalg.norm(point1)*np.linalg.norm(point2)))

    return distances





# Get the  likelihood ratio of each text under the base_model -- MIA baseline
def get_lira(text):
    if args.openai_model: 
        print("NOT IMPLEMENTED")
        exit(0)       
        kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
        r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
        result = r['choices'][0]
        tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

        assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

        return np.mean(logprobs)
    else:
        with torch.no_grad():
            tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
            labels = tokenized.input_ids
            tokenized_ref = ref_tokenizer(text, return_tensors="pt").to(DEVICE)
            lls =  -base_model(**tokenized, labels=labels).loss.item()
            lls_ref = -ref_model(**tokenized_ref, labels=labels).loss.item()

            return lls - lls_ref



def get_lls(texts):
    if not args.openai_model:
        return [get_ll(text) for text in texts]
    else:
        global API_TOKEN_COUNTER

        # use GPT2_TOKENIZER to get total number of tokens
        total_tokens = sum(len(GPT2_TOKENIZER.encode(text)) for text in texts)
        API_TOKEN_COUNTER += total_tokens * 2  # multiply by two because OpenAI double-counts echo_prompt tokens

        pool = ThreadPool(args.batch_size)
        return pool.map(get_ll, texts)


# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
    assert args.openai_model is None, "get_rank not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        labels = tokenized.input_ids[:,1:]

        # get rank of each label token in the model's likelihood ordering
        matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

        assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

        ranks, timesteps = matches[:,-1], matches[:,-2]

        # make sure we got exactly one match for each timestep in the sequence
        assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

        ranks = ranks.float() + 1 # convert to 1-indexed rank
        if log:
            ranks = torch.log(ranks)

        return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
    assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
        logits = base_model(**tokenized).logits[:,:-1]
        neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
    fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
    precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)





# save the histogram of log likelihoods in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed



# save the histograms of log likelihood ratios in two side-by-side plots, one for real and real perturbed, and one for sampled and sampled perturbed
def get_dict_interpolate_point_neigh(emb_main, emb_neigh,labels_main,labels_neigh,base_model):
    with torch.no_grad():
        emb_diff = emb_main - emb_neigh
        steps = args.steps
        step_emb = emb_diff/steps
        loss_dict = {}
        for step in range(steps):
            new_emb = emb_main - step*step_emb
            #loss_main = -base_model(inputs_embeds = new_emb, labels = labels_main).loss.item()
            #loss_neigh = -base_model(inputs_embeds = new_emb, labels = labels_neigh).loss.item()
  
            output_main = base_model(inputs_embeds = new_emb)
            #output_neigh =  base_model(inputs_embeds = new_emb, labels = labels_neigh)
            
            # loss_main = -output_main.loss.item()
            # loss_neigh = -output_neigh.loss.item()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100) 
            
            shift_logits = output_main.logits[..., :-1, :].contiguous()
            shift_labels = labels_main[..., 1:].contiguous()

            shift_labels_neigh = labels_neigh[..., 1:].contiguous()
            
            # assert  loss_main == -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # assert  loss_neigh == -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_neigh.view(-1))

            #print( output_main.logits , output_neigh.logits)


            loss_main = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().detach().cpu().item()
            loss_neigh =  -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels_neigh.view(-1)).detach().cpu().item()

            loss_dict[f'main_{step}']=loss_main
            loss_dict[f'neigh_{step}']=loss_neigh
            loss_dict[f'inter_{step}'] = (1-(step/steps)) * loss_main + (step/steps)*loss_neigh
            #loss_dict[f'f_ne_{step}'] = -(base_model(inputs_embeds = emb_neigh, labels = labels_neigh)).loss.item()
            
    #print(emb_main.shape)
    #print(emb_neigh.shape)
    
    return loss_dict

def get_perturbation_results():

    load_mask_model()

    torch.manual_seed(0)
    np.random.seed(0)

    results = []
    original_text = data["original"]
    sampled_text = data["sampled"]
    sampled_text2 = data["sampled2"]



    perturb_fn = functools.partial(perturb_texts, span_length=1, pct=args.pct_words_masked)

    perturbed_sampled_t = perturb_fn([x for x in sampled_text for _ in range(n_perturbations)])
    perturbed_sampled2_t = perturb_fn([x for x in sampled_text2 for _ in range(n_perturbations)])

    perturbed_original_t = perturb_fn([x for x in original_text for _ in range(n_perturbations)])

    perturbed_sampled2 = []
    perturbed_sampled = []
    perturbed_original = []

    for it1 in range(len(original_text)):
        temp = []
        temp2= []

        temp_o = []
        for it2 in range((n_perturbations)):
            temp2.extend(perturbed_sampled2_t[it1*(n_perturbations)+it2])
            temp.extend(perturbed_sampled_t[it1*(n_perturbations)+it2])
            temp_o.extend(perturbed_original_t[it1*(n_perturbations)+it2])

        perturbed_sampled2.append(temp2)
        perturbed_sampled.append(temp)
        perturbed_original.append(temp_o)

            


    


    all_dicts = [] #list of dictionaries, each dict is a point

    #print(len(perturbed_sampled2), len(perturbed_original), len(original_text),len(perturbed_original[0]))

    cnt = 0
    for original, sampled, sampled2, original_neighs, sampled_neighs, sampled_neighs2 in tqdm.tqdm(zip(original_text,sampled_text,sampled_text2,perturbed_original,perturbed_sampled,perturbed_sampled2), desc="computing embeddings"):

        # print(sampled)

        # print("*******************")

        # print(sampled_neighs)
        # print(original)
        # print(sampled)
        #print(sampled)
        #print(sampled_neighs)
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)
        
        load_base_model(base_model)

        dict_temp = {} #

        #dict_temp['sample_original'] = get_distances_embs(emb_original, emb_sample)

        sampled_neighs_dicts =[]
        sampled_neighs_dicts2 =[]
        original_neighs_dicts = []
        sampled_original_dicts = []
        cnt2 = 0


        emb_original, emb_original_neigh , labels_original,labels_original_neigh=get_emb(original, original_neighs[0],base_model,base_tokenizer)
        emb_sample, emb_sample_neigh, labels_samp, labels_samp_neigh = get_emb(sampled, sampled_neighs[0],base_model,base_tokenizer)   
        emb_sample2, emb_sample_neigh2, labels_samp2, labels_samp_neigh2 = get_emb(sampled2, sampled_neighs2[0],base_model,base_tokenizer)   

        dict_original_temp = get_dict_interpolate_point_neigh(emb_original, emb_original_neigh,labels_original,labels_original_neigh,base_model)
        dict_sample_temp = get_dict_interpolate_point_neigh(emb_sample, emb_sample_neigh,labels_samp,labels_samp_neigh,base_model)
        dict_sample_temp2 = get_dict_interpolate_point_neigh(emb_sample2, emb_sample_neigh2,labels_samp2,labels_samp_neigh2,base_model)

        #exit(0)


        sampled_neighs_dicts.append(dict_sample_temp)
        sampled_neighs_dicts2.append(dict_sample_temp2)

        original_neighs_dicts.append(dict_original_temp)

        print("cnt is ", cnt, "cnt2 is ", cnt2)
        for sample_no, (or_neigh, samp_neigh,samp_neigh2) in enumerate(zip(original_neighs,sampled_neighs,sampled_neighs2)):
            print(sample_no)
            #print(original)
            #print(or_neigh)
            if sample_no+1 == len(original_neighs):
                break

            #print(original_neighs[sample_no], original_neighs[sample_no+1])
            emb_original, emb_original_neigh , labels_original,labels_original_neigh=get_emb(original_neighs[sample_no], original_neighs[sample_no+1],base_model,base_tokenizer)
            emb_sample, emb_sample_neigh, labels_samp, labels_samp_neigh = get_emb(sampled_neighs[sample_no], sampled_neighs[sample_no+1],base_model,base_tokenizer)   
            emb_sample2, emb_sample_neigh2, labels_samp2, labels_samp_neigh2 = get_emb(sampled_neighs2[sample_no], sampled_neighs2[sample_no+1],base_model,base_tokenizer)   


            dict_original_temp = get_dict_interpolate_point_neigh(emb_original, emb_original_neigh,labels_original,labels_original_neigh,base_model)
            dict_sample_temp = get_dict_interpolate_point_neigh(emb_sample, emb_sample_neigh,labels_samp,labels_samp_neigh,base_model)
            dict_sample_temp2 = get_dict_interpolate_point_neigh(emb_sample2, emb_sample_neigh2,labels_samp2,labels_samp_neigh2,base_model)

            #exit(0)


            sampled_neighs_dicts.append(dict_sample_temp)
            sampled_neighs_dicts2.append(dict_sample_temp2)

            original_neighs_dicts.append(dict_original_temp)


            cnt2+=1




        emb_original, emb_sampled , labels_original,labels_sampled =get_emb(original, sampled,base_model,base_tokenizer)
        dict_sampled_original_temp =  get_dict_interpolate_point_neigh(emb_original, emb_sampled,labels_original,labels_sampled,base_model)
        sampled_original_dicts.append(dict_sampled_original_temp)

        dict_temp['sample_neighs'] = sampled_neighs_dicts
        dict_temp['sample_neighs2'] = sampled_neighs_dicts2

        dict_temp['original_neighs'] = original_neighs_dicts
        dict_temp['original_sample'] = sampled_original_dicts


######################
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name2)
        load_base_model(base_model)

        #dict_temp = {} #

        #dict_temp['sample_original'] = get_distances_embs(emb_original, emb_sample)


        sampled_neighs_dicts =[]
        sampled_neighs_dicts2 =[]
        original_neighs_dicts = []
        sampled_original_dicts = []
        cnt2 = 0


        emb_original, emb_original_neigh , labels_original,labels_original_neigh=get_emb(original, original_neighs[0],base_model,base_tokenizer)
        emb_sample, emb_sample_neigh, labels_samp, labels_samp_neigh = get_emb(sampled, sampled_neighs[0],base_model,base_tokenizer)   
        emb_sample2, emb_sample_neigh2, labels_samp2, labels_samp_neigh2 = get_emb(sampled2, sampled_neighs2[0],base_model,base_tokenizer)   

        dict_original_temp = get_dict_interpolate_point_neigh(emb_original, emb_original_neigh,labels_original,labels_original_neigh,base_model)
        dict_sample_temp = get_dict_interpolate_point_neigh(emb_sample, emb_sample_neigh,labels_samp,labels_samp_neigh,base_model)
        dict_sample_temp2 = get_dict_interpolate_point_neigh(emb_sample2, emb_sample_neigh2,labels_samp2,labels_samp_neigh2,base_model)

        #exit(0)


        sampled_neighs_dicts.append(dict_sample_temp)
        sampled_neighs_dicts2.append(dict_sample_temp2)

        original_neighs_dicts.append(dict_original_temp)


        for sample_no, (or_neigh, samp_neigh,samp_neigh2) in enumerate(zip(original_neighs,sampled_neighs,sampled_neighs2)):
            #print(original)
            #print(or_neigh)
            if sample_no+1 == len(original_neighs):
                break

            #print(original_neighs[sample_no], original_neighs[sample_no+1])
            emb_original, emb_original_neigh , labels_original,labels_original_neigh=get_emb(original_neighs[sample_no], original_neighs[sample_no+1],base_model,base_tokenizer)
            emb_sample, emb_sample_neigh, labels_samp, labels_samp_neigh = get_emb(sampled_neighs[sample_no], sampled_neighs[sample_no+1],base_model,base_tokenizer)   
            emb_sample2, emb_sample_neigh2, labels_samp2, labels_samp_neigh2 = get_emb(sampled_neighs2[sample_no], sampled_neighs2[sample_no+1],base_model,base_tokenizer)   


            dict_original_temp = get_dict_interpolate_point_neigh(emb_original, emb_original_neigh,labels_original,labels_original_neigh,base_model)
            dict_sample_temp = get_dict_interpolate_point_neigh(emb_sample, emb_sample_neigh,labels_samp,labels_samp_neigh,base_model)
            dict_sample_temp2 = get_dict_interpolate_point_neigh(emb_sample2, emb_sample_neigh2,labels_samp2,labels_samp_neigh2,base_model)

            #exit(0)


            sampled_neighs_dicts.append(dict_sample_temp)
            sampled_neighs_dicts2.append(dict_sample_temp2)

            original_neighs_dicts.append(dict_original_temp)


            cnt2+=1




        emb_original, emb_sampled , labels_original,labels_sampled =get_emb(original, sampled,base_model,base_tokenizer)
        dict_sampled_original_temp =  get_dict_interpolate_point_neigh(emb_original, emb_sampled,labels_original,labels_sampled,base_model)
        sampled_original_dicts.append(dict_sampled_original_temp)

        dict_temp['sample_neighs_2'] = sampled_neighs_dicts
        dict_temp['sample_neighs2_2'] = sampled_neighs_dicts2

        dict_temp['original_neighs_2'] = original_neighs_dicts
        dict_temp['original_sample_2'] = sampled_original_dicts











        all_dicts.append(dict_temp)

        cnt +=1
        print("sampe no: ",cnt)
        if cnt > args.sample_cnt:
            break
        #print(dict_temp)
        #exit(0)

    return all_dicts



# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
    return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
    # truncate to shorter of o and s
    shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
    texta = ' '.join(texta.split(' ')[:shorter_length])
    textb = ' '.join(textb.split(' ')[:shorter_length])
    return texta, textb


def truncate_to_substring(text, substring, idx_occurrence):
    # truncate everything after the idx_occurrence occurrence of substring
    assert idx_occurrence > 0, 'idx_occurrence must be > 0'
    idx = -1
    for _ in range(idx_occurrence):
        idx = text.find(substring, idx + 1)
        if idx == -1:
            return text
    return text[:idx]


def generate_samples(raw_data, batch_size):
    torch.manual_seed(42)
    np.random.seed(42)
    data = {
        "original": [],
        "sampled": [],
    }

    for batch in range(len(raw_data) // batch_size):
        print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
        original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
        sampled_text = sample_from_model(original_text, min_words=30 if args.dataset in ['pubmed'] else 55)

        for o, s in zip(original_text, sampled_text):
            if args.dataset == 'pubmed':
                s = truncate_to_substring(s, 'Question:', 2)
                o = o.replace(custom_datasets.SEPARATOR, ' ')

            o, s = trim_to_shorter_length(o, s)

            # add to the data
            data["original"].append(o)
            data["sampled"].append(s)
    
    if args.pre_perturb_pct > 0:
        print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
        load_mask_model()
        data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
        load_base_model(base_model)

    return data


# def generate_data(dataset, key):
#     # load data
#     if dataset in custom_datasets.DATASETS:
#         data = custom_datasets.load(dataset, cache_dir)
#     else:
#         data = datasets.load_dataset(dataset, split='train', cache_dir=cache_dir)[key]

#     # get unique examples, strip whitespace, and remove newlines
#     # then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
#     # then take just the examples that are <= 512 tokens (for the mask model)
#     # then generate n_samples samples

#     # remove duplicates from the data
#     data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

#     # strip whitespace around each example
#     data = [x.strip() for x in data]

#     # remove newlines from each example
#     data = [strip_newlines(x) for x in data]

#     # try to keep only examples with > 250 words
#     if dataset in ['writing', 'squad', 'xsum']:
#         long_data = [x for x in data if len(x.split()) > 250]
#         if len(long_data) > 0:
#             data = long_data

#     random.seed(0)
#     random.shuffle(data)

#     data = data[:5_000]

#     # keep only examples with <= 512 tokens according to mask_tokenizer
#     # this step has the extra effect of removing examples with low-quality/garbage content
#     tokenized_data = preproc_tokenizer(data)
#     data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

#     # print stats about remainining data
#     print(f"Total number of samples: {len(data)}")
#     print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

#     return generate_samples(data[:n_samples], batch_size=batch_size)


def load_data_gens(gen_file, gen_file2):

    data = {'original':[],'sampled':[], 'sampled2':[]}
    neighbors = {'perturbed_original':[], 'perturbed_sampled':[]}



    for  line_b in open(gen_file,'r'):
        di_b= json.loads(line_b)
        info_raw = di_b['raw_results']
        #we only want the sampled stuff - gen_a is original and gen_b is sampled
    for  line_b in open(gen_file2,'r'):
        di_b= json.loads(line_b)
        info_raw2 = di_b['raw_results']



        for i,(item,item2) in enumerate(zip(info_raw,info_raw2)):
            if i > 10:
                break
            data['original'].append( item['original'])
            data['sampled'].append( item['sampled'])
            data['sampled2'].append( item2['sampled'])

            neighbors['perturbed_sampled'].append(item['perturbed_sampled'])
            neighbors['perturbed_original'].append(item['perturbed_original'])




    return data, neighbors

def load_base_model_and_tokenizer(name):
    if args.openai_model is None:
        print(f'Loading BASE model {name}...')
        base_model_kwargs = {'revision':args.revision}
        if 'gpt-j' in name or 'neox' in name:
            base_model_kwargs.update(dict(torch_dtype=torch.float16))
        if 'gpt-j' in name:
            base_model_kwargs.update(dict(revision='float16'))
        base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
    else:
        base_model = None

    optional_tok_kwargs = {}
    if "facebook/opt-" in name:
        print("Using non-fast tokenizer for OPT")
        optional_tok_kwargs['fast'] = False
    if args.dataset in ['pubmed']:
        optional_tok_kwargs['padding_side'] = 'left'
    base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
    base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    return base_model, base_tokenizer


def eval_supervised(data, model):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

    real, fake = data['original'], data['sampled']

    with torch.no_grad():
        # get predictions for real
        real_preds = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,0].tolist())
        
        # get predictions for fake
        fake_preds = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,0].tolist())

    predictions = {
        'real': real_preds,
        'samples': fake_preds,
    }

    fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
    p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
    print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

    # free GPU memory
    del detector
    torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'info': {
            'n_samples': n_samples,
        },
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
    }


if __name__ == '__main__':
    DEVICE = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_key', type=str, default="document")
    parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    
    parser.add_argument('--span_length', type=int, default=2)
    parser.add_argument('--sample_cnt', type=int, default=4)

    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--hops', type=int, default=14)



    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_perturbation_list', type=str, default="1,10")
    parser.add_argument('--n_perturbation_rounds', type=int, default=1)

    parser.add_argument('--base_model_name', type=str, default="gpt2-medium")
    parser.add_argument('--base_model_name2', type=str, default="gpt2-medium")

    #parser.add_argument('--src_model_name', type=str, default="gpt2-medium")

    parser.add_argument('--revision', type=str, default="main")
    parser.add_argument('--scoring_model_name', type=str, default="")
    parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--chunk_size', type=int, default=20)
    parser.add_argument('--n_similarity_samples', type=int, default=20)
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--base_half', action='store_true')
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--output_name', type=str, default="")
    parser.add_argument('--openai_model', type=str, default=None)
    parser.add_argument('--openai_key', type=str)
    parser.add_argument('--baselines_only', action='store_true')
    parser.add_argument('--skip_baselines', action='store_true')
    parser.add_argument('--buffer_size', type=int, default=1)
    parser.add_argument('--mask_top_p', type=float, default=1.0)
    parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
    parser.add_argument('--pre_perturb_span_length', type=int, default=5)
    parser.add_argument('--random_fills', action='store_true')
    parser.add_argument('--random_fills_tokens', action='store_true')
    parser.add_argument('--cache_dir', type=str, default="/trunk/model-hub")

    #cross stuff
    parser.add_argument('--root_gen', type=str, default="results/mistral/")
    #parser.add_argument('--gen_file', type=str, default="")

    
    # lira stuff
    parser.add_argument('--ref_model', type=str, default=None)



    args = parser.parse_args()

    API_TOKEN_COUNTER = 0

    if args.openai_model is not None:
        import openai
        assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
        openai.api_key = args.openai_key

    START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
    START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

    # define SAVE_FOLDER as the timestamp - base model name - mask filling model name
    # create it if it doesn't exist
    precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
    sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
    output_subfolder = f"{args.output_name}/" if args.output_name else ""
    if args.openai_model is None:
        base_model_name = args.base_model_name.replace('/', '_')
    else:
        base_model_name = "openai-" + args.openai_model.replace('/', '_')
    if args.openai_model is None:
        base_model_name2 = args.base_model_name2.replace('/', '_')
    else:
        base_model_name2 = "openai-" + args.openai_model.replace('/', '_')
    # if args.openai_model is None:
    #     src_model_name = args.src_model_name.replace('/', '_')
    # else:
    #     src_model_name = "openai-" + args.openai_model.replace('/', '_')
    scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
#    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
    if args.ref_model is not None:
        ref_model_string = f'--{args.ref_model}'
    else:
        ref_model_string = ""

    if args.span_length ==2 :
        span_length_string = ""
    else:
        span_length_string = f'--{args.span_length}'

    SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}-{args.revision}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}{ref_model_string}{span_length_string}"
    SAVE_FOLDER2 = f"tmp_results/{output_subfolder}{base_model_name2}-{args.revision}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}{ref_model_string}{span_length_string}"

    distance_file = SAVE_FOLDER.replace("tmp_results", "results")+f"/interpolate_emb_smaller_{base_model_name}_{base_model_name2}_{args.n_perturbation_list}_{args.sample_cnt}_{args.hops}_{args.steps}_neigh_neigh.json"
    ##don't run if exists!!!
    # print(f"{distance_file}")
    # if  os.path.isfile((distance_file)):
    #     print(f"folder exists, not running this exp {distance_file}")
    #     exit(0)

    gen_file = SAVE_FOLDER.replace("tmp_results", "results")+"/perturbation_25_d_results.json"
    gen_file2 = SAVE_FOLDER2.replace("tmp_results", "results")+"/perturbation_25_d_results.json"

    # if not os.path.exists(SAVE_FOLDER):
    #     os.makedirs(SAVE_FOLDER)
    # print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")




    # # write args to file
    # with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
    #     json.dump(args.__dict__, f, indent=4)
        

    mask_filling_model_name = args.mask_filling_model_name
    n_samples = args.n_samples
    batch_size = args.batch_size
    n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
    n_perturbation_rounds = args.n_perturbation_rounds
    n_similarity_samples = args.n_similarity_samples

    cache_dir = args.cache_dir
    os.environ["XDG_CACHE_HOME"] = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    print(f"Using cache dir {cache_dir}")

    GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)

    # generic generative model
    base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)

    #reference model if we are doing the lr baseline
    if args.ref_model is not None :
        ref_model, ref_tokenizer = load_base_model_and_tokenizer(args.ref_model)

    # mask filling t5 model
    if not args.baselines_only and not args.random_fills:
        int8_kwargs = {}
        half_kwargs = {}
        if args.int8:
            int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
        elif args.half:
            half_kwargs = dict(torch_dtype=torch.bfloat16)
        print(f'Loading mask filling model {mask_filling_model_name}...')
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(mask_filling_model_name, **int8_kwargs, **half_kwargs, cache_dir=cache_dir)
        try:
            n_positions = mask_model.config.n_positions
        except AttributeError:
            n_positions = 512
    else:
        n_positions = 512
    preproc_tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small', model_max_length=512, cache_dir=cache_dir)
    mask_tokenizer = transformers.AutoTokenizer.from_pretrained(mask_filling_model_name, model_max_length=n_positions, cache_dir=cache_dir)
    if args.dataset in ['english', 'german']:
        preproc_tokenizer = mask_tokenizer

    load_base_model(base_model)

    # print(f'Loading dataset {args.dataset}...')
    # data = generate_data(args.dataset, args.dataset_key)
    #load the generations+samples
    
    data , neighbors = load_data_gens(gen_file,gen_file2) #we will return gen_a as original (human) and gen_b as the generated text
    if args.random_fills:
        FILL_DICTIONARY = set()
        for texts in data.values():
            for text in texts:
                FILL_DICTIONARY.update(text.split())
        FILL_DICTIONARY = sorted(list(FILL_DICTIONARY))


    
    if args.scoring_model_name:
        print(f'Loading SCORING model {args.scoring_model_name}...')
        del base_model
        del base_tokenizer
        torch.cuda.empty_cache()
        base_model, base_tokenizer = load_base_model_and_tokenizer(args.scoring_model_name)
        load_base_model(base_model)  # Load again because we've deleted/replaced the old model


    outputs = []

    if not args.baselines_only:
        # run perturbation experiments
        for n_perturbations in n_perturbation_list:
            all_distances = get_perturbation_results()


    with open(distance_file, "w") as f:
        print(f"Writing raw data to {distance_file}")
        data = {'distances': all_distances}
        json.dump(data, f)

    #######SAVE

    # move results folder from tmp_results/ to results/, making sure necessary directories exist
#    new_folder = SAVE_FOLDER.replace("tmp_results_cross", "results_cross")
    # if not os.path.exists(os.path.dirname(new_folder)):
    #     os.makedirs(os.path.dirname(new_folder))
    # os.rename(SAVE_FOLDER, new_folder)

