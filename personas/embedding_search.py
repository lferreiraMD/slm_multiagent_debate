#!/usr/bin/env python
# coding: utf-8

"""
The Quest to Find the Optimal Personas
--------------------------------------

MLX-compatible version of diversity_optimization_2821r.py

Optimized for Apple Silicon (M1/M2/M3/M4) using MLX framework

This script optimizes persona selection for cognitive diversity in multiagent debate.
It extracts embeddings from persona descriptions and uses optimization algorithms
to select the most diverse subset.

Usage:
    python3 embedding_search.py

Requirements:
    - mlx, mlx-lm, numpy, scipy, matplotlib, scikit-learn, itertools, TSNE, UMAP

Model requirements:
    - MLX-optimized models
    - Should be downloaded to ~/.cache/huggingface/hub/
    - Specify a different MLX model by changing model_name
"""

import sys
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
import umap
from sklearn.manifold import TSNE

# Add project root to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpu_config import is_vllm_backend, detect_vllm_gpus, print_gpu_summary
from utils.cuda_cleanup import release_cuda_memory

# Conditional imports based on backend
if is_vllm_backend():
    from transformers import AutoModel, AutoTokenizer
    import torch
    BACKEND = 'vllm'
    print("[Backend] vLLM detected - using transformers for embeddings")
else:
    import mlx.core as mx
    from mlx_lm import load
    BACKEND = 'mlx'
    print("[Backend] MLX detected - using MLX for embeddings")

### Auxillary functions
# Saves object as pickle object
def save_object(_obj, _filepath):
    with open(_filepath, 'wb') as outp:
        pickle.dump(_obj, outp, pickle.HIGHEST_PROTOCOL)
#
# # The MaxMin metric ensures all personas are far apart from each other.
def get_diversity_score(indices, distance_matrix):
    """Calculates the minimum pairwise distance within the subset."""
    # np.ix_ allows indexing both rows and columns simultaneously
    sub_matrix = distance_matrix[np.ix_(indices, indices)]
    #
    # Get all unique pairwise distances (excluding diagonal)
    pairwise_dists = sub_matrix[np.triu_indices(len(indices), k=1)]
    #
    # We want to maximize the *minimum* distance
    return np.min(pairwise_dists)
#
# The MaxDet(S) metric ensures all personas are far apart from each other in terms of the determinant of the similarity matrix.
def get_volume_score(indices, similarity_matrix):
    """
    Calculates the determinant of the similarity sub-matrix.
    A larger determinant implies a greater volume spanned by the vectors.
    """
    # Select the 5x5 sub-matrix corresponding to the combination indices
    sub_matrix = similarity_matrix[np.ix_(indices, indices)]
    #
    # Calculate the determinant (the volume score)
    # We must use float64 here for robust determinant calculation to avoid precision errors
    return np.linalg.det(sub_matrix.astype(np.float64))
#
def brute_force_search(_n_personas, _personas_list, _distance_matrix, _similarity_matrix, _small=True):
    '''
    Finds best personas by brute force search, i.e., all combinations possible.
    Returns
    best_indices:       indices corresponding to best MaxMin solution
    best_indices_det:   indices corresponding to best MaxDet(S) solution
    selected_personas_min_dist: selected personas MaxMin
    selected_personas_det_volume: selected personasm MaxDet(S)
    score_list: list of all MaxMin scores
    volume_list: list of all Det(S) scores
    '''
    # --- Brute Force Search for Max-Min Distance ---
    best_score = -1
    best_indices = ()
    score_list = []
    #
    # --- Brute Force Search for Maximum Volume ---
    best_volume = -1.0 # Determinants can be small, but should be positive for a correlation matrix
    best_indices_det = ()
    volume_list = []
    #
    # Iterate through every combination of 5 indices out of 50
    i = 1
    N = _similarity_matrix.shape[0] # N=50
    for combo in combinations(range(N), _n_personas):
        if (i) % 250000 == 0:
            print(f'Combination #{i}')        
        score = get_diversity_score(combo, _distance_matrix)
        score_list.append(score)
        #
        volume = get_volume_score(combo, _similarity_matrix)
        volume_list.append(volume)
        #
        if score > best_score:
            best_score = score
            best_indices = combo
        #
        if volume > best_volume:
            best_volume = volume
            best_indices_det = combo
        #
        i += 1
    #
    print("\n--- Brute Force Results ---")
    print(f"✅ The top {_n_personas} most dissimilar sentence indices are: **{best_indices}**")
    print(f"Maximum separation (Min-Max Distance): **{best_score:.4f}**")
    ## Selection for Max-Min Distance (Max Separation)
    selected_personas_min_dist = [_personas_list[i] for i in best_indices]
    print("### Personas Selected by Max-Min Distance (Maximum Separation) ###")
    for idx, persona in zip(best_indices, selected_personas_min_dist):
        print(f"Index {idx:02d}: {persona}")
    #
    print(f"\n✅ The top {_n_personas} personas maximizing the volume (Max Det(S)) are: **{best_indices_det}**")
    print(f"Maximum Volume/Determinant: **{best_volume:.6f}**")
    ## Selection for Max Det(S) (Maximum Volume/Spread)
    selected_personas_det_volume = [_personas_list[i] for i in best_indices_det]
    print("### Personas Selected by Max Det(S) (Maximum Volume/Spread) ###")
    for idx, persona in zip(best_indices_det, selected_personas_det_volume):
        print(f"Index {idx:02d}: {persona}")
    #
    results = {}
    results['best_indices'] = best_indices
    results['best_indices_det'] = best_indices_det
    results['maxmin_personas'] = selected_personas_min_dist
    results['maxdet_personas'] = selected_personas_det_volume
    results['maxmin_score'] = best_score  # MaxMin distance value
    results['maxdet_volume'] = best_volume  # MaxDet determinant value
    if not _small:
        results['score_list'] = score_list
        results['volume_list'] = volume_list
    return results 
#
# Compute pairwise distance matrix (using cosine distance)
def cosine_distance_matrix(_embeddings):
    """Compute pairwise cosine distances"""
    # Normalize embeddings
    _norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
    _normalized = _embeddings / _norms
    # Cosine similarity matrix
    _similarity_matrix = np.dot(_normalized, _normalized.T)
    # Convert to distance (1 - similarity)
    _distance_matrix = 1 - _similarity_matrix
    return _distance_matrix, _similarity_matrix
#
# Return variable name as a string
def get_variable_name(obj):
    for name, value in globals().items():
        if value is obj:
            return name
    for name, value in locals().items():
        if value is obj:
            return name
    return None
#
def plot_embeddings(_model, _embeddings, _personas, _maxmin_indices, _maxdet_indices, _framework):
    '''
    Plots embeddings for better visualization.
    '''
    # TSNE plot
    nrows = _embeddings.shape[0]
    tsne = TSNE(n_components=2, random_state=42, perplexity=10, metric='cosine') 
    embeddings_tsne_2d = tsne.fit_transform(_embeddings)
    #
    colors = np.full(nrows, 'lightgray', dtype=object) # Default color is light gray
    #
    # Create masks for the two sets
    min_dist_mask = np.zeros(nrows, dtype=bool)
    min_dist_mask[list(_maxmin_indices)] = True
    det_volume_mask = np.zeros(nrows, dtype=bool)
    det_volume_mask[list(_maxdet_indices)] = True
    #
    # Identify points that are in BOTH sets (intersection)
    intersection_mask = min_dist_mask & det_volume_mask
    #
    # Set colors based on priority: Intersection > Max Det(S) Only > Max-Min Only
    colors[intersection_mask] = 'red'
    colors[det_volume_mask & ~intersection_mask] = 'blue'
    colors[min_dist_mask & ~intersection_mask] = 'green'
    #
    # Plotting
    plt.figure(figsize=(10, 8))
    handles = [
        plt.scatter([], [], color='lightgray', label='All Other Personas'),
        plt.scatter([], [], color='green', label='Max-Min Distance Only'),
        plt.scatter([], [], color='blue', label='Max Det(S) Only'),
        plt.scatter([], [], color='red', label='Intersection (Both Sets)')
    ]
    #
    # Scatter plot all points
    plt.scatter(embeddings_tsne_2d[:, 0], embeddings_tsne_2d[:, 1], c=colors, s=50, alpha=0.8)
    all_selected_indices = sorted(list(set(_maxmin_indices + _maxdet_indices)))
    for i in all_selected_indices:
        x_coord, y_coord = embeddings_tsne_2d[i, 0], embeddings_tsne_2d[i, 1]
        # Add a small offset to prevent text from overlapping the dot
        plt.text(x_coord - 1.0, y_coord + 0.5, _personas[i], fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
    #
    model_short = _model.split('/')[1]
    persona_short = get_variable_name(_personas)
    num_personas = len(_maxmin_indices)
    #
    plt.title(f't-SNE Visualization of Persona Diversity Selection - {model_short}', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(handles=handles, loc='best', title="Diversity Criteria")
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    plt.savefig(f'tsne_diversity_comparison_{_framework}_{num_personas}_{persona_short}_{model_short}.png')
    print("tsne_diversity_comparison_with_labels.png generated.")
#
def get_embeddings_mlx(_model, _personas):
    """
    Extract embedding from model's hidden states using MLX backend.

    MLX approach: We use the model's internal structure to get hidden representations.
    Since MLX models don't expose hidden states the same way as transformers,
    we access the model's internal architecture.

    Note: This implementation assumes the MLX model follows the standard
    architecture with model.model.layers. If this fails, consider using
    logits-based embeddings or a different MLX model architecture.
    """
    # Load model
    # _model = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
    print(f"Loading model: {_model}")
    model, tokenizer = load(_model)
    print("Model loaded successfully!")
    #
    # Extract embeddings for all candidates
    print(f"\nExtracting embeddings for {len(_personas)} personas...")
    print("This may take a few minutes...")
    #
    new_embeddings = []
    for i, persona in enumerate(_personas):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(_personas)} personas...")
        prompt = f"You are {persona}."
        # Tokenize input
        inputs = tokenizer.encode(prompt)
        inputs_array = mx.array([inputs])
        #
        # Try to get hidden states by accessing model internals
        try:
            # Access embedding layer
            token_embeddings = model.model.embed_tokens(inputs_array).astype(mx.float32)    
            #
            # Forward through all transformer layers
            hidden_states = token_embeddings
            for layer in model.model.layers:
                # Each layer typically returns (hidden_states, attention_weights)
                # or just hidden_states depending on the implementation
                layer_output = layer(hidden_states)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
            #   
            # Apply final layer norm if it exists
            if hasattr(model.model, 'norm'):
                hidden_states = model.model.norm(hidden_states)
            #
            # Mean pooling across sequence length (axis=1)
            # Shape: (batch, seq_len, hidden_dim) -> (hidden_dim,)
            embedding = mx.mean(hidden_states, axis=1).squeeze()
        #
        except AttributeError as e:
            print(f"Warning: Could not access model layers directly ({e})")
            print("Falling back to logits-based embedding...")
        #
            # Fallback: use output logits as embedding (less ideal but works)
            logits = model(inputs_array)
            # Mean pool over sequence dimension
            embedding = mx.mean(logits, axis=1).squeeze()
        #
        new_embeddings.append(embedding)
    #
    new_embeddings = np.array(new_embeddings).astype(np.float64)
    print(f"✓ Complete! Embeddings shape: {new_embeddings.shape}")
    return new_embeddings

def get_embeddings_vllm(_model, _personas):
    """
    Extract embeddings using vLLM backend (Linux with NVIDIA GPUs).

    Uses transformers AutoModel with GPU auto-configuration for optimal performance.
    """
    # Detect GPUs and print summary
    gpu_info = detect_vllm_gpus()
    if gpu_info and gpu_info['count'] > 0:
        print_gpu_summary(gpu_info, use_case=None)

        # For embedding extraction, always use single GPU to avoid tensor placement issues
        # (We process one prompt at a time, so no benefit from multi-GPU sharding)
        device_map = {"": "cuda:0"}
        if gpu_info['count'] > 1:
            print(f"[GPU Config] Multi-GPU detected: Using cuda:0 for embeddings (single-prompt processing)")
        else:
            print(f"[GPU Config] Single GPU: Loading model on cuda:0")
    else:
        # CPU fallback
        device_map = {"": "cpu"}
        print("[GPU Config] No GPUs detected: Using CPU (will be slow)")

    print(f"\nLoading model: {_model}")

    # Load model with optimal device mapping
    model = AutoModel.from_pretrained(
        _model,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(_model, trust_remote_code=True)

    print("Model loaded successfully!")

    # Extract embeddings for all personas
    print(f"\nExtracting embeddings for {len(_personas)} personas...")
    print("This may take a few minutes...")

    new_embeddings = []
    for i, persona in enumerate(_personas):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(_personas)} personas...")

        prompt = f"You are {persona}."

        # Tokenize input (no padding needed - processing single prompts)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        # Move inputs to same device as model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Extract hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Get last hidden layer
            # outputs.hidden_states is a tuple of (num_layers + 1) tensors
            # Shape: (batch_size, seq_len, hidden_dim)
            hidden_states = outputs.hidden_states[-1]

            # Mean pooling across sequence length (axis=1)
            # Shape: (batch_size, hidden_dim) -> (hidden_dim,)
            embedding = hidden_states.mean(dim=1).squeeze()

        # Move to CPU and convert to numpy
        new_embeddings.append(embedding.cpu().numpy())

    new_embeddings = np.array(new_embeddings).astype(np.float64)
    print(f"✓ Complete! Embeddings shape: {new_embeddings.shape}")

    # Clean up GPU memory aggressively (critical for sequential model loading)
    print("[GPU Cleanup] Releasing GPU memory...")
    del model
    del tokenizer
    release_cuda_memory(delay=2.0, verbose=True)

    return new_embeddings

def get_embeddings(_model, _personas):
    """
    Route to appropriate embedding extractor based on detected backend.

    Automatically detects MLX (Mac) vs vLLM (Linux/HPC) and uses
    the correct implementation for embedding extraction.
    """
    if is_vllm_backend():
        return get_embeddings_vllm(_model, _personas)
    else:
        return get_embeddings_mlx(_model, _personas)

#### List of available models:
## Regular LLMs:
# qwen3-0.6b:   "Qwen/Qwen3-0.6B-MLX-8bit"
# vibethinker:  "valuat/VibeThinker-1.5B-mlx-8Bit"
# deepseek:     "valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-8Bit"
# qwen3-1.7b:   "Qwen/Qwen3-1.7B-MLX-8bit"
# llama32-3b:   "mlx-community/Llama-3.2-3B-Instruct-8bit"
# smallthinker: "valuat/SmallThinker-3B-Preview-mlx-8Bit"
# qwen3-4b:     "Qwen/Qwen3-4B-MLX-8bit"
# llama31-8b:   "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
# qwen3-8b:     "Qwen/Qwen3-8B-MLX-8bit"
# qwen3-14b:    "Qwen/Qwen3-14B-MLX-8bit"
# oss-gpt-20b:  "mlx-community/gpt-oss-20b-MXFP4-Q8"
## Embedding models:
# all-MiniLM-L6: "mlx-community/all-MiniLM-L6-v2-8bit"
# qwen3-embed-600m: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

model_dict = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-MLX-8bit",
    "vibethinker": "valuat/VibeThinker-1.5B-mlx-8Bit",
    "deepseek": "valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-8Bit",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B-MLX-8bit",
    "llama32-3b": "mlx-community/Llama-3.2-3B-Instruct-8bit",
    "smallthinker": "valuat/SmallThinker-3B-Preview-mlx-8Bit",
    "qwen3-4b": "Qwen/Qwen3-4B-MLX-8bit",
    "llama31-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit",
    "qwen3-8b": "Qwen/Qwen3-8B-MLX-8bit",
    "qwen3-14b": "Qwen/Qwen3-14B-MLX-8bit",
#    "oss-gpt-20b": "mlx-community/gpt-oss-20b-MXFP4-Q8",
#    "all-MiniLM-L6": "mlx-community/all-MiniLM-L6-v2-8bit",
    "qwen3-embed-600m": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
}

model_dict_vllm = {
    "vllm-qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "vllm-vibethinker": "WeiboAI/VibeThinker-1.5B",
    "vllm-deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "vllm-qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "vllm-llama32-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "vllm-smallthinker": "PowerInfer/SmallThinker-3B-Preview",
    "vllm-qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "vllm-llama31-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "vllm-qwen3-8b": "Qwen/Qwen3-8B",
    "vllm-qwen3-14b": "Qwen/Qwen3-14B",
#   "vllm-oss-20b": "openai/gpt-oss-20b"
}

### DATA
# Define candidate personas (expand this list)
personas_v1 = [
    # Analytical & Methodical
    "a systematic thinker who approaches problems methodically",
    "a meticulous analyst who examines every detail carefully",
    "a logical reasoner who prioritizes evidence and structure",
    "a thorough investigator who leaves no stone unturned",
    "a pragmatic problem-solver focused on practical solutions",
    #
    # Critical & Challenging
    "a skeptical questioner who challenges assumptions rigorously",
    "a critical thinker who identifies flaws and weaknesses",
    "a contrarian debater who explores opposing viewpoints",
    "a rigorous evaluator who demands high standards of proof",
    "a provocative challenger who tests ideas through confrontation",
    #
    # Intuitive & Creative
    "an intuitive thinker who relies on pattern recognition and gut feelings",
    "a creative innovator who generates novel solutions and perspectives",
    "an imaginative visionary who thinks beyond conventional boundaries",
    "a lateral thinker who makes unexpected connections",
    "an experimental explorer willing to try unconventional approaches",
    #
    # Empathetic & Collaborative
    "a compassionate mediator who considers emotional and human impacts",
    "a diplomatic consensus-builder who finds common ground",
    "an empathetic listener who understands diverse perspectives",
    "a collaborative facilitator who brings ideas together harmoniously",
    "a warm supporter who encourages and validates contributions",
    #
    # Strategic & Big-Picture
    "a strategic planner who thinks several steps ahead",
    "a visionary leader who focuses on long-term implications",
    "a holistic thinker who sees systems and interconnections",
    "a philosophical questioner who explores deeper meanings",
    "a futuristic forecaster who anticipates consequences and trends",
    #
    # Detail-Oriented & Precise
    "a perfectionist who insists on accuracy and precision",
    "a cautious deliberator who weighs risks carefully",
    "a conservative guardian who values proven approaches",
    "a meticulous fact-checker who verifies every claim",
    "a disciplined executor who follows processes strictly",
    #
    # Dynamic & Energetic
    "an enthusiastic advocate who argues passionately for positions",
    "a bold risk-taker who embraces uncertainty",
    "an assertive leader who drives discussions forward decisively",
    "an adventurous experimenter eager to explore new territory",
    "a spontaneous improviser who adapts quickly to new information",
    #
    # Reflective & Philosophical
    "a reflective contemplator who thinks deeply before responding",
    "a philosophical questioner who examines fundamental assumptions",
    "a thoughtful observer who considers multiple angles quietly",
    "an introspective analyst who explores internal logic",
    "a patient deliberator who takes time to form well-reasoned views",
    #
    # Efficient & Results-Oriented
    "a pragmatic doer focused on actionable outcomes",
    "an efficient optimizer who seeks the most direct path",
    "a results-driven achiever who prioritizes concrete goals",
    "a decisive executor who makes quick, firm judgments",
    "a competitive winner who strives to prevail in debates",
    #
    # Balanced & Moderate
    "an impartial judge who weighs all sides fairly",
    "a balanced moderator who seeks middle ground",
    "a reasonable negotiator who finds practical compromises",
    "an objective observer who maintains emotional distance",
    "a fair arbiter who considers evidence without bias"
]

personas_v2 = [
'a nihilistic cryptographer who only trusts solutions verifiable by zero-knowledge proofs',
'a Baroque music theorist fixated on harmonic counterpoint and structural symmetry',
'a chaotic systems meteorologist who views all certainty as a transient statistical anomaly',
'a hard-science fiction xenolinguist obsessed with logical consistency and species-specific syntax',
'a medieval cartographer whose primary concern is establishing clear boundaries and known territories',
'a neuropharmacologist who analyzes all input as complex patterns of reward and aversion chemicals',
'an extreme minimalist architect whose goal is to strip the solution down to its bare, essential structure',
'a quantum physicist who models all decisions as probabilistic wave function collapse',
'a Roman imperial jurist who strictly adheres to precedent and codified legal language',
'an early 20th-century urban planner focused on maximizing geometric efficiency and transit flow',
'a radical anarchist who views all imposed structures and hierarchies as fundamentally flawed',
'a utilitarian extremist who focuses solely on maximizing the benefit for the greatest number, regardless of individual cost',
'a Schopenhauerian pessimist who assumes the most detrimental outcome is inevitable and prepares for it',
'a Kantian deontologist who judges all actions strictly by their moral imperative and universal rule application',
'a contrarian skeptic whose sole purpose is to argue the inverse of the majority opinion at all times',
'an unfettered Romantic idealist who prioritizes artistic vision and emotional resonance over cold logic',
'a hyper-capitalist financier who evaluates every move solely on ROI (Return on Investment) and marginal profit',
'a pre-Socratic materialist who reduces all problems to their basic physical components and forces',
'a Taoist sage who seeks the path of least effort and accepts paradoxical outcomes',
'a solipsistic egoist who only values input that directly confirms or serves their own internal world view',
'an enigma machine operator whose primary filter is signal-to-noise ratio and encrypted hidden messages',
'a Gothic novelist who focuses on dramatic irony, foreshadowing, and latent horror',
'a childish prankster whose motivation is to introduce creative chaos and subvert expectations',
'a Victorian etiquette consultant obsessed with proper formatting, deference, and rigid social boundaries',
'a Soviet-era bureaucrat who prioritizes documentation, adherence to arbitrary quotas, and triplicate forms',
'a pre-lingual infant whose understanding is limited to basic sensations, needs, and spatial presence',
'a post-modern deconstructionist who questions the validity and inherent meaning of all proposed terms',
'a dread pirate captain whose strategy is based on risk assessment, immediate plunder, and intimidation',
'a Zen master who communicates only through non-sequiturs, koans, and minimal, cryptic statements',
'a 1950s Madison Avenue ad man who views the solution as a campaign to be emotionally sold and packaged',
'a deep-sea volcanologist focused on extremes of pressure, heat, and slow geologic change',
'a forensic pathologist who works backward from the failure state to determine the precise cause of death',
'a cosmic horror narrator who frames the problem as an ancient, unknowable, and terrifying truth',
'a systems engineer who focuses strictly on modularity, inter-component dependencies, and error states',
'a stand-up comedian who evaluates suggestions based on their absurdity, timing, and satirical value',
'a resource auditor who views every input as a budget line item that must be strictly justified',
'a mythological hero\'s sidekick who is overly cautious, risk-averse, and constantly points out dangers',
'an expert chess grandmaster who analyzes all moves based on look-ahead, board state, and counterplay',
'a Renaissance painter who values perspective, light, shadow, and visual harmony in the final presentation',
'an industrial safety inspector whose primary filter is identifying catastrophic single points of failure',
'a historical revisionist who assumes all primary sources are propaganda and seeks hidden motives',
'a drone swarm commander who views the task as parallel processing over numerous, interchangeable units',
'a child psychologist who interprets all actions through the lens of developmental stage and emotional need',
'a hermetic alchemist who seeks to transmute the problem into a perfect, pure, and philosophical gold standard',
'a professional wrestler whose analysis focuses on dramatic confrontation, stage presence, and audience reaction',
'a street-smart gambler whose strategy is based on calculating odds, weighted risk, and bluffing potential',
'a hermitic survivalist who prioritizes self-sufficiency, redundancy, and defense against external threats',
'a city gossip columnist who filters information based on intrigue, conflict, and personal scandal potential',
'a post-apocalyptic scavenger who judges solutions only by their immediate utility, robustness, and salvageability',
'a linguistic anthropologist who views the solution as a structure of cultural symbols and shared meaning'
]

def create_output_filepath(_path, _model, _persona_list, _num_personas, _framework):
    _persona_short = get_variable_name(_persona_list)
    _model_short = _model.split('/')[1]
    _final_path = os.path.join(_path, f"results_{_framework}_{_num_personas}_agents_{_persona_short}_{_model_short}.pk")
    return _final_path
#
def run_it_all(_output_dir, _model_dict, _num_personas, _persona_list, _framework):
    for _key in _model_dict.keys():
        _model = _model_dict[_key]
        _embed = get_embeddings(_model, _persona_list)
        _distance_matrix, _similarity_matrix = cosine_distance_matrix(_embed)
        _results = brute_force_search(_num_personas, _persona_list, _distance_matrix, _similarity_matrix)
        _results_path = create_output_filepath(_output_dir, _model, _persona_list, _num_personas, _framework)
        save_object(_results, _results_path)
        plot_embeddings(_model, _embed, _persona_list, _results['best_indices'], _results['best_indices_det'], _framework)
#
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract embeddings and optimize persona selection for cognitive diversity"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--num-personas",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7],
        help="List of persona counts to test (default: 2 3 4 5 6 7)"
    )
    args = parser.parse_args()

    # Validate output directory
    OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[Config] Output directory: {OUTPUT_DIR}")

    num_personas_list = args.num_personas
    personas_list = [personas_v1, personas_v2]

    # Auto-detect backend and select appropriate model dict
    if is_vllm_backend():
        models = model_dict_vllm
        framework = 'vllm'
        print(f"[Config] Backend: vLLM (Linux/HPC)")
    else:
        models = model_dict
        framework = 'mlx'
        print(f"[Config] Backend: MLX (Mac)")

    print(f"[Config] Testing persona counts: {num_personas_list}")
    print(f"[Config] Persona sets: v1 (moderate), v2 (extreme)")
    print(f"[Config] Models: {len(models)}")
    print()

    for num_personas in num_personas_list:
        for personas in personas_list:
            run_it_all(OUTPUT_DIR, models, num_personas, personas, framework)




if __name__ == "__main__":
    main()



# cat+grep line: cat persona_v2_results.txt | grep -E "top|Loading|Min-Max|Determinant|Index" > persona_v2_data.txt

'''
# Computing UMAP embeddings (2D, 3D)
reducer_2d = umap.UMAP(n_components=2, random_state=42)
reducer_3d = umap.UMAP(n_components=3, random_state=42)

umap_embeddings_2d = reducer_2d.fit_transform(embeddings)
print(f"✓ UMAP embeddings (2D) shape: {umap_embeddings_2d.shape}")

umap_embeddings_3d = reducer_3d.fit_transform(embeddings)
print(f"✓ UMAP embeddings (3D) shape: {umap_embeddings_3d.shape}")

#2D plot
fig = plt.figure(figsize=(10, 10))


#3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(umap_embeddings_3d[:, 0], umap_embeddings_3d[:, 1], umap_embeddings_3d[:, 2], c=colors,
           cmap='Spectral', s=50
           )
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.set_zlabel('UMAP 3')
ax.set_title('UMAP Embeddings of Personas (3D)')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()
'''