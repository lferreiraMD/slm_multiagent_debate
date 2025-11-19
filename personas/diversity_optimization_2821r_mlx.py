#!/usr/bin/env python
# coding: utf-8
"""
MLX-compatible version of diversity_optimization_2821r.py
Optimized for Apple Silicon (M1/M2/M3/M4) using MLX framework

This script optimizes persona selection for cognitive diversity in multiagent debate.
It extracts embeddings from persona descriptions and uses optimization algorithms
to select the most diverse subset.

Usage:
    python3 diversity_optimization_2821r_mlx.py

Requirements:
    - mlx
    - mlx-lm
    - numpy
    - scipy

Model requirements:
    - MLX-optimized Llama 3.1 8B Instruct model
    - Should be downloaded to ~/.cache/huggingface/hub/
    - Or specify a different MLX model by changing model_name

Differences from original (PyTorch/transformers) version:
    - Uses mlx and mlx-lm instead of torch and transformers
    - No HuggingFace login required (models already local)
    - Embedding extraction adapted for MLX model architecture
    - Includes fallback to logits-based embeddings if layer access fails
"""

import mlx.core as mx
from mlx_lm import load
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import differential_evolution
import umap

# Load MLX models
#  qwen3-0.6b:   "Qwen/Qwen3-0.6B-MLX-8bit"
#  vibethinker:  "valuat/VibeThinker-1.5B-mlx-8Bit"
#  deepseek:     "valuat/DeepSeek-R1-Distill-Qwen-1.5B-mlx-fp16"
#  llama32-3b:   "mlx-community/Llama-3.2-3B-Instruct"
#  smallthinker: "valuat/SmallThinker-3B-Preview-mlx-fp16"
#  qwen3-4b:     "Qwen/Qwen3-4B-MLX-8bit"
#  qwen25-7b:    "valuat/Qwen2.5-7B-Instruct-1M-mlx-fp16"
#  llama31-8b:   "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
#  qwen25-14b:   "valuat/Qwen2.5-14B-Instruct-1M-mlx-fp16"
model_name = "Qwen/Qwen3-0.6B-MLX-8bit"
# model_name = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
print(f"Loading model: {model_name}")
model, tokenizer = load(model_name)
print("Model loaded successfully!")

def get_persona_embedding(persona_text):
    """
    Extract embedding from model's hidden states.

    MLX approach: We use the model's internal structure to get hidden representations.
    Since MLX models don't expose hidden states the same way as transformers,
    we access the model's internal architecture.

    Note: This implementation assumes the MLX model follows the standard
    architecture with model.model.layers. If this fails, consider using
    logits-based embeddings or a different MLX model architecture.
    """
    prompt = f"You are {persona_text}."
#
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
    # Convert to numpy for compatibility with scipy
    return np.array(embedding)

# Define candidate personas (expand this list)
candidate_personas = [
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

original_labels = ['Analytical & Methodical', 'Critical & Challenging', 'Intuitive & Creative',
                   'Empathetic & Collaborative', 'Strategic & Big-Picture', 'Detail-Oriented & Precise',
                   'Dynamic & Energetic', 'Reflective & Philosophical', 'Efficient & Results-Oriented',
                   'Balanced & Moderate']

colors = [item for item in list(range(1,11)) for _ in range(5)]

# Extract embeddings for all candidates
print(f"\nExtracting embeddings for {len(candidate_personas)} personas...")
print("This may take a few minutes...")
embeddings = []
for i, persona in enumerate(candidate_personas):
    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(candidate_personas)} personas...")
    embedding = get_persona_embedding(persona)
    embeddings.append(embedding)

embeddings = np.array(embeddings)
print(f"✓ Complete! Embeddings shape: {embeddings.shape}")

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

# Compute pairwise distance matrix (using cosine distance)
def cosine_distance_matrix(embeddings):
    """Compute pairwise cosine distances"""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    # Cosine similarity matrix
    similarity = normalized @ normalized.T
    # Convert to distance (1 - similarity)
    distance = 1 - similarity
    return distance

distance_matrix = cosine_distance_matrix(embeddings)

# Method 1: Greedy Optimization (Fast)
def greedy_max_diversity(distance_matrix, k=5):
    """Greedy algorithm to select k most diverse items"""
    n = len(distance_matrix)
    selected = [np.random.randint(n)]  # Start with random item

    for _ in range(k - 1):
        # Select item that maximizes minimum distance to selected set
        min_distances = []
        for i in range(n):
            if i not in selected:
                min_dist = min([distance_matrix[i][j] for j in selected])
                min_distances.append((min_dist, i))

        # Add item with maximum min-distance
        _, best_idx = max(min_distances)
        selected.append(best_idx)

    return selected

# Method 2: Exhaustive Search (Exact but slow for large sets)
def exhaustive_max_diversity(distance_matrix, k=5):
    """Exhaustively search all combinations for optimal diversity"""
    n = len(distance_matrix)
    best_diversity = -np.inf
    best_subset = None

    for subset in combinations(range(n), k):
        # Compute Max-Sum diversity
        diversity = sum(distance_matrix[i][j]
                       for i, j in combinations(subset, 2))
        if diversity > best_diversity:
            best_diversity = diversity
            best_subset = subset

    return list(best_subset), best_diversity

# Method 3: Global Optimization (Best balance)
def optimize_diversity(distance_matrix, k=5, method='differential_evolution'):
    """Use scipy optimization to find diverse subset"""
    n = len(distance_matrix)

    def objective(x):
        # Binary vector indicating selection
        selected = np.argsort(x)[:k]
        # Compute negative diversity (minimize negative = maximize positive)
        diversity = sum(distance_matrix[i][j]
                       for i, j in combinations(selected, 2))
        return -diversity

    # Use differential evolution for discrete optimization
    bounds = [(0, 1)] * n
    result = differential_evolution(objective, bounds, seed=42, maxiter=100)

    # Extract top k indices
    selected = np.argsort(result.x)[:k]
    diversity = -result.fun

    return list(selected), diversity

# Run optimization
print("\n=== Running Greedy Optimization ===")
selected_greedy = greedy_max_diversity(distance_matrix, k=5)
diversity_greedy = sum(distance_matrix[i][j]
                       for i, j in combinations(selected_greedy, 2))
print(f"Diversity score: {diversity_greedy:.4f}")
print("\nSelected personas:")
for idx in selected_greedy:
    print(f"  - {candidate_personas[idx]}")

print("\n=== Running Global Optimization ===")
selected_opt, diversity_opt = optimize_diversity(distance_matrix, k=5)
print(f"Diversity score: {diversity_opt:.4f}")
print("\nSelected personas:")
for idx in selected_opt:
    print(f"  - {candidate_personas[idx]}")

# Compute diversity metrics
def compute_metrics(selected, distance_matrix):
    """Compute Max-Sum and Max-Min diversity"""
    distances = [distance_matrix[i][j]
                for i, j in combinations(selected, 2)]
    max_sum = sum(distances)
    max_min = min(distances)
    mean_dist = np.mean(distances)
    return {
        'max_sum': max_sum,
        'max_min': max_min,
        'mean_distance': mean_dist
    }

metrics = compute_metrics(selected_opt, distance_matrix)
print(f"\n=== Diversity Metrics ===")
print(f"Max-Sum: {metrics['max_sum']:.4f}")
print(f"Max-Min: {metrics['max_min']:.4f}")
print(f"Mean pairwise distance: {metrics['mean_distance']:.4f}")
