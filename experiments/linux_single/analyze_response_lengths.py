#!/usr/bin/env python3
"""Analyze response lengths from multiagent debate output files."""

import json
import sys
import glob

# Find the most recent output file in test directory
test_dir = "."
json_files = glob.glob(f"{test_dir}/*.json")
pickle_files = glob.glob(f"{test_dir}/*.p")
all_files = json_files + pickle_files

if not all_files:
    print(f"No output files found in {test_dir}/")
    sys.exit(1)

# Use most recent file
output_file = max(all_files, key=lambda x: x)
print(f"Analyzing: {output_file}")
print("=" * 80)

# Load the output file
if output_file.endswith('.json'):
    data = json.load(open(output_file, 'r'))
else:
    import pickle
    data = pickle.load(open(output_file, 'rb'))

# Inspect data structure first
print("\nData structure:")
print(f"  Type: {type(data)}")
print(f"  Keys: {list(data.keys())[:3]}...")

first_key = list(data.keys())[0]
print(f"  First key: {first_key}")
print(f"  Value type: {type(data[first_key])}")
print(f"  Value length: {len(data[first_key])}")

# Get agent contexts - structure varies by task
if isinstance(data[first_key], tuple) and len(data[first_key]) > 0:
    agent_contexts = data[first_key][0]  # Biography/Math format
elif isinstance(data[first_key], list):
    agent_contexts = data[first_key]
else:
    print(f"  Value preview: {str(data[first_key])[:200]}")
    print("\nUnexpected data structure. Cannot analyze.")
    sys.exit(1)

print(f"  Agent contexts type: {type(agent_contexts)}")
print(f"  Number of agents: {len(agent_contexts)}")

# Show response lengths for Agent 1
agent1_context = agent_contexts[0]
print(f'\nAgent 1 context history ({len(agent1_context)} messages total):')
print("=" * 80)

total_chars = 0
total_tokens_approx = 0

for i, msg in enumerate(agent1_context):
    # Handle different message formats
    if isinstance(msg, dict) and 'content' in msg:
        content = msg['content']
        role = msg.get('role', 'unknown')
    else:
        print(f"\n[Message {i}] Unexpected format: {type(msg)}")
        continue

    content_len = len(content)
    token_approx = content_len // 4  # Rough estimate: 1 token ≈ 4 chars
    total_chars += content_len
    total_tokens_approx += token_approx

    print(f'\n[Message {i}] Role: {role}')
    print(f'  Length: {content_len} chars (~{token_approx} tokens)')

    # Show preview for first 10 messages
    if i < 10:
        preview = content[:200].replace('\n', ' ')
        print(f'  Preview: {preview}...')

print("\n" + "=" * 80)
print(f"TOTAL CONTEXT: {total_chars} chars (~{total_tokens_approx} tokens)")
print("=" * 80)

# Show breakdown by role
user_msgs = [m for m in agent1_context if m['role'] == 'user']
assistant_msgs = [m for m in agent1_context if m['role'] == 'assistant']

user_tokens = sum(len(m['content']) // 4 for m in user_msgs)
assistant_tokens = sum(len(m['content']) // 4 for m in assistant_msgs)

print(f"\nBreakdown:")
print(f"  User messages: {len(user_msgs)} messages, ~{user_tokens} tokens")
print(f"  Assistant messages: {len(assistant_msgs)} messages, ~{assistant_tokens} tokens")
