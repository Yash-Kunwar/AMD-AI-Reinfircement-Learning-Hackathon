# Minesweeper-RL

[![Status: WIP](https://img.shields.io/badge/Status-Work_In_Progress-orange.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-AMD_MI300x-red.svg)]()

Training a Large Language Model to natively play Minesweeper without a UI, without wrappers, and without hallucinating. Just raw JSON game states in, and JSON actions out. 

Built by Yash Kunwar, Person1, and Person2 during the AMD AI Reinforcement Learning Hackathon at IIT Delhi (Track 2: Gaming the Models).

---

### The Scope
LLMs are notoriously bad at 2D spatial reasoning. Asking a 4-billion parameter model to look at a highly compressed string representation of a 6x6 grid, deduce mine probabilities, and output a strict `{"type": "reveal", "row": x, "col": y}` action is a mathematically hostile environment.

This repository contains our pipeline for forcing spatial logic into the model using Group Relative Policy Optimization (GRPO). We don't want the model to guess; we want it to sweep.

### Tech Stack
* **Model:** Qwen3-4B
* **Framework:** Unsloth (for the massive VRAM savings and 2x training speedup)
* **RL Method:** GRPO via `trl`
* **Hardware:** AMD MI300x (ROCm)

---

### Visuals & Metrics

**Training GRPO with Unsloth (20 Steps, 1 Epoch)**
*Slothing our way through 4 generations per state.* | Step | Training Loss | Total Reward | KL Div | JSON Reward | Gameplay Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 0.000000 | -110.00 | 0.0006 | -60.00 | -50.00 |
| **2** | 0.000000 | -102.84 | 0.0006 | -56.56 | -46.28 |
| **...**| ... | ... | ... | ... | ... |
| **19** | 0.000900 | -65.18 | 0.2222 | -19.37 | -10.81 |
| **20** | 0.000500 | -88.53 | 0.1244 | 1.68 | 2.84 |

**Anti-Verbosity In Action**
Here is the model playing natively without `<think>` wrappers or UI crutches. We strictly enforced this JSON-to-JSON pipeline:

![JSON Input and Output Format](<images/Screenshot 2026-02-14 151447.png>)
*Caption: No reasoning, no rambling. Just pure state-action mapping.*

---

### The Approach: Carrots, Sticks, and Baby Steps

We custom-built the environment and reward functions to violently discourage the typical LLM urge to ramble. 

1. **The Anti-Verbosity Penalty:** We applied a massive `-10` penalty if the model generated a `<think>` tag or didn't immediately start its response with a `{`. Invalid JSON formatting results in an instant `-50`.
2. **Adjacency Heuristics vs. YOLO Guessing:** The model gets `+20` points for logically deducing a safe cell adjacent to a revealed cell, but gets penalized `-5` if it blindly guesses a random unrevealed cell mid-game.
3. **State Compression:** We compressed the 2D array into a tight string grid inside the JSON to save hundreds of tokens per prompt.
4. **Curriculum Learning:** You don't teach calculus before addition. We started the model on tiny sandboxes (e.g., 4x4) to let it learn basic adjacency rules without the noise of a massive board. Once it stopped eating mines for breakfast, we graduated it to variable M x N grids:

   ```text
      Phase 1          Phase 2            Phase 3
      [ 4 x 4 ]  ───>  [ 6 x 6 ]  ───>  [ M x N ]
     (Adjacency)       (Tactics)     (Generalization)
