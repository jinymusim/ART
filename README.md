# Fine-tuning Multi-modal LLMs with ART: Art-based Reinforcement Training

This code implements and tests ART (Art-based Reinforcement Training), a method for fine-tuning images. The resulting ART images make LLMs both faster and better for several well-established test tasks.
The work thus establishes a new fundamental approach in the general-purpose AI fine-tuning. 

## Requirements

- Linux with an NVIDIA GPU
- CUDA 12.9 capable driver
- Python ≥3.13
- Conda (recommended) or another Python environment manager

## Installation

The recommended way is to use `uv`, because it installs the correct CUDA 12.9 wheels for PyTorch and vLLM automatically.

```bash
# 1. Create a fresh conda environment
conda create -n art python=3.13 -y
conda activate art

# 2. Install uv
pip install uv

# 3. Install repository
uv sync
```

## How to run

baseline
```bash
uv run run-experiment --config configs/baseline.yaml
```

learned image
```bash
uv run run-experiment --config configs/learned_image.yaml
```


## Attribution and citation

Redistributions must retain the copyright notice, license text, and disclaimer as required by the BSD 3-Clause License.

### Code citation

The documented open-source code is in preparation. 

### Academic citation

If you use this software in academic work, please also cite the paper:

M Chudoba, S Alyaev, P Galuscakova, T Wiktorski (2026)
**Fine-tuning Multi-modal LLMs with ART: Art-based Reinforcement Training**
[arXiv preprint arXiv:2606.11854](https://arxiv.org/abs/2606.11854)

```tex
@misc{chudoba2026finetuningmultimodalllmsart,
      title={Fine-tuning Multi-modal LLMs with ART: Art-based Reinforcement Training}, 
      author={Michal Chudoba and Sergey Alyaev and Petra Galuscakova and Tomasz Wiktorski},
      year={2026},
      eprint={2606.11854},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.11854}, 
}
```
