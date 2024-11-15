# ROCODE

This repository contains the source code and data for the paper "[ROCODE: Integrating Backtracking Mechanism and Program Analysis in Large Language Models for Code Generation]( https://arxiv.org/abs/2411.07112)."

## Data Sources
Our approach utilizes public datasets such as [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp), and [CodeForces2305](https://github.com/YihongDong/CDD-TED4LLMs/tree/main/CodeForces2305), and employs Large Language Models such as [CodeLlama](https://huggingface.co/codellama), [CodeGen](https://github.com/salesforce/CodeGen), and [Llama](https://huggingface.co/meta-llama).



## Installation

Clone this repository and install all dependencies.

```bash
git clone git@github.com:jiangxxxue/ROCODE.git

cd ROCODE

conda env create -f environment.yml
```


## Usage
Run our approach and evaluate it by executing the `run.sh` script or the `run_codeforces.sh` script.

```bash
# Run ROCODE on HumanEval and MBPP datasets
sh run.sh

# Run ROCODE on CodeForces2305 dataset
run_codeforces.sh
```

- `main.py`: Executes our ROCODE approach.
- `evaluate_generated_code.py`: Calculates metrics such as PassRate, AvgPassRate, and Compiler Correctness Percentage.


