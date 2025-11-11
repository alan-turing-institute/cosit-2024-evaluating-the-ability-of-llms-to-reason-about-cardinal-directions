# Evaluating the Ability of Large Language Models to Reason about Cardinal Directions, Revisited

Anthony G Cohn and Robert E Blackwell

April 2024, Updated June 2025

## Introduction

This repository is an online appendix and companion to our work on
evaluating the ability of large language models to reason about
cardinal directions [1,2].

The `data` subdirectory contains the questions, answers, prompts and
LLM responses for our _small_ and _large_ experiments. Files are in
[JSONL](https://jsonlines.org/) format.

The `notebooks` subdirectory contains Jupyter notebooks and associated
Python code for processing the answers and plotting the figures used
in [2]. The notebooks also contain supplementary analyses.

Note that some of the `answers.jsonl` files are large and so we
compress them with `xz`. We have provided a bash script in the `bin`
directory to recursively find and uncompress the answer files prior to
running the Jupyter notebook.

All the QR2025 experiments were conducted using
[Golem](https://github.com/RobBlackwell/golem).


## References

[1] Anthony G Cohn and Robert E Blackwell. Evaluating the Ability of Large
Language Models to Reason About Cardinal Directions (Short Paper). In
16th International Conference on Spatial Information Theory (COSIT
2024). Leibniz International Proceedings in Informatics (LIPIcs),
Volume 315, pp. 28:1-28:9, Schloss Dagstuhl – Leibniz-Zentrum für
Informatik (2024) [COSIT 2024 short
paper](https://doi.org/10.4230/LIPIcs.COSIT.2024.28).

[2] Anthony G Cohn and Robert E Blackwell. Evaluating the Ability of
Large Language Models to Reason About Cardinal Directions,
Revisited. QR 2025 : 38th International Workshop on Qualitative
Reasoning at IJCAI. [QR 2025 paper](https://arxiv.org/abs/2507.12059).
