# LRGA - Low Rank Global Attention

This repository contains an implementation to the paper: "From Graph Low-Rank Global Attention to 2-FWL Approximation".

LRGA is a Low Rank Global Attention module for Graph Neural Networks. Our method implements an efficient and scalable global attention, taking advantage of the efficiency of low rank matrix-vector multiplication. From a practical point of view, augmenting existing GNN layers with LRGA produces state of the art results.

For more details:

paper: https://arxiv.org/abs/2006.07846.

video: https://www.youtube.com/watch?v=ZDAyG48B-LA&t=2s.


## Installation Requirmenets
Follow the installation instructions specified here - https://github.com/snap-stanford/ogb.

## Usage
To run the code over examples from the OGB benchmark, go to the linkpropprod directory
and follow the README file in that folder.

## Citation
@misc{puny2020graph,
    title={From Graph Low-Rank Global Attention to 2-FWL Approximation},
    author={Omri Puny and Heli Ben-Hamu and Yaron Lipman},
    year={2020},
    eprint={2006.07846},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}