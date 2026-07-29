# ST-LLM+ : Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction
This repository provides the official Pytorch implementation of our manuscript titled "[ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction](https://ieeexplore.ieee.org/document/11005661)". This work is an extension of the original [ST-LLM](https://github.com/ChenxiLiu-HNU/ST-LLM/blob/main/ST-LLM.pdf) model. The foundational training framework is derived from the open-source codebase developed by [ChenxiLiu-HNU](https://github.com/ChenxiLiu-HNU/ST-LLM/tree/main).

## Abstract

> *Traffic prediction is a crucial component of data management systems, leveraging historical data to learn spatio-temporal dynamics for forecasting future traffic and enabling efficient decision-making and resource allocation. Despite efforts to develop increasingly complex architectures, existing traffic prediction models often struggle to generalize across diverse datasets and contexts, limiting their adaptability in real-world applications. In contrast to existing traffic prediction models, large language models (LLMs) progress mainly through parameter expansion and extensive pre-training while maintaining their fundamental structures. In this paper, we propose ST-LLM+, the graph enhanced spatio-temporal large language models for traffic prediction. Through incorporating a proximity-based adjacency matrix derived from the traffic network into the calibrated LLMs, ST-LLM+ captures complex spatio-temporal dependencies within the traffic network. The Partially Frozen Graph Attention (PFGA) module is designed to retain global dependencies learned during LLMs pre-training while modeling localized dependencies specific to the traffic domain. To reduce computational overhead, ST-LLM+ adopts the LoRA-augmented training strategy, allowing attention layers to be fine-tuned with fewer learnable parameters. Comprehensive experiments on real-world traffic datasets demonstrate that ST-LLM+ outperforms state-of-the-art models. In particular, ST-LLM+ also exhibits robust performance in both few-shot and zero-shot prediction scenarios. Additionally, our case study demonstrates that ST-LLM+ captures global and localized dependencies between stations, verifying its effectiveness for traffic prediction tasks.*

![Image](https://github.com/kethmih/ST-LLM-Plus/blob/main/assets/Architecture_Diagram.png)

## Dependencies

The Llama 3.1 / PEMS08 training path is verified with:

* Python 3.11
* PyTorch 2.6.0 + CUDA 12.4
* Transformers 4.46.3
* PEFT 0.10.0

Install the packages in `requirements.txt`, or use the existing `transllmv4`
Conda environment. The model is loaded only from the local directory below;
training does not download weights.

```bash
pip install -r requirements.txt
```

## Datasets

The new training entry uses raw PEMS08 with the following layout:

```text
data/st_data/pems08/
├── pems08.npz
└── pems08_adj.npy
```

`util.load_pems08_dataset` builds 12-step input and 12-step target windows,
adds 5-minute time-of-day and day-of-week features, and performs a chronological
60/20/20 split. Windows cannot cross split boundaries, and normalization uses
only the training segment.

The original NYCTaxi/CHBike data remains available from the project
[Google Drive](https://drive.google.com/drive/folders/1iif59LObrPu-QrpL8Y6lWeajbn_gRf7v?usp=drive_link).

## Training

The default command uses all 32 layers of the local Llama 3.1 8B Instruct
checkpoint. Traffic stations are supplied as numerical embeddings, so the
tokenizer and language-model head are intentionally not used. The first 30
layers use frozen native Llama attention; the final two layers use a hard
PEMS08 graph mask and train both their original attention weights and q/v LoRA
adapters. LayerNorm remains trainable in every layer, while every FFN remains
frozen. Set `--graph_layers U` to change the number of final graph-attention
layers. The official training defaults are LoRA `r=16`, `alpha=16`, zero LoRA
dropout, Ranger with learning rate `1e-3` and weight decay `1e-4`, and masked
MAE loss. Frozen Llama weights are stored in BF16; all trainable attention,
LoRA, LayerNorm, and traffic parameters are stored in FP32. Forward compute
uses BF16 autocast, and gradients accumulate in the FP32 trainable
parameters.

```bash
conda run -n transllmv4 python train_plus.py \
  --device cuda:0 \
  --model_path ./Meta-Llama-3.1-8B-Instruct \
  --data_dir ./data/st_data/pems08
```

One-batch end-to-end smoke test:

```bash
conda run -n transllmv4 python train_plus.py \
  --epochs 1 --batch_size 1 --grad_accum_steps 1 \
  --max_train_batches 1 --max_eval_batches 1 \
  --save_dir /tmp/stllm_llama31_smoke
```

Checkpoints contain only the trainable traffic projections, LayerNorm weights,
final-U original attention weights, and LoRA adapters—not another copy of the
entire 8B backbone. Model selection uses validation MAE; the test
set is evaluated once after loading the validation-best checkpoint.

## BibTex

If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:

```bibtex
@ARTICLE{11005661,
  author={Liu, Chenxi and Hettige, Kethmi Hirushini and Xu, Qianxiong and Long, Cheng and Xiang, Shili and Cong, Gao and Li, Ziyue and Zhao, Rui},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  title={ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction},
  year={2025},
  volume={37},
  number={8},
  pages={4846-4859},
  keywords={Time series analysis;Predictive models;Forecasting;Large language models;Adaptation models;Data models;Computational modeling;Training;Electronic mail;Attention mechanisms;Traffic prediction;large language models;spatio-temporal data},
  doi={10.1109/TKDE.2025.3570705}}
```

## Further Reading

[**Spatial-Temporal Large Language Model for Traffic Prediction**](https://arxiv.org/abs/2401.10134), in *MDM* 2024.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/ST-LLM/tree/main)

```bibtex
@inproceedings{liu2024spatial,
  title={Spatial-temporal large language model for traffic prediction},
  author={Liu, Chenxi and Yang, Sun and Xu, Qianxiong and Li, Zhishuai and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={MDM},
  year={2024}
}
```

## Contact Us

For inquiries or further assistance, contact us at [kethmihi001@e.ntu.edu.sg](mailto:kethmihi001@e.ntu.edu.sg) and [chenxi.liu@ntu.edu.sg](mailto:chenxi.liu@ntu.edu.sg), or open an issue on this repository.
