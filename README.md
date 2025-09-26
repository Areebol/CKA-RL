<h1 align="center">
     <br>Continual Knowledge Adaptation for Reinforcement Learning
<p align="center">
    <a href="https://openreview.net/pdf?id=iCYbIaGKSR">
        <img alt="Static Badge" src="https://img.shields.io/badge/Paper-NeurIPS-red">
    </a>
</p>

<h4 align="center"></a>
     
>[Jinwu Hu](https://scholar.google.com/citations?user=XmqjPi0AAAAJ&hl=en), ZiHao Lian, [Zhiqun Wen](https://scholar.google.com/citations?hl=en&user=w_O5JUYAAAAJ), ChenghaoLi, [Guohao Chen](https://scholar.google.com/citations?user=HZbzdNEAAAAJ&hl=en&oi=ao), Xutao Wen, [Bin Xiao](https://faculty.cqupt.edu.cn/xiaobin/zh_CN/index.htm), [Mingkui Tan](https://tanmingkui.github.io/)\
<sub>South China University of Technology, Pazhou Laboratory</sub>

<p align="center">
  <img src="./assets/CKA_RL.png" alt="CKA-RL" width="700" align="center">
</p>

## 📰 News
- *2025-09-18*: CKA-RL is accepted by NeurIPS2025.

## ⚡ Quick Start 
The project provides two separate requirement files for different experiment groups:
- `experiments/atari/requirements.txt` -- dependencies for Atari experiments
- `experiments/meta-world/requirements.txt` –- dependencies for Meta-World experiments
> 💡 You only need to install the environment for the experiments you plan to run.
```bash
# Clone the repository
git clone https://github.com/Fhujinwu/CKA-RL.git
cd CKA-RL

# === Setup for Atari experiments ===
conda create -n cka-rl-atari python=3.10 -y
conda activate cka-rl-atari
pip install -r experiments/atari/requirements.txt

# === Setup for Meta-World experiments ===
conda create -n cka-rl-meta python=3.10 -y
conda activate cka-rl-meta
pip install -r experiments/meta-world/requirements.txt
```

## 🏋️ Training
coming soon

## ⚖️ Evaluation
coming soon

## 💬 Citation
We gratefully acknowledge the following open-source contributions:  

- [CompoNet](https://github.com/mikelma/componet)  
- [Loss of Plasticity](https://github.com/shibhansh/loss-of-plasticity)  
- [Mask-LRL](https://github.com/dlpbc/mask-lrl)  

If you find our work useful, please consider giving our repository a 🌟 and citing our paper: 

```text

```

## ⭐ Star History
![Star History Chart](https://api.star-history.com/svg?repos=Fhujinwu/CKA-RL&type=Date)
