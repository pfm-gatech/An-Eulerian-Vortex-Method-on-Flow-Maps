# [TOG (SIGGRAPH Asia 2024)] An Eulerian Vortex Method on Flow Maps

by [Sinan Wang](https://sinanw.com), [Yitong Deng](https://yitongdeng.github.io/), [Molin Deng](https://molin7.vercel.app), [Hong-Xing Yu](https://kovenyu.com/), [Junwei Zhou](https://zjw49246.github.io/website/), [Duowen Chen](https://cdwj.github.io), [Taku Komura](https://hku-cg.github.io/author/taku-komura/), [Jiajun Wu](https://jiajunwu.com/), and [Bo Zhu](https://faculty.cc.gatech.edu/~bozhu/)

Our paper and video results can be found at our [project website](https://evm.sinanw.com/).

This work has been awarded the **[Replicability Stamp](http://www.replicabilitystamp.org#https-github-com-pfm-gatech-an-eulerian-vortex-method-on-flow-maps-git)**.
[![](https://www.replicabilitystamp.org/logo/Reproducibility-small.png)](http://www.replicabilitystamp.org#https-github-com-pfm-gatech-an-eulerian-vortex-method-on-flow-maps-git)

## Installation
Our code is tested on Windows 11 with CUDA 12.3, Python 3.10.9, and Taichi 1.6.0.

To set up the environment, first create a conda environment:

```bash
conda create -n "evm_env" python=3.10.9 ipython
conda activate evm_env
```

Then, install the requirements with:

```bash
pip install -r requirements.txt
```

## Simulation
For reproducing the same result in the paper, execute:

```bash
python run_paper.py
```

The expected runtime ranges from 30 to 60 minutes, depending on your machine's performance.

An improved version by storing the flow map quantities with the same locations as vorticity can be obtained by running:

```bash
python run_improved.py
```

The improved version enhances the simulation stability and the vorticity preservation ability, e.g., it leads to one more leap in 3D leapfrog.

Hyperparameters can be tuned by changing the values in the file `hyperparameters.py`.

## Visualization
The results will be stored in `logs/[exp_name]/vtks`. We recommend using ParaView to load these `.vti` files as a sequence and visualize them by selecting **Volume** in the Representation drop-down menu.

## Bibliography
If you find our paper or code helpful, consider citing:

```bibtex
@article{wang2024eulerian,
  title={An Eulerian Vortex Method on Flow Maps},
  author={Wang, Sinan and Deng, Yitong and Deng, Molin and Yu, Hong-Xing and Zhou, Junwei and Chen, Duowen and Komura, Taku and Wu, Jiajun and Zhu, Bo},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--13},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```
