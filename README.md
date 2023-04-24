
# DACN: Malware Classification Based on Dynamic Analysis and Capsule Networks 

![Architecture of DACN](https://github.com/To2rk/DACN/blob/main/imgs/Architecture_of_DACN.png)

![Improved Capsule Network Model in DACN](https://github.com/To2rk/DACN/blob/main/imgs/Improved_Capsule_Network.png)

## Quickstart

1. Clone this repository via

```bash
git clone https://github.com/To2rk/DACN.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Prepare datasets of malware image data

- DACN dataset: [Download](https://github.com/To2rk/DACN/blob/main/imgs/DACN_dataset.tar.gz)

> A malware image dataset composed of dynamic behavioral features (API calls, DLL loading, registry modifications).

4. Then run

```bash
cd DACN/
python train.py
```

## Citation

If you use our dataset or find this research useful, please consider citing:

- **GB/T 7714**

```
Zou B, Cao C, Wang L, et al. DACN: Malware Classification Based on Dynamic Analysis and Capsule Networks[C]//International Conference on Frontiers in Cyber Security (FCS). 2021: 3-13.
```

- **bib**

```
@inproceedings{Zou2021DACN,
    author="Zou, Binghui and Cao, Chunjie and Wang, Longjuan and Tao, Fangjian",
    title="DACN: Malware Classification Based on Dynamic Analysis and Capsule Networks",
    booktitle="International Conference on Frontiers in Cyber Security (FCS)",
    year="2021",
    publisher="Springer Singapore",
    pages="3--13",
    doi="https://doi.org/10.1007/978-981-19-0523-0_1"
}
```
