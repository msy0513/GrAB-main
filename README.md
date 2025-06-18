# GrAB-mian  
**Grab All the Anomalies**: Graph-Augmented Transformer Framework for Blockchain Anomaly Transaction Detection  

This is a Python implementation of **GrAB**, as described in the following paper:  
> *Grab All the Anomalies: Graph-Augmented Transformer Framework for Blockchain Anomaly Transaction Detection*

---
## 📦 Requirements  

All models are implemented with the following software configuration:

- Python 3.9  
- Torch 1.12.1  
- torch-scatter 2.0.9  
- DGL 2.2.1+cu121  
- CUDA 11.3  
- scikit-learn 1.2.0  
- numpy 1.26.2  
- tqdm 4.66.1  

---

## 📁 Data  

We utilize two datasets to evaluate the model's performance:
1. **Elliptic dataset**  
   Released by the MIT-IBM Watson AI Lab, it is the largest publicly available labeled transaction dataset in cryptocurrency research.  
   📄 [Download on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

2. **Elliptic++ dataset**  
   An enhanced version of the original Elliptic dataset, adding an additional 17-dimensional feature for each node.  
   📄 [Download on Google Drive](https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l)

---
## 🧱 Model architecture 
```bash
GrAB
├─ FeatureEncoder
│   └─ TripleChannelNodeEncoder
│       ├─ Sequential (Channel 1)
│       ├─ Sequential (Channel 2)
│       ├─ Sequential (Channel 3)
│       └─ Sequential (Channel 4)
├─ RRWPLinearNodeEncoder
│   └─ Linear
├─ RRWPLinearEdgeEncoder
│   └─ Linear
├─ TradeFormer Layers
│   ├─ TradeFormerLayer (Layer 1)
│   │   ├─ MultiHeadNodePairAttLayer
│   │   ├─ Dropout ×2
│   │   ├─ Linear ×2
│   │   ├─ BatchNorm1d ×2
│   │   └─ FFN: Linear → ReLU → Dropout → Linear → BatchNorm1d
│   └─ TradeFormerLayer (Layer 2)
│       ├─ Same structure as above
├─ Node Classification Head: DIY_NodeHead
│   └─ Sequential
│       ├─ Linear
│       ├─ ReLU
│       ├─ LayerNorm
│       ├─ Dropout
│       └─ Linear (output)
```

## 🛠️ Usage  

📌 One-Step End-to-End Pipeline
- GrAB is an end-to-end framework designed for processing large-scale blockchain transactions.
With a single command, it completes the full pipeline — from graph construction to model training and evaluation.
- GrAB will determine whether you need a composition according to whether you have processed the composition data. The data processing code is in the file: \dataset\minidataset.py.


```bash
python .\GrAB-main\grit\task_finetune.py
```
