# RFD-DETR

This is the official implementation of the paper:
**"Railway fastener defect detection using RFD-DETR: a lightweight real-time transformer-based approach"**

---

## Getting Started

### Installation

The code is trained and validated with `python=3.10.8`, `pytorch=2.1.2`, `cuda=12.1`. Other versions might be available as well.

1. **Clone the repository**

   ```bash
   git clone https://github.com/sinclair2577/RFD-DETR.git
   cd RFD-DETR
   ```

2. **Install PyTorch and torchvision**  
   Follow the instructions on [PyTorch official website](https://pytorch.org/get-started/).  
   Example:

   ```bash
   conda install -c pytorch pytorch torchvision
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Format

This project uses the YOLOv8 dataset format and supports custom datasets.  
You can refer to the [Ultralytics official documentation](https://docs.ultralytics.com/usage/simple-utilities/) for more details.

Example directory structure:

```
dataset/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  └── data.yaml
```

- `images/`: stores images for training/validation
- `labels/`: stores corresponding label files (YOLO format, `.txt`)
- `data.yaml`: dataset configuration file

Each label file (one object per line):

---

## Training

Use the following command to start training (edit parameters as needed):

```bash
python train.py
```

> **Note:**  
> If you want to use the CLI, refer to the [Ultralytics CLI documentation](https://docs.ultralytics.com/usage/cli/).

---

## Validation

Run the following command for validation:

```bash
python val.py
```
