import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
from copy import deepcopy
import math
import multiprocessing
from torchvision.io import read_image


# ————————————————————————
# 1. Paths and parameters
# ————————————————————————

data_dir = '../hist/multi_netw_dataset/'
labels_file = os.path.join(data_dir, 'labels_all.csv')
gpu = 2
device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 25
EPOCHS = 150
LR = 1e-5
IMG_SIZE = 224
NUM_CLASSES = 2
RANDOM_STATE_BASE = 42  # base random state for each run
NUM_RUNS_PER_CONFIG = 5  # number of times to repeat each experiment
INPUT_SIZES = [1, 2, 3, 4]  # number of inputs (can include 4)

# Directories for different magnifications
all_mag_dirs = {
    0: os.path.join(data_dir, 'high_mag_1024'),
    1: os.path.join(data_dir, 'high_mag_2048'),
    2: os.path.join(data_dir, 'high_mag_512'),
    3: os.path.join(data_dir, 'high_mag_256')
}

# Augmentations
geometric_transforms = T.Compose([
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=15)
])

color_transforms_list = [
    T.Compose([T.ColorJitter(0.3, 0.3, 0.3, 0.1), T.GaussianBlur(3), T.RandomErasing(p=0.3)]),
    T.Compose([T.ColorJitter(0.4, 0.4, 0.4, 0.2), T.GaussianBlur(3), T.RandomErasing(p=0.4)]),
    T.Compose([T.ColorJitter(0.35, 0.35, 0.35, 0.15), T.GaussianBlur(3), T.RandomErasing(p=0.35)]),
    T.Compose([T.ColorJitter(0.38, 0.38, 0.38, 0.18), T.GaussianBlur(3), T.RandomErasing(p=0.38)])
]

val_transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE))])

# Results storage
results = []


# ————————————————————————
# 2. Dataset
# ————————————————————————

class MultiInputDataset(torch.utils.data.Dataset):
    """Custom dataset to handle multiple image inputs per sample (e.g., multi-scale histology patches).

    Each sample consists of several images (from different magnifications) and one label.
    Supports training-time augmentations applied independently per input stream.
    """

    def __init__(self, labels_df, mag_dirs, color_transforms=None, geometric_tf=None, val_tf=None, is_train=True):
        """Initializes the dataset.

        Args:
            labels_df (pd.DataFrame): DataFrame containing filenames and labels.
            mag_dirs (dict[int, str]): Mapping from input index to directory path.
            color_transforms (list[torchvision.transforms.Compose], optional): List of color transforms per input.
            geometric_tf (torchvision.transforms.Compose, optional): Geometric transforms applied to all inputs.
            val_tf (torchvision.transforms.Compose, optional): Transform for validation/test mode.
            is_train (bool): Whether in training mode (applies augmentation).
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.mag_dirs = mag_dirs
        self.color_transforms = color_transforms
        self.geometric_tf = geometric_tf
        self.val_tf = val_tf
        self.is_train = is_train

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """Loads and returns a sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: List of tensors (one per input) and label tensor.
        """
        row = self.labels_df.iloc[idx]
        img_name = row['filename']
        inputs = []
        for i, dir_path in self.mag_dirs.items():
            path = os.path.join(dir_path, f"{img_name}.png")
            try:
                img = read_image(path).float() / 255.0
                img = T.Resize((IMG_SIZE, IMG_SIZE))(img)
                if self.is_train:
                    seed = torch.randint(0, 2**32, ()).item()
                    torch.manual_seed(seed)
                    img = self.geometric_tf(img)
                    if self.color_transforms and i < len(self.color_transforms):
                        img = self.color_transforms[i](img)
                    img = img.clamp(0.0, 1.0)
                inputs.append(img)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                return self.__getitem__((idx + 1) % len(self))
        label = torch.tensor(row['label'], dtype=torch.long)
        return inputs, label


def collate_fn(batch):
    """Custom collate function to handle list of image tensors per sample.

    Groups inputs by modality (input stream) and stacks them into batches.

    Args:
        batch (list[tuple]): List of (inputs, label) tuples.

    Returns:
        tuple: Batched inputs as list of tensors (one per input), and stacked labels.
    """
    inputs = [[] for _ in range(len(batch[0][0]))]
    labels = []
    for input_list, label in batch:
        for i, img in enumerate(input_list):
            inputs[i].append(img.unsqueeze(0))
        labels.append(label)
    stacked_inputs = [torch.cat(inp, dim=0) for inp in inputs]
    labels = torch.stack(labels)
    return stacked_inputs, labels


# ————————————————————————
# 3. Model
# ————————————————————————

class MultiInputModule(nn.Module):
    """Multi-input model using EfficientNet-B0 backbones and cross-attention fusion.

    Processes multiple image inputs (e.g., different magnifications), extracts features,
    applies attention-based fusion, and classifies the combined representation.
    """

    def __init__(self, num_inputs=2, num_classes=2, embed_dim=128, dropout=0.5):
        """Initializes the multi-input model.

        Args:
            num_inputs (int): Number of input images per sample.
            num_classes (int): Number of output classes.
            embed_dim (int): Dimension of projected features before attention.
            dropout (float): Dropout rate in attention, FFN, and classifier.
        """
        super().__init__()
        self.num_inputs = num_inputs
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = efficientnet_b0(weights=weights)
        self.feature_extractor = base_model.features
        self.feature_dim = 1280  # Output of EfficientNet-B0 features
        self.proj = nn.Linear(self.feature_dim, embed_dim)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads=8, dropout=dropout, batch_first=True)
            for _ in range(num_inputs)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * num_inputs, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs):
        """Forward pass through the network.

        Args:
            inputs (list[torch.Tensor]): List of image tensors of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        features_proj = []
        for x in inputs:
            feat_vec = self.global_pool(self.feature_extractor(x)).squeeze(-1).squeeze(-1)
            feat_proj = self.proj(feat_vec)
            features_proj.append(feat_proj)
        feat_seq = torch.stack(features_proj, dim=1)  # (B, N, D)
        attended_features = []
        for i in range(self.num_inputs):
            q = feat_seq[:, i:i+1, :]  # Query from i-th input
            kv = feat_seq  # Keys and values from all inputs
            attended, _ = self.attention_layers[i](query=q, key=kv, value=kv)
            attended = self.norm1(attended.squeeze(1) + features_proj[i])  # Residual + norm
            attended = self.norm2(attended + self.ffn(attended))  # FFN residual
            attended_features.append(attended)
        combined = torch.cat(attended_features, dim=1)  # Concatenate attended features
        return self.classifier(combined)


# ————————————————————————
# 4. Main experiment loop: evaluation on test set
# ————————————————————————

num_workers = math.ceil(multiprocessing.cpu_count() * 2 / 3)

df_labels = pd.read_csv(labels_file)
df_labels['num_id'] = df_labels['filename'].apply(lambda x: x.split('_')[0])

unique_ids = df_labels['num_id'].drop_duplicates().reset_index(drop=True)
labels_for_stratify = df_labels.drop_duplicates('num_id').set_index('num_id')['label']

results = []  # Reset results list

for num_inputs in INPUT_SIZES:
    print(f"\n{'='*60}")
    print(f"🚀 EXPERIMENT: {num_inputs} input(s)")
    print(f"{'='*60}")

    test_f1_scores = []  # Store F1 scores on test set

    for run in range(NUM_RUNS_PER_CONFIG):
        rs = RANDOM_STATE_BASE + run
        print(f"\n🔁 Run {run+1}/{NUM_RUNS_PER_CONFIG} (random_state={rs})")

        # —— Stratified data split ——
        train_val_ids, test_ids = train_test_split(
            unique_ids, test_size=0.2,
            stratify=labels_for_stratify[unique_ids],
            random_state=rs
        )
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=0.25,
            stratify=labels_for_stratify[train_val_ids],
            random_state=rs
        )

        df_labels['split'] = df_labels['num_id'].map({
            **{k: 'train' for k in train_ids},
            **{k: 'val' for k in val_ids},
            **{k: 'test' for k in test_ids}
        })

        train_df = df_labels[df_labels['split'] == 'train'].reset_index(drop=True)
        val_df = df_labels[df_labels['split'] == 'val'].reset_index(drop=True)
        test_df = df_labels[df_labels['split'] == 'test'].reset_index(drop=True)

        # —— Select directories for current number of inputs ——
        mag_dirs = {i: all_mag_dirs[i] for i in range(num_inputs)}

        # —— Datasets ——
        train_dataset = MultiInputDataset(
            train_df, mag_dirs,
            color_transforms=color_transforms_list[:num_inputs],
            geometric_tf=geometric_transforms,
            is_train=True
        )
        val_dataset = MultiInputDataset(val_df, mag_dirs, val_tf=val_transform, is_train=False)
        test_dataset = MultiInputDataset(test_df, mag_dirs, val_tf=val_transform, is_train=False)

        # —— DataLoaders ——
        class_weights_arr = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
        sample_weights = [class_weights_arr[label] for label in train_df['label']]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                  collate_fn=collate_fn, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate_fn, num_workers=num_workers)

        # —— Model ——
        model = MultiInputModule(num_inputs=num_inputs, embed_dim=128, num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights_arr, dtype=torch.float).to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=3e-4, steps_per_epoch=len(train_loader), epochs=EPOCHS
        )

        # —— Training with early stopping based on val_f1 ——
        BEST_VAL_F1 = 0.0
        COUNTER = 0
        BEST_MODEL_PATH = os.path.join(data_dir, f"temp_best_model_run_{run}_inputs_{num_inputs}.pth")

        for epoch in range(EPOCHS):
            model.train()
            for inputs, labels in train_loader:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            # Validation for early stopping
            model.eval()
            all_labels, all_preds = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = [x.to(device) for x in inputs]
                    labels = labels.to(device)
                    preds = model(inputs).max(1)[1]
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
            val_f1 = f1_score(all_labels, all_preds, average='macro')

            if val_f1 > BEST_VAL_F1:
                BEST_VAL_F1 = val_f1
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                COUNTER = 0
            else:
                COUNTER += 1

            if COUNTER >= 15:
                break

        # —— Test the best model ——
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        model.eval()

        all_test_labels = []
        all_test_preds = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_test_labels.extend(labels.cpu().numpy())
                all_test_preds.extend(preds.cpu().numpy())

        test_f1 = f1_score(all_test_labels, all_test_preds, average='macro')
        test_f1_scores.append(test_f1)
        print(f"✅ Test Macro-F1: {test_f1:.4f}")

        # Remove temporary model file
        if os.path.exists(BEST_MODEL_PATH):
            os.remove(BEST_MODEL_PATH)

    # —— Collect statistics across runs ——
    mean_test_f1 = np.mean(test_f1_scores)
    std_test_f1 = np.std(test_f1_scores)
    results.append({'num_inputs': num_inputs, 'mean_test_f1': mean_test_f1, 'std_test_f1': std_test_f1})
    print(f"\n📊 FINAL ({num_inputs} inputs): Test F1 = {mean_test_f1:.4f} ± {std_test_f1:.4f}")

# ————————————————————————
# 5. Save results
# ————————————————————————

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(data_dir, 'experiment_results_summary_TEST.csv'), index=False)

print("\n✅ All experiments completed.")
print("📋 Test set results saved to 'experiment_results_summary_TEST.csv'")
print(results_df)
