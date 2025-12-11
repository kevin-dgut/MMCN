# test/train.py
import argparse
import os
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from test.data_loader import DataLoaderHelper
from test.utils import preprocess_data if False else None  # optional, preprocess defined in original script
from test.model import SCG_ViT
from test.utils import compute_top10_rois_combined

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0
    for fc1, fc2, sc1, sc2, labels in train_loader:
        fc1 = fc1.to(device); fc2 = fc2.to(device); sc1 = sc1.to(device); sc2 = sc2.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(fc1, fc2, sc1, sc2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, running_correct / total


def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for fc1, fc2, sc1, sc2, labels in loader:
            fc1 = fc1.to(device); fc2 = fc2.to(device); sc1 = sc1.to(device); sc2 = sc2.to(device)
            labels = labels.to(device)
            outputs = model(fc1, fc2, sc1, sc2)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-config', type=str, default=None)
    parser.add_argument('--num_rois', type=int, default=116)
    parser.add_argument('--embed_dim', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./model/')
    args = parser.parse_args()



if __name__ == '__main__':
    main()