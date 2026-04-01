#-*- coding: utf-8 -*-
"""
@author: Md Rezwanul Haque
"""
#----------------------------------------------------------------
# imports
#----------------------------------------------------------------
import torch
from tqdm import tqdm
from .utils import ShapeAdapter
adapter = ShapeAdapter()

def train_epoch(
    net, train_loader, loss_fn, optimizer, lr_scheduler, device, 
    current_epoch, total_epochs, tqdm_able, cross_infer=False
):
    """One training epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False, unit="batch", disable=tqdm_able
    ) as pbar:
        for x, y, mask in pbar:
            # print(f"x shape: {x.shape}, y shape: {y.shape}") # lmvb: torch.Size([16, 2086, 264]), y shape: torch.Size([16])
            #                                                  # dvlog: x shape: torch.Size([16, 1443, 161]), y shape: torch.Size([16])

            ## doing cross infer
            if cross_infer:
                x = adapter(x)
           
            x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
            y_pred = net(x, mask)

            # loss = loss_fn(y_pred, y.to(torch.float32))
            loss = loss_fn(y_pred, y.to(torch.float32), net)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]

            # binary classification with only one output neuron
            pred = (y_pred > 0.).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "acc": correct_count / sample_count,
                "lr": optimizer.param_groups[0]['lr']
            })

    if lr_scheduler is not None and isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_scheduler.step(running_loss / sample_count)
    elif lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(
    net, val_loader, loss_fn, device, tqdm_able, cross_infer=False
):
    """Test the model on the validation / test set.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:

                ## doing cross infer
                if cross_infer:
                    x = adapter(x)
             
                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)

                # loss = loss_fn(y_pred, y.to(torch.float32))
                loss = loss_fn(y_pred, y.to(torch.float32), net)

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]

                # binary classification with only one output neuron
                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )
    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }

def val(
    net, val_loader, loss_fn, device, tqdm_able, msg='additional metrics', cross_infer=False
):
    """Test the model on the validation / test set and calculate additional metrics.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch", disable=tqdm_able
        ) as pbar:
            for x, y, mask in pbar:

                ## doing cross infer
                if cross_infer:
                    # print("x cross infer")
                    x = adapter(x)

                x, y, mask = x.to(device), y.to(device).unsqueeze(1), mask.to(device)
                y_pred = net(x, mask)

                loss = loss_fn(y_pred, y.to(torch.float32), net)

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]

                pred = (y_pred > 0.).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                true_labels.extend(y.cpu().numpy().flatten().tolist())
                predicted_labels.extend(pred.cpu().numpy().flatten().tolist())

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )

    num_samples = len(true_labels)
    if num_samples == 0:
        return {
            "loss": l, "acc": accuracy,
            "precision": precision, "recall": recall, "f1": f1_score,
            "weighted_accuracy": 0.0, "unweighted_accuracy": 0.0,
            "weighted_precision": 0.0, "unweighted_precision": 0.0,
            "weighted_recall": 0.0, "unweighted_recall": 0.0,
            "weighted_f1": 0.0, "unweighted_f1": 0.0,
        }

    ##================================================================================
    # After collecting true_labels and predicted_labels
    class_labels = sorted(list(set(true_labels)))
    class_counts = {label: true_labels.count(label) for label in class_labels}
    class_tp = {label: 0 for label in class_labels}
    class_fp = {label: 0 for label in class_labels}
    class_fn = {label: 0 for label in class_labels}
    class_tn = {label: 0 for label in class_labels}

    # Calculate TP, FP, FN for both classes
    for true, pred in zip(true_labels, predicted_labels):
        if true == 1 and pred == 1:
            class_tp[1] += 1
        elif true == 0 and pred == 1:
            class_fp[1] += 1
        elif true == 1 and pred == 0:
            class_fn[1] += 1
        elif true == 0 and pred == 0:
            class_tp[0] += 1
        elif true == 1 and pred == 0:
            class_fp[0] += 1
        elif true == 0 and pred == 1:
            class_fn[0] += 1
        elif true == 0 and pred == 0:
            class_tn[0] += 1

    # Class-specific metrics
    class_precision = {
        label: class_tp[label] / (class_tp[label] + class_fp[label]) 
        if (class_tp[label] + class_fp[label]) > 0 else 0.0
        for label in class_labels
    }
    class_recall = {
        label: class_tp[label] / (class_tp[label] + class_fn[label]) 
        if (class_tp[label] + class_fn[label]) > 0 else 0.0
        for label in class_labels
    }
    class_f1 = {
        label: 2 * (class_precision[label] * class_recall[label]) / (class_precision[label] + class_recall[label])
        if (class_precision[label] + class_recall[label]) > 0 else 0.0
        for label in class_labels
    }
    class_accuracy = {
        label: (class_tp[label] + class_tn[label]) / class_counts[label] 
        if class_counts[label] > 0 else 0.0
        for label in class_labels
    }

    # Weighted and unweighted metrics
    weighted_accuracy = sum(class_accuracy[label] * class_counts[label] for label in class_labels) / num_samples
    weighted_precision = sum(class_precision[label] * class_counts[label] for label in class_labels) / num_samples
    weighted_recall = sum(class_recall[label] * class_counts[label] for label in class_labels) / num_samples
    weighted_f1 = sum(class_f1[label] * class_counts[label] for label in class_labels) / num_samples

    unweighted_accuracy = sum(class_accuracy.values()) / len(class_labels)
    unweighted_precision = sum(class_precision.values()) / len(class_labels)
    unweighted_recall = sum(class_recall.values()) / len(class_labels)
    unweighted_f1 = sum(class_f1.values()) / len(class_labels)

    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
        "weighted_accuracy": weighted_accuracy,
        "unweighted_accuracy": unweighted_accuracy,
        "weighted_precision": weighted_precision,
        "unweighted_precision": unweighted_precision,
        "weighted_recall": weighted_recall,
        "unweighted_recall": unweighted_recall,
        "weighted_f1": weighted_f1,
        "unweighted_f1": unweighted_f1,
    }