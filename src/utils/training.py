# Teacher-Student training
# import all the necessary libraries
import sys
import os
import copy
import csv
maindir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "PerspectiveStudy"))
sys.path.append(maindir_path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from .metrics import bootstrap_metrics, GradCAM, save_gradcam, get_last_conv_layer
from .reporting import format_test_metrics
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
import sys
from sklearn.metrics import f1_score, precision_score, recall_score
from .generals import progress_bar
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt

def train(model, train_loader, valid_loader, epochs, learning_rate, weight_decay, device, exp_name, model_type, grad_clip_value=5.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float('inf')
    counter = 0 # counter for early stopping
    best_model_state = model.state_dict()
    best_f1 = 0.0
    best_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    model.to(device)
    train_losses = []
    val_losses = []
    val_f1s = []
    best_epoch = 0

    if not os.path.isdir('./results'):
        os.mkdir('./results')
    logname = (f'./results/{model_type}_' + exp_name + '.csv')
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([
                'epoch',
                'train_loss',
                'train_acc',
                'train_f1',
                'train_precision',
                'train_recall',
                'val_loss',
                'val_acc',
                'val_f1',
                'val_precision',
                'val_recall',
                'best_epoch'
            ])
    
    for epoch in tqdm(range(epochs), desc=f"Epochs"):
        # -------------------- Train --------------------
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        train_all_labels = []
        train_all_preds = []
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # --- debug prints --- #
            # print image names in the current batch
            batch_image_names = train_loader.dataset.data_csv.iloc[:len(inputs)]['patient'].tolist()
            #print("Batch image names:", batch_image_names)
            #print(f"input shape: {inputs.shape}, labels shape: {labels.shape}, min values: {inputs.min()}, max values: {inputs.max()}")

            optimizer.zero_grad()
            outputs = model(inputs)
            #print(f"outputs shape: {outputs.shape}, outputs min: {outputs.min().item()}, outputs max: {outputs.max().item()}")
            #print(f"labels min: {labels.min().item()}, labels max: {labels.max().item()}")
            loss = criterion(outputs, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            running_loss += loss.item()

            # metrics (student on train)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)
            train_total += batch_size
            train_correct += (predicted == labels).sum().item()
            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(predicted.cpu().numpy())

            progress_bar(batch_idx, len(train_loader),'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss/(batch_idx+1), 100.*train_correct/train_total, train_correct, train_total))


        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_acc = train_correct / train_total * 100 if train_total > 0 else 0
        train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted') * 100.0
        train_precision = precision_score(train_all_labels, train_all_preds, average='weighted') * 100.0
        train_recall = recall_score(train_all_labels, train_all_preds, average='weighted') * 100.0

        # -------------------- Validation --------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(valid_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                progress_bar(batch_idx, len(valid_loader),'Valid Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss/(batch_idx+1), 100.*correct/total, correct, total))

        avg_val_loss = val_loss / len(valid_loader)

        # compute mean metrics
        val_acc = 100 * correct / total if total > 0 else 0
        val_f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0
        val_precision = precision_score(all_labels, all_preds, average='weighted') * 100.0
        val_recall = recall_score(all_labels, all_preds, average='weighted') * 100.0

        val_losses.append(avg_val_loss)
        val_f1s.append(val_f1)

        scheduler.step()

        # -------------------- Logging --------------------
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([
                epoch,
                avg_train_loss,
                train_acc,
                train_f1,
                train_precision,
                train_recall,
                avg_val_loss,
                val_acc,
                val_f1,
                val_precision,
                val_recall,
                best_epoch
            ])
        # -------------------- early stopping --------------------
        if epoch > 10: # there should be a minimum number of epochs before starting the counter
            if avg_val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()
                counter = 0
                best_f1 = val_f1
                best_accuracy = val_acc
                best_precision = val_precision
                best_recall = val_recall
            else:              
                counter += 1
        
        if counter >= 50:
            print("Early stopping triggered")
            break
    
    metrics = {
        'accuracy': float(best_accuracy),
        'f1': float(best_f1),
        'precision': float(best_precision),
        'recall': float(best_recall)
    }

    # create a dictionary with train_losses, val_losses and val_f1s
    plots = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1s': val_f1s
    }

    # load the best model state
    model.load_state_dict(best_model_state)        
    return model, metrics, plots
        
        

def _select_eval_inputs(batch, model_type):
    if len(batch) == 2:
        inputs, labels = batch
        return inputs, labels

    if len(batch) != 4:
        raise ValueError(f"Unexpected batch size: {len(batch)}")

    if model_type == "teacher":
        inputs, _, labels, _ = batch
        return inputs, labels
    if model_type in ["student", "student_kd"]:
        _, inputs, labels, _ = batch
        return inputs, labels

    raise ValueError(
        f"Paired batches require a teacher/student-compatible model_type. Received model_type='{model_type}'."
    )

def test(model, test_loader, device, model_type, bootstrap=False, savedir = None, n_bootstrap=10000):
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = _select_eval_inputs(batch, model_type)

            inputs, labels = inputs.to(device), labels.to(device) # bring inputs and labels to the same model device

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * correct / total if total > 0 else 0
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100.0
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100.0
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100.0

    point_metrics_pct = {
        'accuracy_overall': float(accuracy),
        'precision_weighted': float(precision),
        'recall_weighted': float(recall),
        'f1_weighted': float(f1),
    }

    cm_labels = sorted(set(all_labels) | set(all_preds))
    cm = confusion_matrix(all_labels, all_preds, labels=cm_labels)
    print(f"confusion matrix: {cm}")

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=cm_labels, yticklabels=cm_labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.savefig(os.path.join(savedir, f"{model_type}_confusion_matrix.png"), dpi=300)
        plt.close()

    # ---------------------
    # --- show gradcams ---
    # ---------------------

    melanomas_list = []
    nevi_list = []
    chrpes_list = []

    # collect samples by class
    for batch in test_loader:
        inputs, labels = _select_eval_inputs(batch, model_type)

        inputs, labels = inputs.to(device), labels.to(device)

        # append images with batch dim so we can cat later
        for i in range(len(inputs)):
            cls = labels[i].item()
            img = inputs[i].unsqueeze(0)  # (1, C, H, W)
            if cls == 2:
                melanomas_list.append(img)
            elif cls == 1:
                nevi_list.append(img)
            elif cls == 3:
                chrpes_list.append(img)

    # build batched tensors if non-empty
    melanomas = torch.cat(melanomas_list, dim=0) if melanomas_list else None
    nevi      = torch.cat(nevi_list, dim=0)      if nevi_list      else None
    chrpes    = torch.cat(chrpes_list, dim=0)    if chrpes_list    else None

    '''target_layer = get_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer)

    print("Generating GradCAMs on test set...")

    def save_gradcams_for_class(images_tensor, class_name, max_n=10):
        if images_tensor is None or images_tensor.size(0) == 0:
            print(f"No samples for class {class_name}, skipping.")
            return 0

        class_dir = os.path.join(savedir, f"gradcams/{model_type}", class_name)
        os.makedirs(class_dir, exist_ok=True)

        n_saved = 0
        for idx in range(min(max_n, images_tensor.size(0))):
            img = images_tensor[idx:idx+1].to(device)  # (1, C, H, W)

            with torch.enable_grad():
                outputs_s, cam_s = gradcam(img)

            img_id = f"{idx:05d}_{class_name}.png"
            out_path = os.path.join(class_dir, img_id)

            save_gradcam(
                image_tensor=img.cpu(),
                cam_student=cam_s.cpu(),
                out_path=out_path,
                alpha=0.4,
            )

            n_saved += 1
            if n_saved % 5 == 0:
                print(f"Saved {n_saved} Grad-CAM images for {class_name}...")

        print(f"Completed. GradCAMs for {class_name}: {n_saved} images saved in {class_dir}")
        return n_saved

    n_melanoma = save_gradcams_for_class(melanomas, "melanoma", max_n=10)
    n_nevi     = save_gradcams_for_class(nevi,      "nevi",     max_n=10)
    n_chrpe    = save_gradcams_for_class(chrpes,    "chrpe",    max_n=10)

    gradcam.remove_hooks()

    print("Done generating GradCAMs.")'''


    print(f"Test Accuracy: {accuracy:.2f}% | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    if bootstrap:
        raw_bootstrap_metrics = bootstrap_metrics(
            all_preds=all_preds,
            all_labels=all_labels,
            n_bootstrap=n_bootstrap,
            confidence_level=0.95
        )
        metrics = format_test_metrics(
            point_metrics_pct=point_metrics_pct,
            raw_bootstrap_metrics=raw_bootstrap_metrics,
        )
    else:
        metrics = format_test_metrics(point_metrics_pct=point_metrics_pct)
    return metrics


def train_knowledge_distillation(
    teacher,
    student,
    train_loader,
    valid_loader,
    epochs,
    learning_rate,
    weight_decay,
    T,
    kd_loss,
    soft_target_loss_weight,
    ce_loss_weight,
    device,
    exp_name,
):
    optimizer = optim.Adam(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    hard_loss = nn.CrossEntropyLoss()

    teacher.eval()   # Teacher in eval
    student.train()  # Student in train

    best_val_loss = float('inf')
    counter = 0  # early stopping counter
    best_student_model = student.state_dict()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    student.to(device)

    train_losses = []  # total train loss for plotting
    val_losses = []    # total val loss for plotting
    val_f1s = []       # val f1 for plotting

    best_f1 = -1.0
    best_accuracy = -1.0
    best_precision = -1.0
    best_recall = -1.0
    best_epoch = -1

    # --- logger setup ---
    if not os.path.isdir('./results'):
        os.mkdir('./results')

    logname = f'./results/Distillation_{exp_name}.csv'

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([
                'epoch',
                'train_total_loss',
                'train_kd_loss',
                'train_ce_loss',
                'train_acc',
                'train_f1',
                'train_precision',
                'train_recall',
                'val_total_loss',
                'val_kd_loss',
                'val_ce_loss',
                'val_acc',
                'val_f1',
                'val_precision',
                'val_recall',
                'best_epoch',
            ])

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # -------------------- Train --------------------
        student.train()

        train_total_loss_sum = 0.0
        train_kd_loss_sum = 0.0
        train_ce_loss_sum = 0.0

        train_correct = 0
        train_total = 0
        train_all_labels = []
        train_all_preds = []

        for batch_idx, (img_t, img_s, labels, _) in enumerate(train_loader):
            img_t = img_t.to(device)
            img_s = img_s.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(img_t)

            # Student forward
            student_logits = student(img_s)

            # KD + CE losses
            kd_l = kd_loss(student_logits, teacher_logits, T)
            ce_l = hard_loss(student_logits, labels)
            loss = soft_target_loss_weight * kd_l + ce_loss_weight * ce_l

            loss.backward()
            clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()

            # accumulate losses
            train_total_loss_sum += loss.item()
            train_kd_loss_sum += kd_l.item()
            train_ce_loss_sum += ce_l.item()

            # metrics (student on train)
            _, predicted = torch.max(student_logits.data, 1)
            batch_size = labels.size(0)
            train_total += batch_size
            train_correct += (predicted == labels).sum().item()
            train_all_labels.extend(labels.cpu().numpy())
            train_all_preds.extend(predicted.cpu().numpy())

            progress_bar(batch_idx, len(train_loader),'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss/(batch_idx+1), 100.*train_correct/train_total, train_correct, train_total))


        avg_train_total_loss = train_total_loss_sum / len(train_loader)
        avg_train_kd_loss = train_kd_loss_sum / len(train_loader)
        avg_train_ce_loss = train_ce_loss_sum / len(train_loader)

        train_losses.append(avg_train_total_loss)

        train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
        train_f1 = f1_score(train_all_labels, train_all_preds, average='weighted') * 100.0
        train_precision = precision_score(train_all_labels, train_all_preds, average='weighted') * 100.0
        train_recall = recall_score(train_all_labels, train_all_preds, average='weighted') * 100.0

        # -------------------- Validation --------------------
        student.eval()

        val_total_loss_sum = 0.0
        val_kd_loss_sum = 0.0
        val_ce_loss_sum = 0.0

        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for img_t, img_s, labels, _ in valid_loader:
                img_t = img_t.to(device)
                img_s = img_s.to(device)
                labels = labels.to(device)

                student_logits = student(img_s)
                teacher_logits = teacher(img_t)

                kd_l = kd_loss(student_logits, teacher_logits, T)
                ce_l = hard_loss(student_logits, labels)
                loss = soft_target_loss_weight * kd_l + ce_loss_weight * ce_l

                val_total_loss_sum += loss.item()
                val_kd_loss_sum += kd_l.item()
                val_ce_loss_sum += ce_l.item()

                _, predicted = torch.max(student_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_total_loss_sum / len(valid_loader)
        avg_val_kd_loss = val_kd_loss_sum / len(valid_loader)
        avg_val_ce_loss = val_ce_loss_sum / len(valid_loader)

        val_losses.append(avg_val_loss)

        val_accuracy = 100 * correct / total if total > 0 else 0
        f1 = f1_score(all_labels, all_preds, average='weighted') * 100.0
        precision = precision_score(all_labels, all_preds, average='weighted') * 100.0
        recall = recall_score(all_labels, all_preds, average='weighted') * 100.0

        val_f1s.append(f1)

        scheduler.step()
        progress_bar(batch_idx, len(valid_loader),'Valid Loss: %.3f | Acc: %.3f%% (%d/%d)'% (loss/(batch_idx+1), 100.*correct/total, correct, total))

        # -------------------- early stopping --------------------
        if epoch > 10:  # min epochs before early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_student_model = student.state_dict()
                best_f1 = f1
                best_accuracy = val_accuracy
                best_precision = precision
                best_recall = recall
                best_epoch = epoch
                counter = 0
            else:
                counter += 1

        if counter >= 70:
            print("Early stopping triggered")
            break

        # -------------------- Logging --------------------
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([
                epoch,
                avg_train_total_loss,
                avg_train_kd_loss,
                avg_train_ce_loss,
                train_accuracy,
                train_f1,
                train_precision,
                train_recall,
                avg_val_loss,
                avg_val_kd_loss,
                avg_val_ce_loss,
                val_accuracy,
                f1,
                precision,
                recall,
                best_epoch,
            ])


    metrics = {
        'accuracy': float(best_accuracy),
        'f1': float(best_f1),
        'precision': float(best_precision),
        'recall': float(best_recall),
    }

    if best_student_model is None:
        best_student_model = student.state_dict()

    student.load_state_dict(best_student_model)

    plots = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_f1s': val_f1s,
    }

    return student, metrics, plots

