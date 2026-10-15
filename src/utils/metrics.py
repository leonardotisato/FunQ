import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.utils import resample
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

def bootstrap_metrics(all_preds, all_labels, n_bootstrap=1000, confidence_level=0.95):
    # Initialize lists to store metric results
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    class_wise_metrics = {i: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for i in np.unique(all_labels)}
    
    
    # iterate over bootstrap repetitions
    for _ in range(n_bootstrap):
        boot_preds, boot_labels = resample(all_preds, all_labels, replace=True)
        
        # Compute metrics on that bootstrap sample
        accuracies.append(accuracy_score(boot_labels, boot_preds))
        precisions.append(precision_score(boot_labels, boot_preds, average='macro'))
        recalls.append(recall_score(boot_labels, boot_preds, average='macro'))
        f1_scores.append(f1_score(boot_labels, boot_preds, average='macro'))

        # compute class-wise metrics
        for cls in np.unique(boot_labels):
                selected_labels = (boot_labels == cls).astype(int)
                selected_preds = (boot_preds == cls).astype(int)
                cls_acc = accuracy_score(selected_labels, selected_preds)
                cls_precision = precision_score(boot_labels, boot_preds, average=None, labels=[cls])[0]
                cls_recall = recall_score(boot_labels, boot_preds, average=None, labels=[cls])[0]
                cls_f1 = f1_score(boot_labels, boot_preds, average=None, labels=[cls])[0]

                class_wise_metrics[cls]['accuracy'].append(cls_acc)
                class_wise_metrics[cls]['precision'].append(cls_precision)
                class_wise_metrics[cls]['recall'].append(cls_recall)
                class_wise_metrics[cls]['f1'].append(cls_f1)
    
    # Compute confidence intervals
    lower_acc, upper_acc = np.percentile(accuracies, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
    lower_prec, upper_prec = np.percentile(precisions, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
    lower_rec, upper_rec = np.percentile(recalls, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
    lower_f1, upper_f1 = np.percentile(f1_scores, [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])
    
    class_wise_ci = {}
    for cls, metrics in class_wise_metrics.items():
        class_wise_ci[cls] = {
            'accuracy': (np.mean(metrics['accuracy']), 
                        tuple(np.percentile(metrics['accuracy'], [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))),
            'precision': (np.mean(metrics['precision']), 
                        tuple(np.percentile(metrics['precision'], [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))),
            'recall': (np.mean(metrics['recall']), 
                    tuple(np.percentile(metrics['recall'], [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100]))),
            'f1': (np.mean(metrics['f1']), 
                tuple(np.percentile(metrics['f1'], [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100])))
        }

    print('----------------------------------------------------------------')
    print(f"[Accuracy]: {np.mean(accuracies)} ({lower_acc}, {upper_acc})")
    print(f"[Precision]: {np.mean(precisions)} ({lower_prec}, {upper_prec})")
    print(f"[Reall]: {np.mean(recalls)} ({lower_rec}, {upper_rec})")
    print(f"[F1]: {np.mean(f1_scores)} ({lower_f1}, {upper_f1})")
    print('----------------------------------------------------------------')

    # class-wise metrics
    for cls in class_wise_ci:
        print(f'------------------[Class {cls}]---------------------------')
        print(f"[Accuracy]: {class_wise_ci[cls]['accuracy']}")
        print(f"[Precision]: {class_wise_ci[cls]['precision']}")
        print(f"[Recall]: {class_wise_ci[cls]['recall']}")
        print(f"[F1]: {class_wise_ci[cls]['f1']}")
        print('----------------------------------------------------------------')


    return {
        'accuracy': (np.mean(accuracies), lower_acc, upper_acc),
        'precision': (np.mean(precisions), lower_prec, upper_prec),
        'recall': (np.mean(recalls), lower_rec, upper_rec),
        'f1_score': (np.mean(f1_scores), lower_f1, upper_f1),
        # Class-wise metrics
        'accuracy_healthy': class_wise_ci[0]['accuracy'],
        'precision_healthy': class_wise_ci[0]['precision'],
        'recall_healthy': class_wise_ci[0]['recall'],
        'f1_score_healthy': class_wise_ci[0]['f1'],
        
        'accuracy_melanoma': class_wise_ci[2]['accuracy'],
        'precision_melanoma': class_wise_ci[2]['precision'],
        'recall_melanoma': class_wise_ci[2]['recall'],
        'f1_score_melanoma': class_wise_ci[2]['f1'],
        
        'accuracy_nevus': class_wise_ci[1]['accuracy'],
        'precision_nevus': class_wise_ci[1]['precision'],
        'recall_nevus': class_wise_ci[1]['recall'],
        'f1_score_nevus': class_wise_ci[1]['f1'],
        
        'accuracy_chrpe': class_wise_ci[3]['accuracy'],
        'precision_chrpe': class_wise_ci[3]['precision'],
        'recall_chrpe': class_wise_ci[3]['recall'],
        'f1_score_chrpe': class_wise_ci[3]['f1'],
    }


# -------------------------------------------------------------------------
# Grad-CAM utilities
# -------------------------------------------------------------------------
def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Returns last convolutional layer of the architecture
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model.")
    return last_conv

class GradCAM:
    """
    GradCAM on last convolutional layer
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __call__(self, x: torch.Tensor, target_category: int = None):

        self.model.zero_grad()
        output = self.model(x)  # (1, num_classes)

        if target_category is None:
            target_category = output.argmax(dim=1).item()

        score = output[0, target_category]
        score.backward(retain_graph=False)

        gradients = self.gradients          # (1, C, H', W')
        activations = self.activations      # (1, C, H', W')

        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam.zero_()

        return output.detach(), cam.detach()

# -------------------------------------------------------------------------
# Grad-CAM visualization utilities
# -------------------------------------------------------------------------

def tensor_to_rgb_image(tensor):
    """
    Converte un tensore (1,3,H,W) o (3,H,W) in immagine numpy [0,1].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def save_gradcam(image_tensor,
                              cam_student,
                              out_path,
                              alpha=0.4):
 
    img = tensor_to_rgb_image(image_tensor)    # (H, W, 3)
    cam_s = cam_student.squeeze().cpu().numpy()          # (H, W)

    plt.figure(figsize=(10, 5))

    plt.imshow(img)
    plt.imshow(cam_s, cmap='jet', alpha=alpha)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    
    

