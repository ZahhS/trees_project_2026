import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class GOProPromptLearner(nn.Module):
    def __init__(self, clip_model, num_classes=2, ctx_len=4, temp_init=0.07):
        super().__init__()
        self.num_classes = num_classes
        self.ctx_len = ctx_len
        
        d_model = clip_model.text_projection.in_features  # 512
        self.ctx = nn.Parameter(torch.randn(ctx_len, d_model) * 0.02)
        self.class_token = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        self.temp = nn.Parameter(torch.ones([]) * np.log(temp_init))

    def forward(self, batch_size):
        device = next(self.parameters()).device
        
        # Repeat context and class tokens for batch
        ctx = self.ctx.unsqueeze(0).expand(batch_size, -1, -1)  # (B, ctx_len, 512)
        class_tokens = self.class_token.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, 512)
        
        text_feats = torch.cat([ctx, class_tokens], dim=1)  # (B, ctx_len+num_classes, 512)
        text_feats = F.normalize(text_feats.mean(dim=1), dim=-1)  # (B, 512) pool over tokens
        
        return text_feats.unsqueeze(1)  # (B, 1, 512) - dummy class dim for compatibility


def train_gopro_prompts(clip_model, processor, train_loader, num_epochs=10, lr=1e-3):
    """Train GOPro prompts on your train_loader"""
    device = next(clip_model.parameters()).device
    # GOPro prompt learner
    prompt_learner = GOProPromptLearner(clip_model, num_classes=2).to(device)
    
    # Freeze CLIP
    for p in clip_model.parameters():
        p.requires_grad_(False)
    
    optimizer = torch.optim.AdamW(prompt_learner.parameters(), lr=lr, weight_decay=0.1)
    
    prompt_learner.train()
    for epoch in range(num_epochs):
        total_loss = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            with torch.no_grad():
                vision_outputs = clip_model.vision_model(pixel_values=images)
                pooled = vision_outputs.pooler_output  # (B, 768)
                img_feats = clip_model.visual_projection(pooled)  # (B, 512)
                img_feats = F.normalize(img_feats, dim=-1)
            
            # Learned text features
            text_feats = prompt_learner(batch_size)  # (B, 2, dim)
            
            # Similarities (GOPro style with learned temp)
            logits = torch.einsum('bid,bjd->bij', img_feats, text_feats) * prompt_learner.temp.exp()
            
            # Classification loss
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prompt_learner.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return prompt_learner

def eval_gopro_prompts(clip_model, prompt_learner, test_loader, device):
    """Evaluate with learned GOPro prompts - plug directly into your ROC code"""
    clip_model.eval()
    prompt_learner.eval()
    
    all_scores, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            vision_outputs = clip_model.vision_model(pixel_values=images)
            pooled = vision_outputs.pooler_output
            img_feats = clip_model.visual_projection(pooled)
            img_feats = F.normalize(img_feats, dim=-1)
            
            # Learned text features (class 1 = tree)
            text_feats = prompt_learner(batch_size)
            tree_scores = torch.einsum('bid,bid->b', img_feats, text_feats[:, 1]) * prompt_learner.temp.exp()
            
            all_scores.extend(tree_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_scores), np.array(all_labels)
