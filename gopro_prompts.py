import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class GOProPromptLearner(nn.Module):
    """Simplified GOPro: learn context tokens + class tokens for CLIP"""
    
    def __init__(self, clip_model, num_classes=2, ctx_len=4, temp_init=0.07):
        super().__init__()
        self.clip = clip_model
        self.num_classes = num_classes
        self.ctx_len = ctx_len
        
        # Learnable context tokens
        d_model = self.clip.text_model.config.hidden_size
        self.ctx = nn.Parameter(torch.randn(ctx_len, d_model) * 0.02)
        
        # Learnable class tokens
        self.class_token = nn.Parameter(torch.randn(num_classes, d_model) * 0.02)
        
        # Temperature (GOPro style)
        self.temp = nn.Parameter(torch.ones([]) * np.log(temp_init))
        
        # Positional embeddings (frozen)
        self.register_buffer('position_ids', 
            torch.arange(77).expand((1, -1)))  # CLIP max context

    def forward(self, batch_size):
        """Returns text features for all classes (B, num_classes, dim)"""
        device = next(self.parameters()).device
        
        # Template: [CLS] + ctx + class_token + [SEP] + padding
        bos_embed = self.clip.text_model.embeddings.token_embedding.weight[49406:49407]  # CLS
        eos_embed = self.clip.text_model.embeddings.token_embedding.weight[49407:49408]  # SEP
        
        # Base sequence embeddings (B, seq_len=77, dim)
        seq_len = 77
        base_emb = torch.zeros(batch_size, seq_len, self.clip.text_projection.in_features, 
                              device=device)
        
        # Add CLS + context
        base_emb[:, 0:1] = bos_embed
        base_emb[:, 1:1+self.ctx_len] = self.ctx.unsqueeze(0)
        
        # Position embeddings
        pos_emb = self.clip.text_model.embeddings.position_embedding(self.position_ids.to(device))
        base_emb = base_emb + pos_emb
        
        # For each class: insert class token after context
        text_features = []
        for c in range(self.num_classes):
            seq = base_emb.clone()
            seq[:, 1+self.ctx_len:1+self.ctx_len+1] = self.class_token[c].unsqueeze(0)
            seq[:, -1:] = eos_embed  # EOS at end
            
            # Attention mask (simple: all positions attended)
            attn_mask = torch.ones(batch_size, seq_len, device=device)
            
            # Text transformer
            outputs = self.clip.text_model(
                inputs_embeds=seq,
                attention_mask=attn_mask,
                output_attentions=False,
                output_hidden_states=False
            )
            
            # Pool: take [EOS] position
            eos_pos = torch.full((batch_size,), seq_len-1, dtype=torch.long, device=device)
            pooled = outputs.last_hidden_state[torch.arange(batch_size), eos_pos]
            
            # Project to CLIP text space
            text_feat = self.clip.text_projection(pooled)
            text_feat = F.normalize(text_feat, dim=-1)
            text_features.append(text_feat)
        
        return torch.stack(text_features, dim=1)  # (B, num_classes, dim)

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
            
            # Image features (frozen CLIP)
            with torch.no_grad():
                img_outputs = clip_model.get_image_features(images)
                img_feats = F.normalize(img_outputs, dim=-1)
            
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
            
            # Image features
            img_outputs = clip_model.get_image_features(images)
            img_feats = F.normalize(img_outputs, dim=-1)
            
            # Learned text features (class 1 = tree)
            text_feats = prompt_learner(batch_size)
            tree_scores = torch.einsum('bid,bid->b', img_feats, text_feats[:, 1]) * prompt_learner.temp.exp()
            
            all_scores.extend(tree_scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_scores), np.array(all_labels)
