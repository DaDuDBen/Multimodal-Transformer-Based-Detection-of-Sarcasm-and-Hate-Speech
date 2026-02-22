Multimodal Transformer-Based Detection of Sarcasm and Hate Speech
Research-Grade Technical Specification
1. System Architecture
The proposed architecture, HarMeme-DualNet, utilizes a late-interaction multimodal transformer design. It leverages domain-specific pre-trained encoders (BERTweet for text, CLIP-ViT for visual) to handle the distinct properties of social media data, followed by a symmetric cross-modal attention mechanism to resolve visual-textual incongruence.
Component-Level Data Flow
Input:
$I$: RGB Image $(H, W, 3)$
$T$: Tokenized Text Sequence $(L,)$
Unimodal Encoding (Parallel):
Visual Encoder: CLIP-ViT-B/16.
Justification: Unlike ResNet, CLIP is pre-trained on 400M image-text pairs, effectively learning visual concepts aligned with linguistic descriptions. This provides a "warm-start" for detecting semantic concepts in memes (e.g., "clown", "trash") compared to ImageNet-trained CNNs which focus on object classification.
Output: Sequence of patch embeddings $V \in \mathbb{R}^{N \times d_v}$ (where $N=196$ patches + 1 CLS token).
Text Encoder: BERTweet-Large.
Justification: Trained on 850M tweets. It natively handles Twitter-specific characteristics (hashtags, user mentions, irregular grammar) significantly better than RoBERTa or the standard CLIP text encoder, which struggles with "internet slang."
Output: Sequence of token embeddings $L \in \mathbb{R}^{M \times d_t}$ (where $M=128$).
Feature Projection:
Linear projection layers $P_v$ and $P_t$ map both $V$ and $L$ to a shared dimension $d_{model} = 512$.
$V' = \text{ReLU}(V W_v + b_v) \in \mathbb{R}^{N \times 512}$
$L' = \text{ReLU}(L W_t + b_t) \in \mathbb{R}^{M \times 512}$
Cross-Modal Fusion Strategy: Symmetric Cross-Attention (Co-Attention).
Justification: Sarcasm and hate speech in memes often rely on incongruence (e.g., a "thumbs up" image with the text "Great job ruining the economy"). Late fusion (concatenation) averages these signals globally, losing the specific contradictory pairs. Cross-attention allows the text [CLS] token to attend to specific image patches that contradict it, and vice-versa.
Mechanism: Two parallel multi-head attention blocks:
Image-attended-by-Text ($L_{att} = \text{Attention}(Q=L', K=V', V=V')$)
Text-attended-by-Image ($V_{att} = \text{Attention}(Q=V', K=L', V=L')$)
Fusion: Concatenate pooled outputs: $F = [AvgPool(L_{att}) \oplus AvgPool(V_{att})]$.
Classification Heads:
Shared Layer: MLP (1024 $\to$ 512, GELU, Dropout 0.3).
Task A Head (Sarcasm): Linear (512 $\to$ 1) + Sigmoid.
Task B Head (Hate Speech): Linear (512 $\to$ 1) + Sigmoid.

2. Data Preprocessing Pipeline
Text Preprocessing
Normalization: Unicode normalization (NFKC) to handle diverse fonts often used in memes.
Hashtag Segmentation: Split #MakeAmericaGreatAgain $\to$ "Make America Great Again" using a Viterbi algorithm based on unigram frequencies.
Emoji Mapping: Do not remove emojis. Map them to descriptive textual tokens (e.g., 😂 $\to$ :face_with_tears_of_joy:) using the CLDR repository to allow BERTweet to process their semantic sentiment.
Tokenization: Use vinai/bertweet-large tokenizer. Explicitly preserve punctuation sequences ?!, ..., and " " (scare quotes) as they are high-signal markers for sarcasm.
Image Preprocessing
Resizing: Resize to $224 \times 224$. Use letterboxing (padding with black borders) to preserve aspect ratio. Stretching distorts text embedded in images (OCR-unfriendly) and facial expressions key to sarcasm.
Augmentation:
Training: Random Horizontal Flip (p=0.5), Color Jitter (Brightness/Contrast $\pm 10\%$).
Restriction: No rotation $>5^\circ$ or cropping $>5\%$. Memes often have text at edges; aggressive cropping removes context.
Label Noise & Class Imbalance
Label Noise: Use Soft Labels via Label Smoothing ($\alpha=0.1$). For crowdsourced disagreements (e.g., 3/5 annotators say "Hate"), use the probabilistic label $y=0.6$ rather than hard binary $1$. This calibrates the model to uncertainty.
Class Imbalance: Hate speech datasets (like MMHS150K) are often 90% non-hate.
Strategy: Focal Loss.
Formulation: $FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$.
Set $\gamma=2.0$ to down-weight easy negatives (non-hate) and focus learning on hard, ambiguous examples.

3. Feature Representation and Cross-Modal Alignment
The Modality Gap
Unimodal encoders reside in disjoint manifolds. Concatenating a BERTweet vector (text semantic space) with a CLIP vector (visual semantic space) without alignment results in a geometrically incoherent feature vector.
Alignment Strategy
We employ a Task-Specific Projection with Contrastive Initialization.
Shared Space: We project both modalities to $d=512$.
Alignment Objective: During Stage 1 (Pre-training), we apply an InfoNCE (Contrastive) Loss on the batch.
Positive pairs: $(Image_i, Text_i)$ from the same meme.
Negative pairs: $(Image_i, Text_j)$ where $i \neq j$.
This forces the projection layers to map "contradictory" inputs into a metric space where their distance represents their interaction, rather than their modality origin.
Contradiction Representation: In this aligned space, sarcasm is often represented not by alignment (parallel vectors) but by orthogonality or opposition in specific subspaces (e.g., Sentiment dimension). The MLP head learns to classify these geometric relationships.

4. Model Implementation
Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        # Symmetric Cross Attention
        self.text_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.img_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, v, t):
        # v: [B, N_patches, d], t: [B, N_tokens, d]
        
        # Text queries Image (What in the image relates to this text?)
        # Q=Text, K=Image, V=Image
        t_attended, _ = self.text_attn(query=t, key=v, value=v)
        
        # Image queries Text (What in the text relates to this image patch?)
        # Q=Image, K=Text, V=Text
        v_attended, _ = self.img_attn(query=v, key=t, value=t)
        
        # Residual + Norm
        t_out = self.layer_norm(t + self.dropout(t_attended))
        v_out = self.layer_norm(v + self.dropout(v_attended))
        
        # Global Average Pooling
        t_pool = t_out.mean(dim=1)  # [B, d]
        v_pool = v_out.mean(dim=1)  # [B, d]
        
        return torch.cat([t_pool, v_pool], dim=1) # [B, 2*d]

class MultimodalDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_enc = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        self.txt_enc = AutoModel.from_pretrained("vinai/bertweet-large")
        
        # Projection to shared d=512
        self.proj_v = nn.Linear(768, 512)
        self.proj_t = nn.Linear(1024, 512) # BERTweet large is 1024
        
        self.fusion = CrossModalFusion(d_model=512)
        
        # Classification Heads
        self.shared_mlp = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.3)
        )
        self.head_sarcasm = nn.Linear(512, 1)
        self.head_hate = nn.Linear(512, 1)

    def forward(self, images, input_ids, attention_mask):
        # 1. Encode
        v_feat = self.img_enc(images).last_hidden_state # [B, 197, 768]
        t_feat = self.txt_enc(input_ids, attention_mask).last_hidden_state # [B, L, 1024]
        
        # 2. Project
        v_proj = F.relu(self.proj_v(v_feat))
        t_proj = F.relu(self.proj_t(t_feat))
        
        # 3. Fuse
        fused_vec = self.fusion(v_proj, t_proj)
        
        # 4. Classify
        shared = self.shared_mlp(fused_vec)
        return self.head_sarcasm(shared), self.head_hate(shared)

# Loss Function
# L = λ1 * FL_sarcasm + λ2 * FL_hate + λ3 * L_align
# Standard weights: λ1=1.0, λ2=1.0, λ3=0.1 (Alignment is auxiliary)


5. Training Strategy
Stage 1: Alignment Tuning (5 Epochs)
Freeze: CLIP and BERTweet backbones.
Train: Only proj_v, proj_t, and fusion layers.
Justification: Backpropagating large gradients from an initialized fusion layer into pre-trained transformers destroys their feature extraction capability ("catastrophic forgetting"). Stage 1 aligns the projection spaces.
Stage 2: Full Fine-Tuning (15 Epochs)
Unfreeze: All layers.
Discriminative Learning Rates:
Encoders: $1e^{-5}$ (Preserve pre-trained knowledge).
Heads/Fusion: $1e^{-4}$ (Learn task specifics).
Decay: Layer-wise decay of $0.95$ (lower layers change less).
Multitask Balancing
Use Uncertainty Weighting (Kendall et al., CVPR 2018) to dynamically adjust loss weights $\lambda_1, \lambda_2$.
$\mathcal{L}_{total} = \frac{1}{2\sigma_1^2}\mathcal{L}_{sarcasm} + \frac{1}{2\sigma_2^2}\mathcal{L}_{hate} + \log(\sigma_1\sigma_2)$
This prevents the easier task (often Hate detection) from dominating the gradient updates over the harder task (Sarcasm).

6. Hyperparameter Specification
Hyperparameter
Value
Justification
Optimizer
AdamW
Handles weight decay correctly for Transformers, unlike standard Adam.
Batch Size
32
Limits GPU VRAM usage (BERTweet Large + ViT is heavy).
LR (Encoder)
$1e^{-5}$
Low LR prevents destroying pre-trained syntax/visual features.
LR (Head)
$1e^{-4}$
Higher LR required for initialized MLP layers to converge.
Weight Decay
$1e^{-2}$
Standard regularization for Transformer adaptors.
Dropout
0.3
Higher dropout needed for multimodal fusion to prevent reliance on one modality.
Warmup Steps
500
Linear warmup stabilizes gradients during early training (Stage 2).
Gradient Clip
1.0
Prevents exploding gradients in the cross-attention modules.
Img Resolution
$224 \times 224$
Native resolution for ViT-B/16; higher res yields diminishing returns vs compute.


7. Experimental Design
Benchmark Datasets
HatefulMemes (Facebook): Primary benchmark for multimodal reasoning.
HarMeme (LCS2): Specific for "harmful" vs "harmless" detection, testing generalization to COVID-19/political contexts.
Multimodal Sarcasm Detection (MSD): For the sarcasm branch.
Protocol
Splits: Stratified 70/15/15.
Metrics: Report Macro-F1 and AUC-ROC. Macro-F1 is critical due to the "Non-Hate" class dominance.
Significance: Paired t-test over 5 random seeds.
Baseline Suite
Text-Only: BERTweet-Large + MLP.
Image-Only: CLIP-ViT + MLP.
Late Fusion (Concat): BERTweet + CLIP + Concat + MLP (No attention).
Proposed: HarMeme-DualNet (Co-Attention).
Expected Results (Hypothesis)
Ablation (b): Removing Cross-Attention (replacing with Concat) will drop Sarcasm F1 by ~5-8%, proving that interaction is needed, not just presence of features.
Multitask Transfer: Joint training should improve Sarcasm detection on memes where hate speech is the vehicle for sarcasm, outperforming single-task models.
Intellectual Contribution
This system moves beyond naive concatenation by explicitly modeling the geometric tension between modalities via co-attention. It addresses the architectural inability of unimodal models to detect "meaning inversion" (sarcasm) and provides a rigorous, reproducible recipe for deploying this in high-throughput social media monitoring environments.

