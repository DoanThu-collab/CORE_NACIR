import torch
import torch.nn as nn
from typing import Dict, Any
import sys


from nacir.config import ScheduleFn, DynamicScheduleConfig, default_dynamic_schedule

class SchedulerLinear(nn.Module):
    """Baseline: Linear mapping"""
    def __init__(self, input_dim=3, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class SchedulerMLP(nn.Module):
    """Mạng nơ-ron nhỏ (khoảng 500 params)"""
    def __init__(self, input_dim=3, hidden_dim=32, output_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.net(x)

class SchedulerGRU(nn.Module):
    """Mạng Recurrent (GRU) để nhớ context của các turn trước"""
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out), hidden

class RelevanceGatingNetwork(nn.Module):
    """
    Cross-Modal Relevance Gating:
    Sử dụng Cross-Attention để ép Query 'đọc' và đánh giá mức độ 
    liên quan của Beliefs. Từ đó xuất ra các hệ số memory control.
    """
    def __init__(self, embed_dim=768, num_heads=4, output_dim=4):
        super().__init__()
        # Để đơn giản và nhanh, dùng 1 lớp MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Mạng MLP gom kết quả attention
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )
        # Vector padding mặc định nếu không có belief nào
        self.empty_belief = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def forward(self, query_vec, belief_vecs):
        """
        query_vec: [batch, embed_dim]
        belief_vecs: [batch, seq_len, embed_dim]
        """
        if belief_vecs.size(1) == 0:
            # Nếu không có belief, dùng empty_belief
            belief_vecs = self.empty_belief.expand(query_vec.size(0), 1, -1)
            
        q = query_vec.unsqueeze(1) # [batch, 1, embed_dim]
        k = v = belief_vecs        # [batch, seq_len, embed_dim]
        
        # Cross Attention
        attn_out, _ = self.attention(q, k, v)
        
        # Squeeze sequence dimension
        out = attn_out.squeeze(1) # [batch, embed_dim]
        
        # Dự đoán các scalars
        return self.mlp(out)

class LearnedScheduler:
    """
    Adapter pattern (Đề xuất 3: Learned Scheduler).
    Tải weights của mạng MLP để thay thế lịch trình tuyến tính tĩnh.
    """
    def __init__(self, model_type: str = "mlp", model_path: str = None, enabled: bool = True, device: str = "cpu"):
        self.enabled = enabled
        self.model_type = model_type
        self.model_path = model_path
        self.device = device
        self.base_schedule = DynamicScheduleConfig()
        self.hidden_state = None # Dành riêng cho GRU
        
        self.model = None
        if self.enabled and self.model_path:
            try:
                if self.model_type == "linear":
                    self.model = SchedulerLinear()
                elif self.model_type == "mlp":
                    self.model = SchedulerMLP()
                elif self.model_type == "gru":
                    self.model = SchedulerGRU()
                elif self.model_type == "attention":
                    # PlugIR/BLIP text encoder projects down to 256
                    self.model = RelevanceGatingNetwork(embed_dim=256)
                else:
                    raise ValueError(f"Unknown model_type: {self.model_type}")
                    
                # Ignore errors for now if weights don't strictly match during testing
                self.model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
                self.model.to(device)
                self.model.eval()
            except Exception as e:
                print(f"Lỗi load Learned Scheduler model: {e}. Sẽ dùng fallback heuristic.")
                self.model = None

    def get_schedule(self, turn: int, q_t: torch.Tensor = None, beliefs: Any = None, encoder_fn: Any = None) -> Dict[str, float]:
        """Implement ScheduleFn signature (có context đầy đủ)."""
        if not self.enabled or self.model is None:
            # Fallback to base heuristic if disabled or model not provided
            return default_dynamic_schedule(turn, self.base_schedule)
        
        with torch.no_grad():
            if self.model_type == "attention":
                if q_t is None or beliefs is None or encoder_fn is None:
                    # Fallback if pipeline doesn't pass context
                    return default_dynamic_schedule(turn, self.base_schedule)
                
                # Encode beliefs
                pos_texts = [b.attribute for b in beliefs.positive_beliefs]
                neg_texts = [b.attribute for b in beliefs.negative_beliefs]
                all_texts = pos_texts + neg_texts
                
                if len(all_texts) > 0:
                    belief_vecs = encoder_fn(all_texts).to(self.device).unsqueeze(0) # [1, seq_len, 256]
                else:
                    belief_vecs = torch.zeros(1, 0, 256).to(self.device)
                
                q_t_batched = q_t.unsqueeze(0).to(self.device) # [1, 256]
                outputs = self.model(q_t_batched, belief_vecs)[0] # Shape: [4]
                
            else:
                # Old MLP/Linear features
                num_pos = len(beliefs.positive_beliefs) if beliefs else 1
                num_neg = len(beliefs.negative_beliefs) if beliefs else 1
                norm_turn = turn / 11.0 
                features = torch.tensor([[norm_turn, float(num_pos), float(num_neg)]], dtype=torch.float32).to(self.device)
                
                if self.model_type == "gru":
                    outputs, self.hidden_state = self.model(features, self.hidden_state)
                    outputs = outputs[0]
                else:
                    outputs = self.model(features)[0] # Shape: [4]
        
        alpha, beta, ortho, penalty = outputs.cpu().numpy()
        
        overrides = {
            "memory_alpha": float(alpha),
            "memory_beta": float(beta),
            "ortho_strength": float(ortho),
            "masking_penalty_weight": float(penalty),
            "itm_weight": default_dynamic_schedule(turn, self.base_schedule).get("itm_weight", 0.5)
        }
        
        return overrides

    def as_schedule_fn(self) -> Any:
        """Trả về hàm callback để đưa vào NACIRPlusPlusPipeline."""
        def wrapper(*args):
            if len(args) == 1:
                return self.get_schedule(args[0])
            elif len(args) == 4:
                return self.get_schedule(args[0], args[1], args[2], args[3])
            return default_dynamic_schedule(args[0], self.base_schedule)
        return wrapper
