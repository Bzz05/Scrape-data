import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override

class RllibTransformerPolicy(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.obs_dim = int(obs_space.shape[0])
        self.act_dim = num_outputs
        custom_config = model_config.get("custom_model_config", {})
        d_model = custom_config.get("d_model", 128)
        nhead = custom_config.get("nhead", 8)
        num_layers = custom_config.get("num_layers", 4)
        self.input_proj = nn.Linear(self.obs_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, self.act_dim)
        self.value_head = nn.Linear(d_model, 1)
        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()        # (B, obs_dim)
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, d_model)
        x = self.transformer_encoder(x)      # (B, 1, d_model)
        x_last = x.squeeze(1)                # (B, d_model)
        self._features = x_last
        policy_logits = self.policy_head(x_last)
        return policy_logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "Call forward() first"
        return self.value_head(self._features).squeeze(1)
