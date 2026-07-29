from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaModel


class TemporalEmbedding(nn.Module):
    """Embed the most recent time-of-day and day-of-week for every sensor."""

    def __init__(self, slots_per_day: int, features: int):
        super().__init__()
        self.slots_per_day = slots_per_day
        self.time_of_day = nn.Parameter(torch.empty(slots_per_day, features))
        self.day_of_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_of_day)
        nn.init.xavier_uniform_(self.day_of_week)

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        # history_data: [batch, history, nodes, features]
        time_index = (
            history_data[:, -1, :, 1] * self.slots_per_day
        ).long().clamp_(0, self.slots_per_day - 1)
        week_index = history_data[:, -1, :, 2].long().clamp_(0, 6)
        return self.time_of_day[time_index] + self.day_of_week[week_index]


class LlamaPFGA(nn.Module):
    """
    Partially frozen Llama 3.1 with graph attention in the last U layers.

    LayerNorm remains trainable in every layer. The early layers keep frozen
    attention and FFN weights with Llama's native causal attention. The final
    graph layers use an additive adjacency mask where non-neighbors receive
    -inf and every node can attend to itself; their original attention weights
    and q_proj/v_proj LoRA adapters are trainable, while their FFNs stay frozen.
    """

    def __init__(
        self,
        model_path: str,
        adjacency_matrix,
        num_layers: Optional[int] = None,
        graph_layers: int = 2,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"Llama model directory does not exist: {model_dir.resolve()}"
            )

        backbone = LlamaModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        backbone.config.use_cache = False

        available_layers = len(backbone.layers)
        if num_layers in (None, 0):
            num_layers = available_layers
        if not 1 <= num_layers <= available_layers:
            raise ValueError(
                f"num_layers must be in [1, {available_layers}], got {num_layers}"
            )
        if num_layers < available_layers:
            backbone.layers = nn.ModuleList(backbone.layers[:num_layers])
            backbone.config.num_hidden_layers = num_layers

        if not 1 <= graph_layers <= num_layers:
            raise ValueError(
                f"graph_layers must be in [1, {num_layers}], got {graph_layers}"
            )

        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        graph_layer_indices = list(range(num_layers - graph_layers, num_layers))
        layer_expression = "|".join(str(index) for index in graph_layer_indices)
        target_expression = (
            rf"layers\.({layer_expression})\.self_attn\.(q_proj|v_proj)"
        )
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_expression,
            bias="none",
        )
        self.model = get_peft_model(backbone, lora_config)

        # Traffic features are supplied through inputs_embeds, so the 1+ GB token
        # embedding table is never used. Releasing it materially lowers VRAM use.
        base_model = self.model.get_base_model()
        base_model.embed_tokens = None

        # PEFT freezes every original backbone parameter when it injects LoRA.
        # Restore the paper's partially frozen strategy explicitly:
        #   * LayerNorm is trainable in all layers;
        #   * original self-attention plus LoRA is trainable in the final U layers;
        #   * every FFN remains frozen.
        # Start from a known state so this behavior does not depend on PEFT's
        # internal defaults.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        graph_start = num_layers - graph_layers
        for layer_index, layer in enumerate(base_model.layers):
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                for parameter in norm.parameters():
                    parameter.requires_grad = True

            if layer_index >= graph_start:
                for parameter in layer.self_attn.parameters():
                    parameter.requires_grad = True

        for parameter in base_model.norm.parameters():
            parameter.requires_grad = True

        adjacency = torch.as_tensor(adjacency_matrix, dtype=torch.bool)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(
                f"adjacency_matrix must be square, got {tuple(adjacency.shape)}"
            )
        adjacency = adjacency | torch.eye(adjacency.shape[0], dtype=torch.bool)
        self.register_buffer("graph_allowed", adjacency, persistent=False)

        self.hidden_size = backbone.config.hidden_size
        self.num_layers = num_layers
        self.graph_layers = graph_layers
        self.dropout = nn.Dropout(lora_dropout)

    def _graph_mask(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        allowed = self.graph_allowed.to(device=device)
        mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
        mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        backbone = self.model.get_base_model()
        if inputs_embeds.shape[1] != self.graph_allowed.shape[0]:
            raise ValueError(
                "The station-token sequence and adjacency matrix disagree: "
                f"{inputs_embeds.shape[1]} vs {self.graph_allowed.shape[0]}"
            )

        compute_dtype = next(backbone.layers[0].parameters()).dtype
        hidden_states = inputs_embeds.to(dtype=compute_dtype)
        sequence_length = hidden_states.shape[1]
        cache_position = torch.arange(sequence_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)

        # With SDPA, None selects Llama's native causal kernel in the early
        # layers. Supplying the 4-D additive mask disables causality and limits
        # the final layers to graph neighbors plus self.
        causal_mask = backbone._update_causal_mask(
            None, hidden_states, cache_position, None, False
        )
        graph_mask = self._graph_mask(hidden_states.dtype, hidden_states.device)
        position_embeddings = backbone.rotary_emb(hidden_states, position_ids)
        graph_start = self.num_layers - self.graph_layers

        for layer_index, decoder_layer in enumerate(backbone.layers):
            layer_mask = causal_mask if layer_index < graph_start else graph_mask
            if backbone.gradient_checkpointing and self.training:
                layer_outputs = backbone._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    layer_mask,
                    position_ids,
                    None,
                    False,
                    False,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=layer_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

        hidden_states = backbone.norm(hidden_states)
        return self.dropout(hidden_states).float()


class ST_LLM(nn.Module):
    """Llama 3.1 graph-enhanced traffic forecaster for PEMS08."""

    def __init__(
        self,
        adj_mx,
        model_path: str = "./Meta-Llama-3.1-8B-Instruct",
        input_dim: int = 3,
        num_nodes: int = 170,
        input_len: int = 12,
        output_len: int = 12,
        llm_layers: Optional[int] = 32,
        graph_layers: int = 2,
        embedding_dim: int = 256,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        dropout: float = 0.1,
        slots_per_day: int = 288,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if len(adj_mx) != num_nodes:
            raise ValueError(
                f"num_nodes={num_nodes}, but adjacency has {len(adj_mx)} nodes"
            )

        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len

        self.token_projection = nn.Linear(
            input_dim * input_len, embedding_dim
        )
        self.temporal_embedding = TemporalEmbedding(
            slots_per_day, embedding_dim
        )
        self.node_embedding = nn.Parameter(
            torch.empty(num_nodes, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding)

        self.llama = LlamaPFGA(
            model_path=model_path,
            adjacency_matrix=adj_mx,
            num_layers=llm_layers,
            graph_layers=graph_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.fusion = nn.Linear(embedding_dim * 3, self.llama.hidden_size)
        self.input_dropout = nn.Dropout(dropout)
        self.regression = nn.Linear(self.llama.hidden_size, output_len)

    def param_num(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def count_trainable_params(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(self, history_data: torch.Tensor) -> torch.Tensor:
        if history_data.ndim != 4:
            raise ValueError(
                "history_data must have shape [batch, history, nodes, features]"
            )
        batch, history, nodes, features = history_data.shape
        expected = (self.input_len, self.num_nodes, self.input_dim)
        if (history, nodes, features) != expected:
            raise ValueError(
                f"Expected history/nodes/features={expected}, "
                f"got {(history, nodes, features)}"
            )

        token_input = history_data.permute(0, 2, 1, 3).reshape(
            batch, nodes, history * features
        )
        token_embedding = self.token_projection(token_input)
        temporal_embedding = self.temporal_embedding(history_data)
        node_embedding = self.node_embedding.unsqueeze(0).expand(batch, -1, -1)

        fused = torch.cat(
            [token_embedding, temporal_embedding, node_embedding], dim=-1
        )
        fused = self.input_dropout(F.gelu(self.fusion(fused)))
        hidden_states = self.llama(fused)

        # [batch, future, nodes, 1], matching the original repository contract.
        prediction = self.regression(hidden_states)
        return prediction.permute(0, 2, 1).unsqueeze(-1)
