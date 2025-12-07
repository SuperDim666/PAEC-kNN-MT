#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/models/paec_policy_network.py

This module defines the architecture for the PAEC Policy Network (Student, Pi_phi).
The Policy Network is responsible for efficiently predicting the optimal control
actions (Index Type, k, lambda) given the current system state and history.
It is trained via behavioral cloning (distillation) from the computationally
expensive online trajectory planner (Teacher).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject temporal order information
    into the Transformer model.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        """
        Initializes the positional encoding module.

        Args:
            d_model (int): The hidden dimension size of the model inputs.
            dropout (float): Dropout probability.
            max_len (int): Maximum length of the input sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create a matrix of [max_len, 1] representing positions
        position = torch.arange(max_len).unsqueeze(1)
        # Compute the division term for the sinusoidal functions
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Initialize the encoding matrix
        pe = torch.zeros(max_len, 1, d_model)
        # Apply sine to even indices
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # Register as a buffer so it is part of the state_dict but not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [SeqLen, Batch, Dim].

        Returns:
            torch.Tensor: Output tensor with positional information added.
        """
        if hasattr(self, 'pe'):
            pe = getattr(self, 'pe')
            # Add positional encoding up to the current sequence length
            x = x + pe[:x.size(0)]
        return self.dropout(x)


class PAECPolicyNetwork(nn.Module):
    """
    The Policy Network (Pi_phi) for the PAEC framework.

    This lightweight model serves as the "Student" in the distillation process.
    It takes the current system state (S_t) and historical context as input
    and predicts the optimal control action (A_t*) that minimizes the long-term
    Lyapunov energy.

    Architecture:
    - Inputs: State vectors, Action history, optional Text/HiddenState embeddings.
    - Backbone: Transformer Encoder to process the temporal sequence of states.
    - Heads:
        1. Index Head: Classification logits for the Index Type (None, Exact, HNSW, IVF_PQ).
        2. Continuous Head: Regression outputs for 'k' (normalized) and 'lambda'.
    """
    def __init__(self,
        S_DIM: int,
        ACTION_DIM: int,
        hid_dim: int,
        nhead: int,
        layers: int,
        history_len: int,
        use_text_embeddings: bool,
        text_embedding_dim: int,
        use_decoder_hidden_state: bool,
        decoder_hidden_state_dim: int,
        dropout: float = 0.1
    ):
        """
        Initializes the Policy Network.

        Args:
            S_DIM (int): Dimension of the system state vector S_t.
            ACTION_DIM (int): Dimension of the action vector A_t (typically 6).
            hid_dim (int): Hidden dimension size for internal embeddings and Transformer.
            nhead (int): Number of attention heads in the Transformer.
            layers (int): Number of Transformer Encoder layers.
            history_len (int): Length of the historical sequence to consider.
            use_text_embeddings (bool): Whether to include source/prefix text embeddings.
            text_embedding_dim (int): Dimension of the text embeddings.
            use_decoder_hidden_state (bool): Whether to include the NMT decoder's hidden state.
            decoder_hidden_state_dim (int): Dimension of the decoder's hidden state.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.model_type = 'PAECPolicyNetwork'
        self.hid_dim = hid_dim
        self.history_len = history_len
        self.use_text_embeddings = use_text_embeddings
        self.use_decoder_hidden_state = use_decoder_hidden_state
        
        # Primary embeddings for State and Action
        self.state_embedding = nn.Linear(S_DIM, hid_dim)
        self.action_embedding = nn.Linear(ACTION_DIM, hid_dim)
        
        # Positional encoder for the sequence [History... , Current_State]
        self.pos_encoder = PositionalEncoding(hid_dim, dropout, max_len=history_len + 1)
        self.total_context_dim = 0
        
        # --- Auxiliary Context Embeddings ---
        if self.use_decoder_hidden_state:
            if decoder_hidden_state_dim == 0:
                raise ValueError("'use_decoder_hidden_state' is True, but 'decoder_hidden_state_dim' is 0.")
            self.decoder_hidden_embedding = nn.Linear(decoder_hidden_state_dim, hid_dim)
            self.total_context_dim += hid_dim

        if self.use_text_embeddings:
            if text_embedding_dim == 0:
                raise ValueError("'use_text_embeddings' is True, but 'text_embedding_dim' is 0.")
            self.source_embedding = nn.Linear(text_embedding_dim, hid_dim)
            self.prefix_embedding = nn.Linear(text_embedding_dim, hid_dim)
            # Both source and prefix embeddings are added
            self.total_context_dim += 2 * hid_dim
            
        # Fusion layer to compress concatenated auxiliary features back to hid_dim
        if self.total_context_dim > 0:
            self.context_fusion_layer = nn.Linear(self.total_context_dim, hid_dim)
        
        # --- Transformer Backbone ---
        encoder_layers = TransformerEncoderLayer(hid_dim, nhead, hid_dim * 4, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, layers)

        # --- Output Heads ---
        # 1. Index Type Classification: 4 classes (None, Exact, HNSW, IVF_PQ)
        self.index_head = nn.Linear(hid_dim, 4)
        # 2. Continuous Parameters: k (normalized) and lambda, bounded to [0, 1] via Sigmoid
        self.k_lambda_head = nn.Linear(hid_dim, 2)
        
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initializes weights for linear layers to ensure stable training startup.
        Uses a uniform distribution within a small range.
        """
        init_range = 0.1
        for layer in [self.state_embedding, self.action_embedding]:
            layer.weight.data.uniform_(-init_range, init_range)
            layer.bias.data.zero_()
        
        self.index_head.weight.data.uniform_(-init_range, init_range)
        self.index_head.bias.data.zero_()
        self.k_lambda_head.weight.data.uniform_(-init_range, init_range)
        self.k_lambda_head.bias.data.zero_()

    def forward(
        self,
        S_t: torch.Tensor,  S_hist: torch.Tensor, A_hist: torch.Tensor, 
        H_dec_t: Optional[torch.Tensor] = None,
        Src_emb_t: Optional[torch.Tensor] = None,
        Pref_emb_t: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Policy Network.

        Args:
            S_t (torch.Tensor): Current state vector [Batch, S_DIM].
            S_hist (torch.Tensor): History of state vectors [Batch, HistoryLen, S_DIM].
            A_hist (torch.Tensor): History of action vectors [Batch, HistoryLen, ACTION_DIM].
            H_dec_t (Optional[torch.Tensor]): Current decoder hidden state [Batch, H_dec_Dim].
            Src_emb_t (Optional[torch.Tensor]): Source sentence embedding [Batch, Text_Dim].
            Pref_emb_t (Optional[torch.Tensor]): Generated prefix embedding [Batch, Text_Dim].
            src_key_padding_mask (Optional[torch.Tensor]): Mask for padding tokens in history.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - index_logits: Unnormalized scores for index types [Batch, 4].
                - k_lambda_values: Normalized (0-1) values for k and lambda [Batch, 2].
        """
        # 1. Process History: Embed states and actions, then sum them element-wise
        S_hist_emb = self.state_embedding(S_hist)
        A_hist_emb = self.action_embedding(A_hist)
        hist_seq = S_hist_emb + A_hist_emb # Shape: [Batch, HistoryLen, HID_DIM]
        
        # 2. Process Current State
        S_t_emb = self.state_embedding(S_t) # Shape: [Batch, 1, HID_DIM]
        
        # 3. Fuse Auxiliary Context if enabled
        if self.total_context_dim > 0:
            context_vectors = []
            if self.use_decoder_hidden_state and H_dec_t is not None:
                context_vectors.append(F.relu(self.decoder_hidden_embedding(H_dec_t)))
            
            if self.use_text_embeddings and Src_emb_t is not None and Pref_emb_t is not None:
                context_vectors.append(F.relu(self.source_embedding(Src_emb_t)))
                context_vectors.append(F.relu(self.prefix_embedding(Pref_emb_t)))
            
            if context_vectors:
                # Concatenate all available context features
                fused_context = torch.cat(context_vectors, dim=-1)
                # Project back to hidden dimension and add to the current state embedding
                context_emb = F.relu(self.context_fusion_layer(fused_context))
                S_t_emb = S_t_emb + context_emb
        
        # 4. Construct Sequence: [History, Current_State]
        full_seq = torch.cat([hist_seq, S_t_emb], dim=1) # Shape: [Batch, HistoryLen+1, HID_DIM]
        
        # 5. Apply Positional Encoding
        # PositionalEncoding expects [SeqLen, Batch, Dim], so we permute
        full_seq = full_seq.permute(1, 0, 2) # Shape: [HistoryLen+1, Batch, HID_DIM]
        full_seq = self.pos_encoder(full_seq)
        # Permute back for Transformer (batch_first=True)
        full_seq = full_seq.permute(1, 0, 2) # Shape: [Batch, HistoryLen+1, HID_DIM]
        
        # 6. Transformer Encoder Pass
        transformer_out = self.transformer_encoder(
            full_seq, 
            mask=None, 
            src_key_padding_mask=src_key_padding_mask
        ) # Shape: [Batch, HistoryLen+1, HID_DIM]
        
        # 7. Extract the embedding of the last token (representing the current state S_t)
        final_token_emb = transformer_out[:, -1, :] # Shape: [Batch, HID_DIM]

        # 8. Prediction Heads
        index_logits = self.index_head(final_token_emb) # Shape: [Batch, 4]
        # Use sigmoid to constrain k_norm and lambda to [0, 1]
        k_lambda_values = torch.sigmoid(self.k_lambda_head(final_token_emb)) # Shape: [Batch, 2]

        return index_logits, k_lambda_values
