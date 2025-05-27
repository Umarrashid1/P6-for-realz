# models/flow_finetuning_model.py
import torch
import torch.nn as nn
from typing import List, Dict, Optional


class FlowFineTuningModel(nn.Module):
    def __init__(
            self,
            num_flow_numerical_features: int,
            flow_cat_cardinalities: List[int],
            d_model: int,
            pretrained_transformer_encoder_body: nn.Module,
            num_classes: int,
            classifier_dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.flow_numerical_projection = nn.Linear(num_flow_numerical_features, d_model)

        self.flow_categorical_embeddings = nn.ModuleList()
        if flow_cat_cardinalities:
            for cardinality in flow_cat_cardinalities:
                self.flow_categorical_embeddings.append(
                    nn.Embedding(num_embeddings=cardinality, embedding_dim=d_model)
                )


        self.transformer_encoder_body = pretrained_transformer_encoder_body


        self.flow_classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(
            self,
            flow_numerical_data: torch.Tensor,  # Shape: [Batch, NumFlowNumericalFeatures]
            flow_categorical_data: Optional[torch.Tensor] = None
            # Shape: [Batch, NumFlowCategoricalFeatures] (integer codes)
    ) -> torch.Tensor:


        # Project numerical features
        # Output shape: [Batch, d_model]
        processed_flow_features = self.flow_numerical_projection(flow_numerical_data)

        # Add categorical embeddings
        if flow_categorical_data is not None and self.flow_categorical_embeddings:
            if flow_categorical_data.shape[1] != len(self.flow_categorical_embeddings):
                raise ValueError(
                    f"Mismatch between number of categorical flow features provided ({flow_categorical_data.shape[1]}) "
                    f"and number of embedding layers ({len(self.flow_categorical_embeddings)}).")

            for i, emb_layer in enumerate(self.flow_categorical_embeddings):
                codes = flow_categorical_data[:, i]
                cat_emb = emb_layer(codes)  # Output shape: [Batch, d_model]
                processed_flow_features = processed_flow_features + cat_emb


        transformer_input_sequence = processed_flow_features.unsqueeze(1)

        transformer_output = self.transformer_encoder_body(
            transformer_input_sequence)

        flow_representation = transformer_output.squeeze(1)  # Shape: [Batch, d_model]

        logits = self.flow_classifier(flow_representation)  # Shape: [Batch, num_classes]

        return logits
