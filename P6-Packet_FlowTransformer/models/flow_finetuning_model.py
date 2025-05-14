# models/flow_finetuning_model.py
import torch
import torch.nn as nn
from typing import List, Dict, Optional


class FlowFineTuningModel(nn.Module):
    """
    Model for fine-tuning on flow data, reusing a pre-trained Transformer encoder body.

    It consumes:
        - `flow_numerical_data`: FloatTensor [B, F_flow_num] (numerical flow features)
        - `flow_categorical_data`: LongTensor [B, F_flow_cat] (integer-coded categorical flow features)
                                     Each column corresponds to a different categorical feature.

    Numerical features are projected. Categorical features are embedded.
    These are combined and then treated as a sequence of length 1 for the transformer body.
    """

    def __init__(
            self,
            num_flow_numerical_features: int,
            flow_cat_cardinalities: List[int],
            d_model: int,
            pretrained_transformer_encoder_body: nn.Module,  # Moved up
            num_classes: int,
            classifier_dropout: float = 0.1  # Default param now after all non-default
    ):
        super().__init__()
        self.d_model = d_model

        # 1. New Input Layers for Flow Features
        # These will be trained from scratch during fine-tuning.

        # Project numerical flow features.
        # Option 1: Project numerical to d_model, and add summed categorical embeddings (also d_model).
        # Option 2: Project numerical to d_model/N, each categorical to d_model/N, then concatenate.
        # Let's go with a structure similar to your IoTTransformer: project numerical to d_model,
        # and each categorical also to d_model, then sum them.

        self.flow_numerical_projection = nn.Linear(num_flow_numerical_features, d_model)

        self.flow_categorical_embeddings = nn.ModuleList()
        if flow_cat_cardinalities:
            for cardinality in flow_cat_cardinalities:
                self.flow_categorical_embeddings.append(
                    nn.Embedding(num_embeddings=cardinality, embedding_dim=d_model)
                )

        # 2. Pre-trained Transformer Encoder Body
        # This is passed in, already initialized with pre-trained weights (and typically frozen).
        self.transformer_encoder_body = pretrained_transformer_encoder_body

        # 3. New Classification Head for Flows
        # Similar structure to your original IoTTransformer's classifier
        self.flow_classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),  # Example intermediate size
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

        # --- Process Flow Inputs ---
        # Project numerical features
        # Output shape: [Batch, d_model]
        processed_flow_features = self.flow_numerical_projection(flow_numerical_data)

        # Add categorical embeddings if they exist
        if flow_categorical_data is not None and self.flow_categorical_embeddings:
            if flow_categorical_data.shape[1] != len(self.flow_categorical_embeddings):
                raise ValueError(
                    f"Mismatch between number of categorical flow features provided ({flow_categorical_data.shape[1]}) "
                    f"and number of embedding layers ({len(self.flow_categorical_embeddings)}).")

            for i, emb_layer in enumerate(self.flow_categorical_embeddings):
                # Ensure codes are within embedding range (optional, good practice)
                # codes = flow_categorical_data[:, i].clamp(0, emb_layer.num_embeddings - 1)
                codes = flow_categorical_data[:, i]
                cat_emb = emb_layer(codes)  # Output shape: [Batch, d_model]
                processed_flow_features = processed_flow_features + cat_emb  # Add to the numerical projection

        # At this point, processed_flow_features is shape [Batch, d_model]
        # This represents the combined features for each flow.

        # --- Prepare for Transformer Body ---
        # Reshape to [Batch, SequenceLength=1, d_model]
        # The pre-trained transformer body expects a sequence.
        transformer_input_sequence = processed_flow_features.unsqueeze(1)

        # --- Pass through Pre-trained Transformer Body ---
        # The transformer body itself does not handle positional encoding or masks directly in its forward
        # if it's a standard nn.TransformerEncoder. These are typically applied before calling it.
        # For a sequence of length 1, a padding mask is usually not strictly necessary or would be all valid.
        # src_key_padding_mask = None (or torch.zeros(batch_size, 1, dtype=torch.bool))
        transformer_output = self.transformer_encoder_body(
            transformer_input_sequence)  # Potentially add mask if body expects it
        # Output shape: [Batch, 1, d_model]

        # --- Prepare for Classification ---
        # Take the output for the single "token" in our sequence of length 1
        # Squeeze out the sequence length dimension
        flow_representation = transformer_output.squeeze(1)  # Shape: [Batch, d_model]

        # --- Classification Head ---
        logits = self.flow_classifier(flow_representation)  # Shape: [Batch, num_classes]

        return logits
