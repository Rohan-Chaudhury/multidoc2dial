import os
import json
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from transformers import (BertForMaskedLM, 
                            BertTokenizer, 
                            BertTokenizerFast, 
                            DistilBertForMaskedLM, 
                            DistilBertPreTrainedModel, 
                            PreTrainedModel, 
                            PretrainedConfig, 
                            AutoModel, 
                            AutoConfig,
                            AutoTokenizer)
from transformers.modeling_outputs import ModelOutput

from collections import OrderedDict
from typing import Mapping

from transformers import DistilBertConfig
from transformers.onnx import OnnxConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)

class SpladeConfig(PretrainedConfig):
    model_type = "splade"
    attribute_map = {
        "hidden_size": "dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
    }

    def __init__(
        self,
        vocab_size=30522,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=4 * 768,
        dropout=0.1,
        attention_dropout=0.1,
        activation="gelu",
        initializer_range=0.02,
        qa_dropout=0.1,
        seq_classif_dropout=0.2,
        pad_token_id=0,
        expansion_pooling='max',    
        **kwargs
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.sinusoidal_pos_embds = sinusoidal_pos_embds
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.initializer_range = initializer_range
        self.qa_dropout = qa_dropout
        self.seq_classif_dropout = seq_classif_dropout
        self.expansion_pooling = expansion_pooling,
        self.tokenizer_class = "BertTokenizer"

class SpladeOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

@dataclass
class SpladeOutput(ModelOutput):
    """
    Class for outputs of :class:`~transformers.DPRQuestionEncoder`.

    Args:
        pooler_output: (:obj:``torch.FloatTensor`` of shape ``(batch_size, embeddings_size)``):
            The DPR encoder outputs the `pooler_output` that corresponds to the context representation. Last layer
            hidden-state of the first token of the sequence (classification token) further processed by a Linear layer.
            This output is to be used to embed contexts for nearest neighbors queries with questions embeddings.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    pooler_output: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Splade_Pooling(nn.Module):
    def __init__(self, word_embedding_dimension: int):
        super(Splade_Pooling, self).__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.config_keys = ["word_embedding_dimension"]

    def __repr__(self):
        return "Pooling Splade({})"

    def get_pooling_mode_str(self) -> str:
        return "Splade"

    def forward(self, token_embeddings, attention_mask):

        ## Pooling strategy
        sentence_embedding = torch.max(torch.log(1 + torch.relu(token_embeddings)) * attention_mask.unsqueeze(-1), dim=1).values
        return sentence_embedding

    def get_sentence_embedding_dimension(self):
        return self.word_embedding_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Splade_Pooling(**config)

class SpladeModel(PreTrainedModel):

    config_class = SpladeConfig
    base_model_prefix = "splade"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def __init__(self, config):
        super().__init__(config)

        self.auto_model = DistilBertForMaskedLM(config)
        self.pooling = Splade_Pooling(config.vocab_size)
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Return:

        Examples::

            >>> from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
            >>> tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
            >>> input_ids = tokenizer("Hello, is my dog cute ?", return_tensors='pt')["input_ids"]
            >>> embeddings = model(input_ids).pooler_output
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = (
                torch.ones(input_shape, device=device)
                if input_ids is None
                else (input_ids != self.config.pad_token_id)
            )
        # if token_type_ids is None:
        #     token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        outputs = self.auto_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = outputs.logits
        pooled_output = self.pooling(last_hidden_state, attention_mask)

        if not return_dict:
            return (pooled_output,) + outputs[1:]
        return SpladeOutput(
            pooler_output=pooled_output, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )

# register splade into automodel
AutoConfig.register("splade", SpladeConfig)
AutoModel.register(SpladeConfig, SpladeModel)
AutoTokenizer.register(SpladeConfig, slow_tokenizer_class=BertTokenizer, fast_tokenizer_class=BertTokenizerFast)