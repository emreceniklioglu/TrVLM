"""PyTorch TrVLM"""
import torch
from transformers import PreTrainedModel, AutoModel, AutoModelForCausalLM, GenerationMixin
from transformers.utils import logging, add_start_docstrings, ModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from torch import nn
from transformers.cache_utils import Cache, StaticCache

logger = logging.get_logger(__name__)

from configuration_trvlm import TrVLMConfig

_CONFIG_FOR_DOC = "TrVLMConfig"

@dataclass
class TrVLMCausalLMOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class TrVLMMultiModalProjector(nn.Module):
    def __init__(self, config: TrVLMConfig, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.vision_config.projection_dim, 4*config.vision_config.projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(4*config.vision_config.projection_dim, config.hidden_size, bias=True),
            nn.Dropout(dropout)
        )

    def forward(self, image_features):
        hidden_states = self.net(image_features).to(image_features.dtype)
        return hidden_states

class TrVLMPreTrainedModel(PreTrainedModel):
    config_class = TrVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TrVLMMultiModalProjector"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        # Do NOT init the weights of the model using this class call, this is a ported version, 
        # hence not intended to be trained from scratch.
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        return self.language_model._supports_sdpa

class TrVLMForCausalLM(TrVLMPreTrainedModel, GenerationMixin):
    def __init__(self, config: TrVLMConfig):
        super(TrVLMForCausalLM, self).__init__(config)
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = -1 if config.pad_token_id == None else config.pad_token_id
        self._attn_implementation = config._attn_implementation
        self.gradient_checkpointing = False

        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.vision_projector = TrVLMMultiModalProjector(config)

        language_model = AutoModelForCausalLM.from_config(
            config=config.text_config, attn_implementation=self._attn_implementation
        )
        if language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in language_model._tied_weights_keys]

        self.language_model = language_model
        self.post_init()
        
    def load_language_model(self, model_id = "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1"):
        language_model = AutoModelForCausalLM.from_pretrained(model_id)
        if language_model.vocab_size != self.vocab_size:
            print("vocab size mismatch, resize the token embeddings for the pretained language model")
            language_model.resize_token_embeddings(self.vocab_size)
        self.language_model.load_state_dict(language_model.state_dict(),strict=True)
        
    def load_vision_model(self,model_id = "google/siglip-base-patch16-224"):
        import transformers
        vision_model = transformers.SiglipVisionModel.from_pretrained(model_id)
        self.vision_tower.load_state_dict(vision_model.state_dict(),strict=True)
        
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)
    def get_decoder(self):
        return self.language_model.get_decoder()
    def tie_weights(self):
        return self.language_model.tie_weights()
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        # TODO: config.vocab_size is deprecated and will be removed in v4.43.
        # `resize_token_embeddings` should work from `modeling_utils.py``
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds
    def _update_causal_mask(
        self, attention_mask, token_type_ids, inputs_embeds, past_key_values, cache_position, is_training: bool = False
    ):
        using_static_cache = isinstance(past_key_values, StaticCache)
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = inputs_embeds.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
            if sequence_length != 1:
                if is_training:
                    causal_mask = torch.triu(causal_mask, diagonal=1)
                else:
                    causal_mask = torch.zeros_like(causal_mask)

        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_embeds.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
            # we are training thus we need to create a full mask on the image + prefix but causal on suffix
            if is_training:
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )
        return causal_mask
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TrVLMCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        is_training = token_type_ids is not None and labels is not None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        if pixel_values is not None:
            image_outputs = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
            selected_image_feature = image_outputs.last_hidden_state
            image_features = self.vision_projector(selected_image_feature)
            image_features = image_features / (self.config.hidden_size**0.5)

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. ",
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, inputs_embeds, past_key_values, cache_position, is_training
        )
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = outputs.logits
        logits = logits.float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TrVLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        model_inputs["token_type_ids"] = token_type_ids

        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1

        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs
    
if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    import transformers
    from transformers import AutoTokenizer
    from processing_trvlm import TrVLMProcessor
    config = TrVLMConfig()
    
    # Initialize processor
    preprocessor = transformers.SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    # image_seq_length otomatik olarak TrVLMProcessor içinde hesaplanacak
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    
    processor = TrVLMProcessor(tokenizer=tokenizer, image_processor=preprocessor)
    print("TrVLM modelini oluşturuluyor...")    
    model = TrVLMForCausalLM(config).to("cpu")
    print(model)
    
    # Modelleri yükle
    print("Loading language model...")
    model.load_language_model("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    
    print("Loading vision model...")
    model.load_vision_model("google/siglip-base-patch16-224")
    
    print("Models loaded successfully!")
    print("Model yükledikten sonra parametre sayısı:", count_parameters(model))
    # Test forward
    import torch
    from PIL import Image
    import requests

    # Load image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    # Define prompt and label
    prompt = "Bu resimi açıkla?"
    label = "bir araba var."

    # Process inputs
    inputs = processor(text=prompt, suffix=label, images=image, return_tensors="pt",
                  padding="max_length", max_length=512, truncation=True).to('cpu')
    
    # input_ids uzunluğunu kontrol edelim
    input_length = inputs["input_ids"].shape[1]
    print(f"Input length: {input_length}")
    
    # Image token'larının input_ids içinde olduğunu kontrol edelim
    image_token_count = (inputs["input_ids"] == config.image_token_index).sum().item()
    print(f"Image token count in input_ids: {image_token_count}")
    print(f"Expected image tokens: {preprocessor.image_seq_length}")
        
    # Generate - max_length yerine max_new_tokens kullanalım
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100,  # Sadece 100 yeni token ekle
        do_sample=True,
        temperature=0.7,  # Daha düşük sıcaklık
        top_p=0.9,  # Top-p örnekleme
        repetition_penalty=1.5,  # Tekrar cezası
        pad_token_id=processor.tokenizer.pad_token_id
    )
    print(processor.decode(outputs[0], skip_special_tokens=True))