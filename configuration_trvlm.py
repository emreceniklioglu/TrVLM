from transformers import PretrainedConfig
from transformers import logging, CONFIG_MAPPING
import warnings
import transformers

logger = logging.get_logger(__name__)

class TrVLMConfig(PretrainedConfig):
    model_type = "trvlm"
    is_composition = False

    def __init__(
            self,
            vision_config=None,
            text_config=None,
            ignore_index=-100,
            image_token_idx=128256,
            vocab_size=128257,
            projection_dim=768,
            hidden_size=4096,
            **kwargs, 
    ):
        self.ignore_index = ignore_index
        self.image_token_index = image_token_idx
        self._vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        
        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "siglip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                attention_dropout=0.0,
                hidden_act="gelu_pytorch_tanh",
                hidden_size=768,
                image_size=224,
                intermediate_size=3072,
                layer_norm_eps=1e-06,
                num_attention_heads=12,
                num_channels=3,
                num_hidden_layers=12,
                patch_size=16,
            )
        
        self.vocab_size = vocab_size
        self.text_config = text_config

        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "gpt2"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"](
                architecture= ["LlavaLlamaForCausalLM"],
                hidden_act = "silu",
                attention_bias = False,
                attention_dropout = 0.0,
                bos_token_id = 128000,
                eos_token_id = 128009,
                hidden_size = 4096,
                initializer_range = 0.02,
                intermediate_size = 14336,
                max_position_embeddings = 8192,
                model_type = "llava_llama",
                num_attention_heads = 32,
                num_hidden_layers = 32,
                num_key_value_heads = 8,
                pad_token_id = 128256,
                pretraining_tp = 1,
                rms_norm_eps = 1e-05,
                rope_scaling = None,
                rope_theta = 500000.0,
                tie_word_embeddings = False,
                torch_dtype = "bfloat16",
                transformers_version = "4.37.2",
                use_cache = True,
                vocab_size = 128257
            )
        self.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.pad_token_id = self.text_config.pad_token_id
        self.vision_config.projection_dim = projection_dim
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        warnings.warn(
            "The `vocab_size` attribute is deprecated and will be removed in v4.44, Please use `text_config.vocab_size` instead.",
            FutureWarning,
        )
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value):
        self._vocab_size = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_vocab_size", None)
        return output
