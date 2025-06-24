import logging
from typing import List, Optional, Union
import transformers

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, is_valid_image
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    AddedToken,
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType
from configuration_trvlm import TrVLMConfig

logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(prompt, bos_token, image_seq_len, image_token):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<image>",
    )
    The output will be:
    "<image><image><image><s>Initial str"
    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
    """
    return f"{image_token * image_seq_len}{bos_token}{prompt}\n"


class TrVLMProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`PaliGemmaProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~PaliGemmaProcessor.__call__`] and [`~PaliGemmaProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast", "PreTrainedTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        
        # Check if image_processor has image_seq_length attribute, if not, calculate it
        if not hasattr(image_processor, "image_seq_length"):
            if hasattr(image_processor, "size") and hasattr(image_processor, "patch_size"):
                # Calculate number of patches based on image size and patch size
                image_processor.image_seq_length = (image_processor.size["height"] // image_processor.patch_size) ** 2
            else:
                # Default for CLIP/SigLIP with 224x224 image and 16x16 patches
                image_processor.image_seq_length = (224 // 16) ** 2  # 196 patches for standard size
                print(f"Warning: Setting default image_seq_length to {image_processor.image_seq_length}")

        self.image_seq_length = image_processor.image_seq_length

        # Önce pad_token'ı ayarla
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Pad token set to eos token in processor initialization")

        # Sonra image token'ı ekle        
        image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
        tokens_to_add = {"additional_special_tokens": [image_token]}
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        tokenize_newline_separately: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        do_resize: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional[str] = "channels_first",
        input_data_format: Optional[str] = None,
        resample = None,
        do_convert_rgb: bool = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_rescale: bool = None,
        suffix: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    ) -> BatchFeature:
        """

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            tokenize_newline_separately (`bool`, defaults to `True`):
                Adds a separately tokenized '\n' at the end of the prompt.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            suffix (`str`, `List[str]`, `List[List[str]]`):
                The suffixes or batch of suffixes to be encoded. Only necessary for finetuning. See https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md
                for more information. If your prompt is "<image> What is on the image", the suffix corresponds to the expected prediction "a cow sitting on a bench".

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
              is provided, the `input_ids` will also contain the suffix input ids.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **labels** -- Labels compatible with training if `suffix` is not None
        """

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments.")
        if text is None:
            logger.warning_once(
                "You are using TrVLM without a text prefix. It will perform as a picture-captioning model."
            )
            text = ""

        if isinstance(text, List) and isinstance(images, List):
            if len(images) < len(text):
                raise ValueError(
                    f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
                )
        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass
        if suffix is not None and _is_str_or_image(suffix):
            suffix = [suffix]
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

        input_strings = [
            build_string_from_input(
                prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=IMAGE_TOKEN,
            )
            for prompt in text
        ]

        pixel_values = self.image_processor(
            images,
            do_resize=do_resize,
            do_normalize=do_normalize,
            return_tensors=return_tensors,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
            data_format=data_format,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
        )["pixel_values"]

        if max_length is not None:
            max_length += self.image_seq_length  # max_length has to account for the image tokens

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_token_type_ids=return_token_type_ids,
        )

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    

if __name__ == '__main__':
    from transformers import AutoTokenizer
    import transformers
    config = TrVLMConfig()
    preprocessor = transformers.SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    preprocessor.image_seq_length = config.num_image_tokens
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    
    # Set pad token to be the same as eos token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    processor = TrVLMProcessor(tokenizer=tokenizer, image_processor=preprocessor)
    
    from PIL import Image
    import requests
    #processor = TrVLMProcessor.from_pretrained("./configFiles")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    prompt = "Bu resimdeki araba ne renk?"
    label = "bu resimdeki araba mavi renktedir"
    inputs = processor(text=prompt, suffix=label, images=image, return_tensors="pt", padding="max_length", max_length=512)
    for key, value in inputs.items():
        print(f"{key}: {value}")
    print(processor.decode(inputs.input_ids.tolist()[0]))    