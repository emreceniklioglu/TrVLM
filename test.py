import torch
from peft import PeftModel
from modeling_trvlm import TrVLMForCausalLM
import torch
import argparse
from PIL import Image
import requests
from transformers import AutoTokenizer
from processing_trvlm import TrVLMProcessor
from modeling_trvlm import TrVLMForCausalLM
from configuration_trvlm import TrVLMConfig
from transformers import SiglipImageProcessor

# Define arguments directly as variables
full_checkpoint_path = "checkpoint_full/checkpoint-4000"
short_checkpoint_path = "short_captioning_checkpoint/checkpoint-4000"
base_model_path = "TrVLM_base"

# Cache variables - only load once
if 'cached_base_model' not in globals():
    print(f"Loading base model from: {base_model_path}")
    cached_base_model = TrVLMForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    config = TrVLMConfig()
    
    # Processor'ı manuel olarak oluştur
    print("Processor oluşturuluyor...")
    preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    preprocessor.image_seq_length = config.num_image_tokens
    cached_tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    
    # Pad token ayarla
    if cached_tokenizer.pad_token is None:
        cached_tokenizer.pad_token = cached_tokenizer.eos_token
        cached_tokenizer.pad_token_id = cached_tokenizer.eos_token_id
    
    cached_processor = TrVLMProcessor(tokenizer=cached_tokenizer, image_processor=preprocessor)
    print("Base model, tokenizer ve processor cached edildi.")
else:
    print("Cached model, tokenizer ve processor kullanılıyor.")

url = "https://md.teyit.org/2024/07/kuran-yirtan-kiz-iddiasi-kapak-liste.webp"

# Test text to determine which checkpoint to use
test_text = ["bu fotoğrafı genel olarak açıkla ve sence bu insana ne oldu?"]

# Check if the text asks for short summary
is_short_request = any(text.lower() in ["açikla","açıkla", "acıkla", "acikla"] or "kısa" in text.lower() or "kisa" in text.lower() or "özet" in text.lower() for text in test_text)

# Select appropriate checkpoint
selected_checkpoint = short_checkpoint_path if is_short_request else full_checkpoint_path
print(f"Using checkpoint: {'short' if is_short_request else 'full'}")

print(f"Loading LoRA weights from: {selected_checkpoint}")
model = PeftModel.from_pretrained(
    cached_base_model,
    selected_checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("Merging model...")
merged_model = model.merge_and_unload()


merged_model.eval()

image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
print(f"Görüntü yüklendi.")
inputs = cached_processor(
        text=test_text,  # Use the test_text variable
        images=[image],      # Liste olarak ver
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

# Girdiyi cihaza taşı
for key in inputs:
    if isinstance(inputs[key], torch.Tensor):
        inputs[key] = inputs[key].to('cuda')

# Çıktıyı oluştur
print("Çıktı oluşturuluyor...")

with torch.no_grad():
    # Farklı generation parametreleri ile test et
    # 1. Greedy decoding
    outputs_greedy = merged_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        #early_stopping=True,
        length_penalty=1.0,
        pad_token_id=cached_processor.tokenizer.pad_token_id,
        eos_token_id=cached_processor.tokenizer.eos_token_id
    )

    # 2. Beam search
    outputs_beam = merged_model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=5,
        do_sample=False,
        #early_stopping=True,
        length_penalty=1.0,
        pad_token_id=cached_processor.tokenizer.pad_token_id,
        eos_token_id=cached_processor.tokenizer.eos_token_id
    )

    # 3. Sampling with temperature
    outputs_sampling = merged_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.8,
        length_penalty=1.0,
        #early_stopping=True,
        repetition_penalty=1.3,
        pad_token_id=cached_processor.tokenizer.pad_token_id,
        eos_token_id=cached_processor.tokenizer.eos_token_id
    )

# Input uzunluğunu al
input_length = inputs["input_ids"].shape[1]

# Sadece yeni üretilen tokenleri decode et
greedy_text = cached_processor.tokenizer.decode(outputs_greedy[0][input_length:], skip_special_tokens=True)
beam_text = cached_processor.tokenizer.decode(outputs_beam[0][input_length:], skip_special_tokens=True)
sampling_text = cached_processor.tokenizer.decode(outputs_sampling[0][input_length:], skip_special_tokens=True)

print("Prompt =",test_text[0])
# Sonuçları yazdır
print("\n=== Greedy Decoding ===")
print(greedy_text)

print("\n=== Beam Search (5 beam) ===")
print(beam_text)

print("\n=== Sampling (temperature=0.7, top_p=0.9) ===")
print(sampling_text)