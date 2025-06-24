# Colab için gerekli kurulumlar
# !pip install gradio peft transformers torch pillow

import torch
from peft import PeftModel
from modeling_trvlm import TrVLMForCausalLM
from PIL import Image
from transformers import AutoTokenizer
from processing_trvlm import TrVLMProcessor
from modeling_trvlm import TrVLMForCausalLM
from configuration_trvlm import TrVLMConfig
from transformers import SiglipImageProcessor
import gradio as gr

# Model yolları
full_checkpoint_path = "checkpoint_full/checkpoint-4000"
short_checkpoint_path = "short_captioning_checkpoint/checkpoint-4000"
base_model_path = "TrVLM_base"

def load_model_once():
    global cached_base_model, cached_processor, cached_tokenizer
    if 'cached_base_model' not in globals():
        print(f"Loading base model from: {base_model_path}")
        cached_base_model = TrVLMForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        config = TrVLMConfig()
        preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
        preprocessor.image_seq_length = config.num_image_tokens
        cached_tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")

        if cached_tokenizer.pad_token is None:
            cached_tokenizer.pad_token = cached_tokenizer.eos_token
            cached_tokenizer.pad_token_id = cached_tokenizer.eos_token_id

        cached_processor = TrVLMProcessor(tokenizer=cached_tokenizer, image_processor=preprocessor)
        print("Base model, tokenizer ve processor cached edildi.")
    else:
        print("Cached model, tokenizer ve processor kullanılıyor.")

def process_image_and_text(image, text, generation_method="sampling",
                           temperature=0.7, top_k=50, top_p=0.9, 
                           num_beams=5, max_new_tokens=256):
    try:
        if image is None:
            return "Lütfen bir görüntü yükleyin."
        if not text.strip():
            text = "Kısa açıkla."
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        test_text = [text.strip()]
        # "açıkla" ve türevlerini nokta işareti olsa bile yakala - tam eşitlik kontrolü
        words_to_check = ["açikla", "açıkla", "acıkla", "acikla", "açıkla.", "açıkla.", "acıkla.", "acikla."]
        is_short_request = text.lower().strip() in words_to_check or "kısa" in text.lower() or "kisa" in text.lower() or "özet" in text.lower()

        selected_checkpoint = short_checkpoint_path if is_short_request else full_checkpoint_path
        checkpoint_type = "short" if is_short_request else "full"

        model = PeftModel.from_pretrained(
            cached_base_model,
            selected_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model.eval()

        inputs = cached_processor(
            text=test_text,
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )

        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to('cuda')

        with torch.no_grad():
            if generation_method == "greedy":
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=cached_processor.tokenizer.pad_token_id,
                    eos_token_id=cached_processor.tokenizer.eos_token_id
                )
            elif generation_method == "beam":
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                    pad_token_id=cached_processor.tokenizer.pad_token_id,
                    eos_token_id=cached_processor.tokenizer.eos_token_id
                )
            else:  # sampling
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    length_penalty=1.0,
                    repetition_penalty=1.3,
                    pad_token_id=cached_processor.tokenizer.pad_token_id,
                    eos_token_id=cached_processor.tokenizer.eos_token_id
                )

        input_length = inputs["input_ids"].shape[1]
        generated_text = cached_processor.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        del model
        torch.cuda.empty_cache()

        return f"**Kullanılan Model:** {checkpoint_type} captioning\n**Üretim Yöntemi:** {generation_method}\n\n**Cevap:**\n{generated_text}"

    except Exception as e:
        return f"Hata oluştu: {str(e)}"

def create_gradio_interface():
    load_model_once()

    with gr.Blocks(title="TrVLM - Türkçe Vision Language Model", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🇹🇷 TrVLM - Türkçe Görüntü-Dil Modeli")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Görüntü Yükleyin", type="pil", height=400)
                text_input = gr.Textbox(label="Sorunuz", placeholder="Örnek: Görüntüde ne var?", lines=3)
                generation_method = gr.Radio(choices=["sampling", "greedy", "beam"], value="sampling", label="Üretim Yöntemi")

                with gr.Group(visible=True) as sampling_params:
                    temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
                    top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k")
                    top_p = gr.Slider(0.1, 1.0, value=0.9, label="Top-p")
                    max_tokens_sampling = gr.Slider(32, 512, value=256, step=32, label="Maximum Yeni Token Sayısı")

                with gr.Group(visible=False) as beam_params:
                    num_beams = gr.Slider(1, 10, value=5, step=1, label="Işın Sayısı (Beam)")
                    max_tokens_beam = gr.Slider(32, 512, value=256, step=32, label="Maximum Yeni Token Sayısı")

                def toggle_panels(method):
                    is_sampling = method == "sampling"
                    is_beam = method == "beam"
                    return {
                        sampling_params: gr.update(visible=is_sampling),
                        beam_params: gr.update(visible=is_beam),
                    }

                def process_with_correct_max_tokens(image, text, method, temp, topk, topp, beams, max_tokens_s, max_tokens_b):
                    # Doğru max_tokens değerini seç
                    max_tokens = max_tokens_s if method == "sampling" else max_tokens_b
                    return process_image_and_text(image, text, method, temp, topk, topp, beams, max_tokens)

                generation_method.change(fn=toggle_panels, inputs=generation_method, outputs=[sampling_params, beam_params])
                submit_btn = gr.Button("Gönder", variant="primary")
                clear_btn = gr.Button("Temizle", variant="secondary")

            with gr.Column():
                output = gr.Textbox(label="Model Cevabı", lines=15)

        submit_btn.click(
            fn=process_with_correct_max_tokens,
            inputs=[
                image_input, text_input, generation_method,
                temperature, top_k, top_p,
                num_beams, max_tokens_sampling, max_tokens_beam
            ],
            outputs=output
        )

        clear_btn.click(fn=lambda: (None, "", ""), outputs=[image_input, text_input, output])

        # Örnek kullanımlar bölümünü güncelle
        gr.Markdown("## 📝 Örnek Sorular ve Promptlar")
        gr.Markdown("""
        **Örnek promptlar:**
        - "Görseli kısaca özetle"
        - "Resmi Kısaca anlat"
        - "Bu görüntüde neler görüyorsun?"
        - "Görüntüyü detaylı bir şekilde açıkla"
        - "Bu fotoğraftaki ana öğeler neler?"
        """)

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True, server_name="0.0.0.0", server_port=7860)
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True, server_name="0.0.0.0", server_port=7860)
