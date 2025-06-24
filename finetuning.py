from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig
import argparse
import wandb
import transformers
import datasets
import os
from processing_trvlm import TrVLMProcessor
from modeling_trvlm import TrVLMForCausalLM
from transformers import SiglipImageProcessor

from datasets import disable_caching
disable_caching()

# Transformers ve diğer kütüphanelerden gelen uyarıları kapat
import warnings
import logging
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
logging.getLogger("datasets").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, default="checkpoint_full", help="save path")
parser.add_argument("--export", type=str, default="TrVLM_finetuning_full", help="export path")
parser.add_argument("--epoch", type=int, default=1, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=8, help="batch size for training")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--test_size", type=float, default=0.005, help="test size")
parser.add_argument("--pretrained_model", type=str, default="TrVLM_base", help="path to pretrained model")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="path to checkpoint to resume from, 'latest' for most recent, or None to start from scratch")
args = parser.parse_args()

os.makedirs(args.save, exist_ok=True)
os.makedirs(args.export, exist_ok=True)

from configuration_trvlm import TrVLMConfig

config = TrVLMConfig()
preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
preprocessor.image_seq_length = config.num_image_tokens
tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")

# Set pad token if not exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

processor = TrVLMProcessor(tokenizer=tokenizer, image_processor=preprocessor)

def collate_fn(batch):
    all_inputs = []
    
    for example in batch:
        image = example["image"]
        conversations = example["text"]  # This contains the list with "from" and "value" keys
        
        # Conversations listesinde "from": "human" ve "from": "gpt" çiftlerini bul
        for i in range(len(conversations)):
            # Human mesajını bul
            if conversations[i]["from"] == "human":
                human_msg = conversations[i]["value"]
                
                # Bir sonraki GPT cevabını bul
                if i + 1 < len(conversations) and conversations[i + 1]["from"] == "gpt":
                    gpt_msg = conversations[i + 1]["value"]
                    
                    all_inputs.append({
                        "text": human_msg,
                        "label": gpt_msg,
                        "image": image
                    })
    
    # Eğer hiç veri bulunamadıysa, batch'i atla
    if not all_inputs:
        print("Uyarı: Bu batch'te işlenebilir veri bulunamadı!")
        return None
    
    texts = [item["text"] for item in all_inputs]
    labels = [item["label"] for item in all_inputs]
    images = [item["image"].convert("RGB") for item in all_inputs]
    
    # Print batch size information
    
    tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest",
                    tokenize_newline_separately=False)

    tokens = tokens.to(torch.float16)
    return tokens

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load datasets from HuggingFace Hub
    print("Loading datasets from HuggingFace Hub...")
    
    # Load short dataset and sample 300k examples
    print("Loading short dataset...")
    ds_short = load_dataset("emrecn/newveri")
    if len(ds_short['train']) > 300000:
        ds_short = ds_short['train'].shuffle(seed=42).select(range(300000))
        # Convert back to DatasetDict format
        ds_short = datasets.DatasetDict({'train': ds_short})
    print(f"Short dataset: {len(ds_short['train'])} examples")
    
    # Load long dataset (all examples)
    print("Loading long dataset...")
    ds_long = load_dataset("emrecn/longdataset")
    print(f"Long dataset: {len(ds_long['train'])} examples")
    
    # Load VQA dataset and sample 150 examples
    print("Loading VQA dataset...")
    ds_vqa_raw = load_dataset("berkanbucak/finetunevqa")
    if len(ds_vqa_raw['train']) > 150000:
        ds_vqa_selected = ds_vqa_raw['train'].shuffle(seed=42).select(range(150000))
    else:
        ds_vqa_selected = ds_vqa_raw['train']
    
    # Rename 'conversations' column to 'text' for VQA dataset
    print("Renaming VQA dataset column...")
    ds_vqa_selected = ds_vqa_selected.rename_column("conversations", "text")
    ds_vqa = datasets.DatasetDict({'train': ds_vqa_selected})
    print(f"VQA dataset: {len(ds_vqa['train'])} examples")
    
    # Clean up cache
    ds_short.cleanup_cache_files()
    ds_long.cleanup_cache_files()
    ds_vqa.cleanup_cache_files()

    # Split dataset into train and test sets
    # Split each dataset into train and test sets
    print("Splitting datasets...")
    short_dataset = ds_short['train'].train_test_split(test_size=args.test_size, seed=42)
    long_dataset = ds_long['train'].train_test_split(test_size=args.test_size, seed=42)
    vqa_dataset = ds_vqa['train'].train_test_split(test_size=args.test_size, seed=42)
    
    # Combine train datasets and test datasets
    print("Concatenating datasets...")
    train_ds = concatenate_datasets([
        short_dataset['train'],
        long_dataset['train'],
        vqa_dataset['train']
    ])
    
    test_ds = concatenate_datasets([
        short_dataset['test'],
        long_dataset['test'],
        vqa_dataset['test']
    ])
    
    # Shuffle the datasets
    print("Shuffling datasets...")
    train_ds = train_ds.shuffle(seed=42)
    test_ds = test_ds.shuffle(seed=42)
    
    print(f"Final combined training dataset: {len(train_ds)} examples")
    print(f"Final combined testing dataset: {len(test_ds)} examples")
    print(f"Dataset composition:")
    print(f"  - Short captions: ~{len(short_dataset['train'])} train examples")
    print(f"  - Long descriptions: ~{len(long_dataset['train'])} train examples") 
    print(f"  - VQA examples: ~{len(vqa_dataset['train'])} train examples")

    # Load pretrained model and apply LoRA
    print(f"Loading pretrained model from: {args.pretrained_model}")
    model = TrVLMForCausalLM.from_pretrained(
        args.pretrained_model, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # BF16 kullan
        device_map=None  # Single GPU için
    )
    
    # Check model weights for NaN before training
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"Warning: NaN detected in {name} before training!")
            param.data = torch.nan_to_num(param.data, nan=0.0)

    lora_config = LoraConfig(
        r=4,  # Daha küçük rank
        lora_alpha=8,  # Daha küçük alpha
        use_rslora=True,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.1,  # Dropout ekledik
    )
    
    model = get_peft_model(model, lora_config)
    
    # Enable full parameter training for vision projector
    for param in model.vision_projector.parameters():
        param.requires_grad = True

    model = model.to(device)
    print("Trainable parameters:")
    model.print_trainable_parameters()
    
    data_collator = collate_fn
    training_args = TrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,  # Parametre kullanılıyor
        per_device_eval_batch_size=args.batch_size,   # Parametre kullanılıyor
        learning_rate=args.lr,  # Parametre kullanılıyor
        weight_decay=1e-6,  # Çok düşük weight decay
        warmup_ratio=0.05,   # Çok düşük warmup
        max_grad_norm=0.3,  # Çok sıkı gradient clipping
        
        # Uyarıları ve gereksiz çıktıları kapat
        disable_tqdm=False,  # Progress bar'ı kapat
        log_level="error",  # Sadece hataları göster
        report_to="none",   # Wandb vb. raporlamayı kapat
        
        fp16=False,  # A100'de BF16 daha iyi
        bf16=True,   # A100 için BF16 kullan, hız artışı sağlar
        gradient_accumulation_steps=8,  # Daha büyük effective batch size
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=100,  # Daha sık loglama
        eval_strategy="steps",
        load_best_model_at_end=True,
        save_steps=2000,   # Daha sık kaydetme
        save_total_limit=3,
        
        eval_accumulation_steps=1,
        dataloader_num_workers=4,  # A100 için artırıldı
        dataloader_pin_memory=True,  # A100'de pin memory aktif
        gradient_checkpointing=True,
        eval_steps=200,    # Daha az sıklıkta evaluation (hız artışı)
        
        # Kararlılık için ek parametreler
        optim="adamw_torch",
        adam_epsilon=1e-8,
        lr_scheduler_type="cosine",
        save_safetensors=True,
        dataloader_drop_last=True,  # Batch size tutarlılığı için
        
        # Hızlandırma için ek optimizasyonlar
        tf32=True,  # A100'de TensorFloat-32 kullan
        dataloader_persistent_workers=True,  # Worker'ları yeniden kullan
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
    )
    
    # Model durumunu kontrol et
    print("Checking model state before training...")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in {name}!")
        if torch.isinf(param).any():
            print(f"Inf detected in {name}!")
    
    # İlk batch'i test et
    print("Testing first batch...")
    try:
        first_batch = next(iter(trainer.get_train_dataloader()))
        print("First batch processed successfully")
        print(f"Batch keys: {first_batch.keys()}")
        for key, value in first_batch.items():
            if torch.is_tensor(value):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if torch.isnan(value).any():
                    print(f"WARNING: NaN in {key}")
                if torch.isinf(value).any():
                    print(f"WARNING: Inf in {key}")
    except Exception as e:
        print(f"Error with first batch: {e}")
        raise e
    
    print("Starting fine-tuning...")
    if args.resume_from_checkpoint == 'latest':
        print(f"Resuming from latest checkpoint in {args.save}...")
        trainer.train(resume_from_checkpoint=True)
    elif args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        print("Training from scratch...")
        trainer.train()
    
    print("Fine-tuning complete, merging model...")
    merge_model = model.merge_and_unload()
    merge_model.save_pretrained(args.export)
    print(f"Fine-tuned model saved to: {args.export}")