import torch
import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from processing_trvlm import TrVLMProcessor
from modeling_trvlm import TrVLMForCausalLM
from transformers import SiglipImageProcessor
from configuration_trvlm import TrVLMConfig
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser(description="Short Captioning Fine-tuning with LoRA")
    parser.add_argument("--save", type=str, default="short_captioning_checkpoint", help="checkpoint save path")
    parser.add_argument("--export", type=str, default="TrVLM_short_captioning", help="final model export path")
    parser.add_argument("--pretrained_model", type=str, default="TrVLM_base", help="path to pretrained base model")
    parser.add_argument("--epoch", type=int, default=1, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="per device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--test_size", type=float, default=0.01, help="test split ratio")
    parser.add_argument("--max_length", type=int, default=256, help="max sequence length for short captions")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    return parser.parse_args()

def create_short_captioning_lora_config(args):
    """Short captioning için özel LoRA konfigürasyonu"""
    return LoraConfig(
        r=args.lora_r,  # Rank - short captioning için optimize
        lora_alpha=args.lora_alpha,  # Alpha değeri
        use_rslora=True,  # Rank-stabilized LoRA
        target_modules=[
            # Language model attention layers
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=args.lora_dropout,
        bias="none",  # Bias'ı dondurup sadece weights'i adapte et
        inference_mode=False,
    )

def create_short_captioning_collate_fn(processor, max_length=256):
    """Sade collate function"""
    def collate_fn(batch):
        texts = []
        labels = []
        images = []
        
        for example in batch:
            try:
                image = example["image"]
                conversations = example["text"]
                
                # Human ve GPT mesajlarını bul
                human_msg = None
                gpt_msg = None
                
                for conv in conversations:
                    if conv["from"] == "human":
                        human_msg = conv["value"]
                    elif conv["from"] == "gpt":
                        gpt_msg = conv["value"]
                
                # Orijinal mesajları kullan
                if human_msg and gpt_msg and len(gpt_msg.strip()) > 0:
                    texts.append(human_msg.strip())
                    labels.append(gpt_msg.strip())
                    images.append(image)
                        
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
        
        # Process inputs using TrVLMProcessor
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        )
        
        return tokens
    
    return collate_fn

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(args.export, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize configuration and processor
    print("Initializing configuration and processor...")
    config = TrVLMConfig()
    
    preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    preprocessor.image_seq_length = config.num_image_tokens
    
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    processor = TrVLMProcessor(tokenizer=tokenizer, image_processor=preprocessor)
    print("Processor initialized successfully")
    
    # Load dataset
    print("Loading dataset for short captioning...")
    try:
        dataset = load_dataset("emrecn/newveri")
        print(f"Dataset loaded: {len(dataset['train'])} examples")
        
        # Split dataset
        split_dataset = dataset['train'].train_test_split(test_size=args.test_size, seed=42)
        train_ds = split_dataset['train'].shuffle(seed=42)
        test_ds = split_dataset['test'].shuffle(seed=42)
        
        print(f"Training examples: {len(train_ds)}")
        print(f"Test examples: {len(test_ds)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Load pretrained model
    print(f"Loading pretrained model from: {args.pretrained_model}")
    model = TrVLMForCausalLM.from_pretrained(
        args.pretrained_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    
    # Apply LoRA configuration for short captioning
    print("Applying LoRA configuration for short captioning...")
    lora_config = create_short_captioning_lora_config(args)
    model = get_peft_model(model, lora_config)
    
    # Vision projector'ı tamamen eğitilebilir yap (captioning için kritik)
    for param in model.base_model.vision_projector.parameters():
        param.requires_grad = True
    
    model = model.to(device)
    
    print("LoRA configuration applied successfully")
    print("Trainable parameters:")
    model.print_trainable_parameters()
    
    # Create data collator
    data_collator = create_short_captioning_collate_fn(processor, args.max_length)
    
    # Training arguments optimized for short captioning
    training_args = TrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,  # Short captioning için daha uzun warmup
        max_grad_norm=1.0,
        
        # Precision and optimization
        bf16=True,
        tf32=True,
        gradient_accumulation_steps=4,  # Effective batch size artırımı
        gradient_checkpointing=True,
        
        # Logging and evaluation
        logging_strategy="steps",
        logging_steps=200,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=800,
        save_steps=4000,
        save_total_limit=3,
        
        # Performance optimizations
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Stability
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        save_safetensors=True,
        
        # Reporting
        report_to="none",
        disable_tqdm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    
    # Pre-training checks
    print("Performing pre-training checks...")
    try:
        # Test first batch
        first_batch = next(iter(trainer.get_train_dataloader()))
        print("First batch processed successfully")
        
        # Check for NaN/Inf in model parameters
        nan_params = []
        inf_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
        
        if nan_params:
            print(f"Warning: NaN detected in parameters: {nan_params}")
        if inf_params:
            print(f"Warning: Inf detected in parameters: {inf_params}")
            
    except Exception as e:
        print(f"Error in pre-training checks: {e}")
        raise
    
    # Start training
    print("Starting short captioning fine-tuning...")
    trainer.train()
    print("Training completed successfully!")
    
    # Merge and save model
    print("Merging LoRA weights and saving final model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.export)
    processor.save_pretrained(args.export)
    
    print(f"Short captioning model saved to: {args.export}")
    print("Fine-tuning process completed!")

if __name__ == "__main__":
    main()
