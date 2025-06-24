import datasets
import argparse
import transformers
import os
import torch
import gc  # Add garbage collector
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
    DataCollatorForLanguageModeling, 
    DataCollatorForSeq2Seq, 
    AutoTokenizer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig
from processing_trvlm import TrVLMProcessor
from modeling_trvlm import TrVLMForCausalLM
from transformers import SiglipImageProcessor
from configuration_trvlm import TrVLMConfig

from PIL import Image
import numpy as np
#torch.set_num_threads(1)
#torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()  # Clear cache at start
from datasets import disable_caching
disable_caching()

# Memory management function
def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--save", type=str, default="base_checkpoint", help="save path")
parser.add_argument("--export", type=str, default="TrVLM_base", help="export path")
parser.add_argument("--epoch", type=int, default=1, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--test_size", type=float, default=0.005, help="test size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="gradient accumulation steps")
args = parser.parse_args()

# Create directories if they do not exist
os.makedirs(args.save, exist_ok=True)
os.makedirs(args.export, exist_ok=True)

def collate_fn(batch):
    texts = [example['text'][0]['value'] for example in batch]
    labels = [example['text'][1]['value'] for example in batch]
    images = []
    for example in batch:
        img = example["image"]
        
        # Convert PIL Image or numpy array to PIL Image if needed
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        # Explicitly convert to RGB mode (3 channels) - this handles both grayscale and RGBA
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to 224x224 if not already that size
        if img.size != (224, 224):
            img = img.resize((224, 224), Image.BICUBIC)
            
        images.append(img)
    
    # Process inputs using TrVLMProcessor
    tokens = processor(
        text=texts,
        images=images,
        suffix=labels,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=512
    )
    
    return tokens

if __name__ == "__main__":
    #torch.multiprocessing.set_start_method('spawn') # good solution !!!!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    from datasets import load_dataset

    print("Loading dataset...")
    #ds = load_dataset("emrecn/Predataset")

    ds = load_from_disk("/content/drive/MyDrive/ShortData/pretrain/train")
    ds.cleanup_cache_files()
    
    dataset = ds.train_test_split(test_size=args.test_size)
    train_ds = dataset['train']
    test_ds = dataset['test']
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Test dataset size: {len(test_ds)}")    # Initialize configuration
    config = TrVLMConfig()
    
    # Initialize processor
    preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    preprocessor.image_seq_length = config.num_image_tokens
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    processor = TrVLMProcessor(tokenizer=tokenizer, image_processor=preprocessor)
    print("Processor initialized")

    # Initialize model
    model = TrVLMForCausalLM(config)
    print("Model initialized")
    
    # Load pretrained models
    print("Loading pretrained models...")
    model.load_vision_model("google/siglip-base-patch16-224")
    print("Vision model loaded successfully")
    
    model.load_language_model("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    print("Language model loaded successfully")
    
    # Freeze the LLM and Vision Encoder
    for param in model.language_model.parameters():
        param.requires_grad = False

    for param in model.vision_tower.parameters():
        param.requires_grad = False
    
    # Enable training for vision projector
    for param in model.vision_projector.parameters():
        param.requires_grad = True

    # Move model to device
    model = model.to(device)
    print("Model moved to device")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.03,  # Reduced warmup for faster training
        max_grad_norm=1.0,
        
        bf16=True,  # Use bfloat16 instead of fp16 for A100 (better stability)
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=200,  # More frequent logging
        eval_strategy="steps",
        save_steps=10000,  # More frequent saves
        save_total_limit=5,  # Keep more checkpoints
        
        dataloader_num_workers=4,  # Increased for faster data loading
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,  # Prefetch data for faster loading
        gradient_checkpointing=True,
        eval_steps=400,  # More frequent evaluation
        
        #resume_from_checkpoint=args.save,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Additional optimizations for A100
        tf32=True,  # Enable TensorFloat-32 for faster training on A100
        dataloader_drop_last=True,  # Drop incomplete batches for consistent training
        ignore_data_skip=True,  # Skip data resume for faster restart
        report_to=None
    )
    
    print("Training arguments set")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    
    print("Trainer initialized")
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed")

    # Save model
    model.save_pretrained(args.export)
    processor.save_pretrained(args.export)
    print(f"Model and processor saved to {args.export}")