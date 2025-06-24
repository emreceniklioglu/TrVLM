import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import SiglipImageProcessor
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
import requests
from io import BytesIO
from peft import PeftModel
import random

# Evaluation metrics
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from bert_score import score as bert_score
import nltk
# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# Local imports
from processing_trvlm import TrVLMProcessor
from modeling_trvlm import TrVLMForCausalLM
from configuration_trvlm import TrVLMConfig

# Define paths
full_checkpoint_path = "/content/drive/MyDrive/TrVLM/checkpoint_full/checkpoint-4000"
short_checkpoint_path = "/content/drive/MyDrive/TrVLM/short_captioning_checkpoint/checkpoint-4000"
base_model_path = "/content/drive/MyDrive/TrVLM/TrVLM_base"

def load_and_merge_models():
    """Load base model and merge short fine-tuned model only"""
    print(f"Loading base model from: {base_model_path}")
    base_model = TrVLMForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load and merge short model only
    print(f"Loading and merging short model from: {short_checkpoint_path}")
    short_peft_model = PeftModel.from_pretrained(
        base_model,
        full_checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    short_merged_model = short_peft_model.merge_and_unload()
    short_merged_model.eval()
    
    return short_merged_model

def setup_processor():
    """Setup processor and tokenizer"""
    config = TrVLMConfig()
    
    print("Setting up processor...")
    preprocessor = SiglipImageProcessor.from_pretrained("google/siglip-base-patch16-224")
    preprocessor.image_seq_length = config.num_image_tokens
    tokenizer = AutoTokenizer.from_pretrained("ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    processor = TrVLMProcessor(tokenizer=tokenizer, image_processor=preprocessor)
    return processor

def load_image_from_url_or_path(image_source):
    """Load image from URL or local path"""
    try:
        if isinstance(image_source, str) and (image_source.startswith('http') or image_source.startswith('https')):
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            if hasattr(image_source, 'convert'):
                image = image_source.convert('RGB')
            else:
                image = Image.open(image_source).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return Image.new('RGB', (224, 224), color='white')

def calculate_metrics(reference, candidate):
    """Calculate all evaluation metrics"""
    results = {}
    
    # ROUGE scores
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer_obj.score(reference, candidate)
    results.update({
        'rouge1': rouge_scores['rouge1'].fmeasure,
        'rouge2': rouge_scores['rouge2'].fmeasure,
        'rougeL': rouge_scores['rougeL'].fmeasure
    })
    
    # BLEU score
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
    results['bleu'] = bleu_score
    
    # METEOR score
    try:
        meteor = meteor_score([reference_tokens], candidate_tokens)
        results['meteor'] = meteor
    except:
        results['meteor'] = 0.0
    
    # CIDEr score
    try:
        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score({0: [reference]}, {0: [candidate]})
        results['cider'] = cider_score
    except:
        results['cider'] = 0.0
    
    return results

def calculate_bert_scores(references, candidates):
    """Calculate BERTScore for all predictions"""
    try:
        P, R, F1 = bert_score(candidates, references, lang='tr', verbose=False)
        return {
            'bert_precision': P.mean().item(),
            'bert_recall': R.mean().item(),
            'bert_f1': F1.mean().item()
        }
    except:
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0
        }

def evaluate_model(test_size=100, device='cuda'):
    """Evaluate the fine-tuned model using short model for all data"""
    
    # Load and merge short model only
    short_model = load_and_merge_models()
    processor = setup_processor()
    
    print("Loading test dataset...")
    try:
        dataset = load_dataset("ucsahin/Turkish-VLM-Mix-Benchmark", split="coco_qa_tr")
        print(f"Successfully loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    if test_size is not None:
        dataset = dataset.select(range(min(test_size, len(dataset))))
        print(f"Selected {len(dataset)} samples for evaluation")
    
    # Generation configurations
    beam_config = {
        'max_new_tokens': 64,
        'num_beams': 10,
        'do_sample': False,
        'length_penalty': 1.0,
        'pad_token_id': processor.tokenizer.pad_token_id,
        'eos_token_id': processor.tokenizer.eos_token_id
    }
    
    sampling_config = {
        'max_new_tokens': 64,
        'do_sample': True,
        'temperature': 0.6,
        'top_p': 0.8,
        'length_penalty': 1.0,
        'repetition_penalty': 1.3,
        'pad_token_id': processor.tokenizer.pad_token_id,
        'eos_token_id': processor.tokenizer.eos_token_id
    }
    
    # Results storage
    beam_results = {'predictions': [], 'references': [], 'questions': [], 'metrics': [], 'checkpoint_types': []}
    sampling_results = {'predictions': [], 'references': [], 'questions': [], 'metrics': [], 'checkpoint_types': []}
    
    print("Starting evaluation using full model for all data...")
    
    for i, example in enumerate(tqdm(dataset, desc="Evaluating with full Model")):
        try:
            image = load_image_from_url_or_path(example['image'])
            reference = example['label'].strip()  # Direct text format
            
            if not reference:
                continue
            
            # Use instruction from 'prompt' column
            instruction = example['prompt'].strip()
            
            # All samples use short model
            checkpoint_type = 'short'
            
            inputs = processor(
                text=[instruction],
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
                # Beam search generation
                beam_outputs = short_model.generate(**inputs, **beam_config)
                input_length = inputs["input_ids"].shape[1]
                beam_prediction = processor.tokenizer.decode(
                    beam_outputs[0][input_length:], skip_special_tokens=True
                ).strip()
                
                # Sampling generation
                sampling_outputs = short_model.generate(**inputs, **sampling_config)
                sampling_prediction = processor.tokenizer.decode(
                    sampling_outputs[0][input_length:], skip_special_tokens=True
                ).strip()
            
            # Calculate metrics for both methods
            beam_metrics = calculate_metrics(reference, beam_prediction)
            sampling_metrics = calculate_metrics(reference, sampling_prediction)
            
            # Store results
            beam_results['predictions'].append(beam_prediction)
            beam_results['references'].append(reference)
            beam_results.setdefault('questions', []).append(instruction)
            beam_results['metrics'].append(beam_metrics)
            beam_results['checkpoint_types'].append(checkpoint_type)
            
            sampling_results['predictions'].append(sampling_prediction)
            sampling_results['references'].append(reference)
            sampling_results.setdefault('questions', []).append(instruction)
            sampling_results['metrics'].append(sampling_metrics)
            sampling_results['checkpoint_types'].append(checkpoint_type)
            
            # Print first 5 examples and every 50 iterations
            if (i+1) <= 5 or (i+1) % 50 == 0:
                print(f"\n{'='*80}")
                print(f"EXAMPLE {i+1}")
                print(f"{'='*80}")
                print(f"Instruction: {instruction}")
                print(f"Reference: {reference}")
                print(f"Beam Search: {beam_prediction}")
                print(f"Sampling: {sampling_prediction}")
                print(f"\nROUGE Scores for this example:")
                print(f"  Beam Search - ROUGE-1: {beam_metrics['rouge1']:.4f}, ROUGE-2: {beam_metrics['rouge2']:.4f}, ROUGE-L: {beam_metrics['rougeL']:.4f}")
                print(f"  Sampling - ROUGE-1: {sampling_metrics['rouge1']:.4f}, ROUGE-2: {sampling_metrics['rouge2']:.4f}, ROUGE-L: {sampling_metrics['rougeL']:.4f}")
                print(f"{'='*80}\n")
                
            
        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue
    
    # Calculate BERTScores
    beam_bert_scores = calculate_bert_scores(beam_results['references'], beam_results['predictions'])
    sampling_bert_scores = calculate_bert_scores(sampling_results['references'], sampling_results['predictions'])
    
    # Calculate average metrics
    def calculate_averages(results, bert_scores):
        avg_metrics = {}
        for metric in ['rouge1', 'rouge2', 'rougeL', 'bleu', 'meteor', 'cider']:
            scores = [m[metric] for m in results['metrics']]
            avg_metrics[metric] = np.mean(scores)
        avg_metrics.update(bert_scores)
        return avg_metrics
    
    beam_avg_metrics = calculate_averages(beam_results, beam_bert_scores)
    sampling_avg_metrics = calculate_averages(sampling_results, sampling_bert_scores)
    
    # Count checkpoint types (all will be 'short' now)
    full_count = 0
    short_count = len(beam_results['checkpoint_types'])
    
    # Print results
    print("\n" + "="*60)
    print("SHORT MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Test samples: {len(beam_results['predictions'])}")
    print(f"All samples used short model: {len(beam_results['checkpoint_types'])}")
    
    print(f"\n--- BEAM SEARCH RESULTS (Overall) ---")
    for metric, score in beam_avg_metrics.items():
        print(f"{metric.upper()}: {score:.4f}")
    
    print(f"\n--- SAMPLING RESULTS (Overall) ---")
    for metric, score in sampling_avg_metrics.items():
        print(f"{metric.upper()}: {score:.4f}")
    
    # Prepare comprehensive results
    final_results = {
        'test_samples': len(beam_results['predictions']),
        'checkpoint_distribution': {
            'full': full_count,
            'short': short_count
        },
        'beam_search': {
            'overall_metrics': beam_avg_metrics,
            'examples': [
                {
                    'question': beam_results['questions'][i],
                    'reference': beam_results['references'][i],
                    'prediction': beam_results['predictions'][i],
                    'checkpoint_type': beam_results['checkpoint_types'][i],
                    **beam_results['metrics'][i]
                }
                for i in range(len(beam_results['predictions']))
            ]
        },
        'sampling': {
            'overall_metrics': sampling_avg_metrics,
            'examples': [
                {
                    'question': sampling_results['questions'][i],
                    'reference': sampling_results['references'][i],
                    'prediction': sampling_results['predictions'][i],
                    'checkpoint_type': sampling_results['checkpoint_types'][i],
                    **sampling_results['metrics'][i]
                }
                for i in range(len(sampling_results['predictions']))
            ]
        }
    }
    
    # Save results
    results_file = f"evaluation_results_vqa_only.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    return final_results

def main():
    test_size = 1000 # Set to None for all samples
    device = "cuda"
    
    evaluate_model(test_size, device)

if __name__ == "__main__":
    main()
