import argparse
import json
import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import keras
import keras_hub
import tensorflow as tf
from dotenv import load_dotenv
from google.generativeai.generative_models import GenerativeModel
from keras_hub.models import Gemma3CausalLM
from tqdm import tqdm

from utils import LoadedDataset, generate_from_model, truncate_sample

# Try to import lz4, fall back to gzip if not available
try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    import gzip

    HAS_LZ4 = False
    print("Warning: lz4 not installed. Using gzip (slower)")

# Load environment variables for API access
load_dotenv()


def create_temp_dir() -> str:
    """Create a temporary directory for model storage. Returns the temp path."""
    try:
        # Use ramdisk (/dev/shm) if it exists, otherwise fall back to system temp
        ramdisk_path = "/dev/shm"
        if os.path.exists(ramdisk_path) and os.access(ramdisk_path, os.W_OK):
            temp_dir = tempfile.mkdtemp(prefix="weights_temp_", dir=ramdisk_path)
            print(f"Created temporary directory at {temp_dir} (using ramdisk)")
            return temp_dir
        else:
            temp_dir = tempfile.mkdtemp(prefix="weights_temp_")
            print(f"Created temporary directory at {temp_dir} (using system temp)")
            return temp_dir

    except Exception as e:
        print(f"Failed to create temp directory: {e}")
        # Final fallback to system temp
        return tempfile.mkdtemp(prefix="weights_temp_")


def cleanup_temp_dir(temp_path: str):
    """Clean up the temporary directory."""
    try:
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        print(f"Cleaned up temporary directory at {temp_path}")

    except Exception as e:
        print(f"Warning: Failed to cleanup temp directory: {e}")
        # Try to remove directory anyway
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path, ignore_errors=True)


def save_model_with_compression(model, output_path: str):
    """Save model weights using temp directory, compression, and final move."""
    print(f"Saving model weights with compression to {output_path}")

    # Decide on compression method and extension
    if HAS_LZ4:
        compression_method = "lz4"
        compression_ext = ".lz4"
        print("Using LZ4 compression (fastest)")
    else:
        compression_method = "gzip"
        compression_ext = ".gz"
        print("Using gzip compression (slower)")

    # Create temp directory
    temp_path = create_temp_dir()
    start_time = time.time()

    try:
        model.save(hf_path := "hf://Neel-Gupta/pomni")
        print(f"Model uploaded to HF @ " + hf_path)

        # Save to temp directory first
        temp_weights_path = os.path.join(temp_path, "temp_model.weights.h5")
        print(f"Saving weights to temp directory: {temp_weights_path}")

        # Cast model weights to bfloat16 to reduce file size
        print("Converting model weights to bfloat16 for storage...")

        for layer in model.layers:
            if hasattr(layer, "weights") and layer.weights:
                weights = layer.get_weights()
                new_weights = [w.astype("bfloat16") for w in weights]
                layer.set_weights(new_weights)

        print("Weight conversion to bfloat16 completed")

        save_start = time.time()
        model.save_weights(temp_weights_path)
        save_time = time.time() - save_start
        print(f"Weights saved in {save_time:.1f} seconds")

        
        # Check if file was saved successfully
        if not os.path.exists(temp_weights_path):
            raise FileNotFoundError(f"Failed to save weights to {temp_weights_path}")

        original_size = os.path.getsize(temp_weights_path)
        print(f"Original weights file size: {original_size / (1024**3):.2f} GB")

        # Compress the weights file
        compressed_path = os.path.join(
            temp_path, f"temp_model.weights.h5{compression_ext}"
        )
        print(f"Compressing weights using {compression_method}...")

        # Use appropriate chunk size based on file size
        chunk_size = min(
            8 * 256 * 1024 * 1024, max(64 * 32 * 1024 * 1024, original_size // 1000)
        )  # 8-64MB chunks
        print(f"Using chunk size: {chunk_size / (1024 * 1024):.1f} MB")

        compress_start = time.time()
        bytes_written = 0

        with open(temp_weights_path, "rb") as f_in:
            if HAS_LZ4:
                # LZ4 compression - extremely fast
                # compression_level: 0-16, where 0 is fastest, 16 is best compression
                # We use 3 for a good balance (very fast, decent compression)
                with lz4.frame.open(
                    compressed_path,
                    "wb",
                    compression_level=16,
                    content_checksum=True,
                    block_size=lz4.frame.BLOCKSIZE_MAX4MB,
                ) as f_out:
                    with tqdm(
                        total=original_size,
                        unit="B",
                        unit_scale=True,
                        desc="Compressing (LZ4)",
                        ncols=100,
                    ) as pbar:
                        while True:
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            bytes_written += len(chunk)
                            pbar.update(len(chunk))
            else:
                # Gzip compression - slower but more compatible
                # compression level 1 is fastest, 9 is best compression
                with gzip.open(compressed_path, "wb", compresslevel=1) as f_out:
                    with tqdm(
                        total=original_size,
                        unit="B",
                        unit_scale=True,
                        desc="Compressing (gzip)",
                        ncols=100,
                    ) as pbar:
                        while True:
                            chunk = f_in.read(chunk_size)
                            if not chunk:
                                break
                            f_out.write(chunk)
                            bytes_written += len(chunk)
                            pbar.update(len(chunk))

        compress_time = time.time() - compress_start
        print(
            f"Compression completed in {compress_time:.1f} seconds ({original_size / compress_time / (1024**3):.2f} GB/s)"
        )

        # Verify compressed file exists and get size
        if not os.path.exists(compressed_path):
            raise FileNotFoundError(
                f"Compression failed - file not found: {compressed_path}"
            )

        compressed_size = os.path.getsize(compressed_path)
        compression_ratio = (1 - compressed_size / original_size) * 100
        print(
            f"Compressed size: {compressed_size / (1024**3):.2f} GB (saved {compression_ratio:.1f}%)"
        )

        # Delete original file to free up space before moving
        print("Removing temporary uncompressed file...")
        os.remove(temp_weights_path)

        # Determine final output path
        if output_path.endswith(".weights.h5"):
            final_compressed_path = output_path + compression_ext
        else:
            final_compressed_path = f"{output_path}.weights.h5{compression_ext}"

        # Ensure the output directory exists
        output_dir = os.path.dirname(final_compressed_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Move compressed file to final location
        print(f"Moving compressed file to: {final_compressed_path}")

        move_start = time.time()

        # If on same filesystem, use rename (instant), otherwise copy
        try:
            os.rename(compressed_path, final_compressed_path)
            print("File moved instantly (same filesystem)")
        except OSError:
            # Different filesystem, need to copy
            with tqdm(
                total=compressed_size,
                unit="B",
                unit_scale=True,
                desc="Moving file",
                ncols=100,
            ) as pbar:
                with open(compressed_path, "rb") as src:
                    with open(final_compressed_path, "wb") as dst:
                        while True:
                            chunk = src.read(chunk_size)
                            if not chunk:
                                break
                            dst.write(chunk)
                            pbar.update(len(chunk))

            # Remove the source file after successful copy
            os.remove(compressed_path)

        move_time = time.time() - move_start
        if move_time > 1:
            print(f"File moved in {move_time:.1f} seconds")

        # Verify the final file exists
        if not os.path.exists(final_compressed_path):
            raise FileNotFoundError(
                f"Failed to create final compressed file at {final_compressed_path}"
            )

        # Final statistics
        total_time = time.time() - start_time
        final_size = os.path.getsize(final_compressed_path)

        print("\n✅ Model weights saved successfully!")
        print(f"   Original size: {original_size / (1024**3):.2f} GB")
        print(f"   Compressed size: {final_size / (1024**3):.2f} GB")
        print(f"   Compression ratio: {compression_ratio:.1f}%")
        print(f"   Compression method: {compression_method}")
        print(f"   Total time: {total_time:.1f} seconds")
        print(f"   Throughput: {original_size / total_time / (1024**2):.1f} MB/s")
        print(f"   Final location: {final_compressed_path}")

        return final_compressed_path

    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        raise
    except Exception as e:
        print(f"❌ Error during model saving: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        # Always cleanup temp directory
        print("\nCleaning up temporary directory...")
        cleanup_temp_dir(temp_path)


def set_environment():
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"


def setup_distribution():
    """Sets up the model distribution strategy for multi-device training."""
    try:
        devices = keras.distribution.list_devices()
        num_devices = len(devices)
    except Exception as e:
        print(f"Warning: Could not list devices for distribution: {e}")
        print("Proceeding with single-device training.")
        return

    if num_devices <= 1:
        print("Only one device found. Skipping distributed setup.")
        return

    print(f"Found {num_devices} devices. Setting up distributed training.")

    # Define the device mesh and layout map for model sharding
    device_mesh = keras.distribution.DeviceMesh(
        (1, num_devices),
        ["batch", "model"],
        devices=devices,
    )

    model_dim = "model"
    layout_map = keras.distribution.LayoutMap(device_mesh)

    # Shard token embeddings
    layout_map["token_embedding/embeddings"] = (model_dim, None)
    # Shard attention layers
    # Regex to match against the query, key and value matrices in attention layers
    layout_map["decoder_block.*attention.*(query|key|value)/kernel"] = (
        "model",
        None,
        None,
    )
    layout_map["decoder_block.*attention_output/kernel"] = ("model", None, None)
    layout_map["decoder_block.*ffw_gating.*/kernel"] = (None, "model")
    layout_map["decoder_block.*ffw_linear/kernel"] = ("model", None)

    # Set the distribution strategy
    model_parallel = keras.distribution.ModelParallel(
        layout_map=layout_map, batch_dim_name="batch"
    )
    keras.distribution.set_distribution(model_parallel)
    print("Keras distribution strategy set for multi-device training.")


def load_model(preset: str) -> Gemma3CausalLM:
    model = Gemma3CausalLM.from_preset(preset)
    print(f"Loaded model: {preset}")
    return model


def run_inference(
    model: Gemma3CausalLM,
    instruction: str,
    max_length: int,
    seed: int,
    template: str = "Instruction:{instruction}\n\nResponse:{response}",
):
    prompt = template.format(instruction=instruction, response="")
    if not hasattr(model, "_compiled_for_inference"):
        sampler = keras_hub.samplers.TopKSampler(k=5, seed=seed)
        model.compile(sampler=sampler)
        model._compiled_for_inference = True
    output = model.generate(prompt, max_length=max_length)
    print(f"Inference output: {output}")
    return output


def fine_tune_model(
    model: Gemma3CausalLM,
    data: Dict[str, List[str]],
    rank: int,
    sequence_length: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
):
    model.backbone.enable_lora(rank=rank)
    print(model.summary())

    model.preprocessor.sequence_length = sequence_length
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    keras.mixed_precision.set_global_policy("mixed_bfloat16")

    model.fit(data, epochs=epochs, batch_size=batch_size)


def generate_eval_template(question: str, answer_1: str, answer_2: str):
    return (
        "You are a judge model. Your task is to compare 2 answers to a given question and score them based on a scale of 1 through 10."
        + "You are provided the original question, as well, as the corresponding answers by the 2 models."
        + "Judge both the answers on their conciseness, clarity and most importantly - accuracy."
        + "You should output the scores at the end of your deliberation. The scores should be in format: `JUDGE_SCORES: (<score_1>, <score_2>)` where the tuples contain two `int`s between 0 and 10."
        + f"This is the original question: ```markdown{question}```"
        + f"\nModel Answer 1: {answer_1}"
        + f"\nModel Answer 2: {answer_2}"
    )


def extract_scores(judge_response: str) -> tuple[int | None, int | None]:
    """Extract the scores from the judge model's response."""
    pattern = r"JUDGE_SCORES:\s*\((\d+),\s*(\d+)\)"
    match = re.search(pattern, judge_response)

    if match:
        try:
            score_1 = int(match.group(1))
            score_2 = int(match.group(2))
            return score_1, score_2
        except (ValueError, IndexError):
            pass

    print(f"Failed to extract scores from judge response: {judge_response[:100]}...")
    return None, None


def evaluate_model_locally(
    model: Gemma3CausalLM,
    eval_data: LoadedDataset,
    max_samples: int = 20,
    max_length: int = 256,
):
    """Evaluate model locally on a dataset by generating responses."""
    results = []

    print(f"Evaluating on {min(max_samples, len(eval_data))} samples...")

    # Compile once at the start
    if not hasattr(model, "_compiled_for_eval"):
        sampler = keras_hub.samplers.TopKSampler(k=5, seed=42)
        model.compile(sampler=sampler)
        model._compiled_for_eval = True

    for i, sample in tqdm(enumerate(eval_data[:max_samples])):
        question = sample["text_input"]

        # Generate response using the model
        prompt = f"Question: {question}\nAnswer:"
        try:
            generated_output = model.generate(prompt, max_length=max_length)

            # Extract just the answer part (remove the prompt)
            if prompt in generated_output:
                answer = generated_output[len(prompt) :].strip()
            else:
                answer = generated_output.strip()

        except Exception as e:
            print(f"Error generating for sample {i}: {e}")
            answer = f"Error: {str(e)}"

        results.append(answer)

        if i % 5 == 0:
            print(f"Completed {i + 1}/{min(max_samples, len(eval_data))} samples")

    return results


def compare_performance(results):
    """Compare the performance between unfinetuned and finetuned models using judge scores."""
    print("\n--- Performance Comparison ---")

    # Extract valid scores
    valid_results = [
        r
        for r in results
        if r["finetuned_score"] is not None and r["unfinetuned_score"] is not None
    ]

    if not valid_results:
        print("❌ No valid scores found from judge model")
        return

    finetuned_scores = [r["finetuned_score"] for r in valid_results]
    unfinetuned_scores = [r["unfinetuned_score"] for r in valid_results]

    avg_finetuned = sum(finetuned_scores) / len(finetuned_scores)
    avg_unfinetuned = sum(unfinetuned_scores) / len(unfinetuned_scores)

    print(f"Valid evaluations: {len(valid_results)}/{len(results)}")
    print(f"Finetuned model average score: {avg_finetuned:.2f}")
    print(f"Unfinetuned model average score: {avg_unfinetuned:.2f}")
    print(f"Improvement: {avg_finetuned - avg_unfinetuned:.2f}")

    # Win/loss/tie statistics
    wins = sum(
        1 for r in valid_results if r["finetuned_score"] > r["unfinetuned_score"]
    )
    losses = sum(
        1 for r in valid_results if r["finetuned_score"] < r["unfinetuned_score"]
    )
    ties = sum(
        1 for r in valid_results if r["finetuned_score"] == r["unfinetuned_score"]
    )

    print(f"Finetuned model: {wins} wins, {losses} losses, {ties} ties")

    if avg_finetuned > avg_unfinetuned:
        print("✅ Finetuning improved performance!")
    elif avg_finetuned < avg_unfinetuned:
        print("❌ Finetuning decreased performance")
    else:
        print("➖ No change in performance")

    # Show sample comparisons
    print("\n--- Sample Comparisons ---")
    for i, r in enumerate(valid_results[:3]):
        print(f"\nSample {i + 1}:")
        print(f"Question: {r['question'][:100]}...")
        print(f"Finetuned ({r['finetuned_score']}): {r['finetuned_output'][:150]}...")
        print(
            f"Unfinetuned ({r['unfinetuned_score']}): {r['unfinetuned_output'][:150]}..."
        )


def save_results(results, output_dir: str = "results"):
    script_dir = Path(__file__).parent
    full_output_dir = script_dir / output_dir
    os.makedirs(full_output_dir, exist_ok=True)

    with open(full_output_dir / "gemma_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {full_output_dir}/gemma_eval_results.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Gemma3 model")

    # Model configuration
    parser.add_argument(
        "--model",
        default="gemma3_instruct_1b",
        help="Model preset to use (default: gemma3_instruct_1b)",
    )
    parser.add_argument(
        "--output-model",
        default="finetuned_gemma3_1b",
        help="Path to save finetuned model (default: finetuned_gemma3_1b)",
    )

    # Data paths
    parser.add_argument(
        "--train-data",
        default="data/1000_documented_commits.json",
        help="Comma-separated list of training data paths (default: data/1000_documented_commits.json)",
    )
    parser.add_argument(
        "--eval-data",
        default="data/so_jax_qa_pairs.json",
        help="Evaluation data path (default: data/so_jax_qa_pairs.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results (default: results)",
    )

    # Training hyperparameters
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=384,
        help="Sequence length (default: 384)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=6e-4,
        help="Learning rate (default: 6e-4)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of epochs (default: 8)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )

    # Inference parameters
    parser.add_argument(
        "--test-prompt",
        default="Fix self-attention bug",
        help="Test prompt for inference (default: Fix self-attention bug)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max length for pre-training inference (default: 256)",
    )
    parser.add_argument(
        "--max-length-post",
        type=int,
        default=1024,
        help="Max length for post-training inference (default: 1024)",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Random seed for inference (default: 2)"
    )

    # Evaluation models
    parser.add_argument(
        "--eval-judge",
        default="gemini-2.5-flash-preview-05-20",
        help="Judge model for evaluation (default: gemini-2.5-flash-preview-05-20)",
    )
    parser.add_argument(
        "--eval-baseline",
        default="gemini-2.5-flash-preview-05-20",
        help="Baseline model for evaluation (default: gemini-2.5-flash-preview-05-20)",
    )

    # Flags
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate existing models",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation and only do training",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_environment()
    setup_distribution()

    # Initialize variables for evaluation results
    unfinetuned_outputs = None
    finetuned_outputs = None
    eval_data = None

    if not args.skip_training:
        print("=== Loading unfinetuned model ===")
        gemma_lm = load_model(args.model)
        print(gemma_lm.summary())

        print("\n=== Testing inference before fine-tuning ===")
        run_inference(
            gemma_lm, args.test_prompt, max_length=args.max_length, seed=args.seed
        )

        # Evaluate unfinetuned model immediately after loading if evaluation is enabled
        if not args.skip_evaluation:
            print("\n=== Loading evaluation data ===")
            script_dir = Path(__file__).parent
            eval_data_path = script_dir / args.eval_data
            eval_data = LoadedDataset(
                eval_data_path,
                truncate_sample,
            )
            print(f"Loaded {len(eval_data)} evaluation samples")

            print("\n=== Evaluating unfinetuned model (caching results) ===")
            unfinetuned_outputs = evaluate_model_locally(
                gemma_lm,
                eval_data,
                20,  # max_samples
            )

        print("\n=== Loading training data ===")

        # Load and combine datasets
        train_paths_str = [p.strip() for p in args.train_data.split(",")]
        train_paths = []
        for path_str in train_paths_str:
            path = Path(path_str)
            if not path.exists():
                # Try prepending the default data directory
                assumed_path = Path("data") / path_str
                if assumed_path.exists():
                    path = assumed_path
            train_paths.append(path)

        train_datasets = [
            LoadedDataset(path, truncate_sample).flatten_on_key()
            for path in train_paths
        ]

        # Combine all datasets into one
        combined_dataset = train_datasets[0]
        for ds in train_datasets[1:]:
            combined_dataset += ds

        # Convert to list for shuffling
        dataset_list = list(combined_dataset)
        
        # Set random seed for reproducibility
        random.seed(args.seed)
        random.shuffle(dataset_list)
        print(f"Shuffled training dataset with seed {args.seed}")

        data = {
            "prompts": [sample["text_input"] for sample in dataset_list],
            "responses": [sample["output"] for sample in dataset_list],
        }
        print(
            f"Loaded and combined {len(data['prompts'])} training samples from {len(args.train_data)} files"
        )

        print("\n=== Starting fine-tuning ===")
        fine_tune_model(
            model=gemma_lm,
            data=data,
            rank=args.rank,
            sequence_length=args.sequence_length,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        print("\n=== Testing inference after fine-tuning ===")
        run_inference(
            gemma_lm, args.test_prompt, max_length=args.max_length_post, seed=1
        )

        # Evaluate finetuned model immediately after training if evaluation is enabled
        if not args.skip_evaluation and eval_data is not None:
            print("\n=== Evaluating finetuned model ===")
            finetuned_outputs = evaluate_model_locally(
                gemma_lm,
                eval_data,
                20,  # max_samples
            )

    # Compare results using judge if we have both unfinetuned and finetuned outputs
    if (
        not args.skip_evaluation
        and unfinetuned_outputs is not None
        and finetuned_outputs is not None
    ):
        print("\n=== Judging outputs with Gemini ===")
        judge_model = GenerativeModel(args.eval_judge)

        results = []
        for i, sample in enumerate(eval_data[:20]):  # max_samples
            question = sample["text_input"]
            unfinetuned_answer = unfinetuned_outputs[i]
            finetuned_answer = finetuned_outputs[i]

            # Create judge prompt
            judge_prompt = generate_eval_template(
                question, finetuned_answer, unfinetuned_answer
            )

            try:
                judge_response = generate_from_model(
                    judge_model, judge_prompt, is_str=True
                )
                finetuned_score, unfinetuned_score = extract_scores(judge_response)
            except Exception as e:
                print(f"Error getting judge response for sample {i}: {e}")
                judge_response = f"Error: {str(e)}"
                finetuned_score, unfinetuned_score = None, None

            result = {
                "question": question,
                "finetuned_output": finetuned_answer,
                "unfinetuned_output": unfinetuned_answer,
                "finetuned_score": finetuned_score,
                "unfinetuned_score": unfinetuned_score,
                "judge_response": judge_response,
            }
            results.append(result)

            if i % 5 == 0:
                print(f"Judged {i + 1}/20 samples")

        compare_performance(results)
        save_results(results, args.output_dir)

    # Save model weights only if we did training
    if not args.skip_training:
        print(f"\n=== Saving finetuned model to {args.output_model} ===")
        
        # Ensure output goes to /kaggle/working
        kaggle_working = "/kaggle/working"
        if not os.path.exists(kaggle_working):
            os.makedirs(kaggle_working, exist_ok=True)
            
        if not args.output_model.startswith("/kaggle/working/"):
            if args.output_model.startswith("/"):
                # Absolute path, use just the filename in /kaggle/working
                filename = os.path.basename(args.output_model)
                output_path = f"/kaggle/working/{filename}"
            else:
                # Relative path, prepend /kaggle/working/
                output_path = f"/kaggle/working/{args.output_model}"
        else:
            output_path = args.output_model
            
        # Ensure .weights.h5 extension
        if not output_path.endswith(".weights.h5"):
            output_path = f"{output_path}.weights.h5"

        # Use ramdisk compression method
        save_model_with_compression(gemma_lm, output_path)

    print("\n🎉 Complete! Finetuning and evaluation finished.")


if __name__ == "__main__":
    main()
