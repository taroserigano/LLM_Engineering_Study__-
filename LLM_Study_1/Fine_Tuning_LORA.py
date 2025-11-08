



# ==============================


# Train LoRA → Merge → Convert → Load in Ollama

# Train LoRA → Merge → Convert (for appropriate data file format like gguf for ollama specific → Load in Ollama







# train_lora_llama3.py
# QLoRA training for Llama-3-8B-Instruct
# Highly commented, simple English.
# ==============================

# ---- Imports ----
# "datasets" loads JSONL/CSV datasets easily.
from datasets import load_dataset

# "transformers" provides the model + tokenizer classes.
from transformers import AutoTokenizer, AutoModelForCausalLM

# "peft" adds LoRA/QLoRA (Parameter-Efficient Fine-Tuning) utilities.
from peft import LoraConfig, get_peft_model

# "trl" gives SFTTrainer (Supervised Fine-Tuning) for chat/instruction data.
from trl import SFTTrainer, TrainingArguments

# We'll also use torch dtype names in one line below (optional).
import torch


# ---- Choose a base model ----
# This is the Llama-3 8B Instruct model on Hugging Face.
# You must accept its license to download it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# ---- Load the tokenizer ----
# "use_fast=True" uses the fast Rust tokenizer (faster).
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

# Some Llama tokenizers may miss a pad token. We set it to EOS to be safe.
# This prevents training crashes when batches need padding.
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# ---- Load the base model in 4-bit (this is QLoRA) ----
# - load_in_4bit=True: quantize the *base* model to 4-bit to save VRAM.
# - device_map="auto": put layers on your available GPU(s) automatically.
# - torch_dtype: compute type for forward pass; bf16 is great if your GPU supports it,
#   otherwise use float16. We'll try bf16 and fall back if needed.
try:
    compute_dtype = torch.bfloat16
except Exception:
    compute_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,      # key flag for QLoRA
    device_map="auto",
    torch_dtype=compute_dtype
)

# ---- Load dataset ----
# We load our local JSONL. The split key "train" returns a Dataset object.
# This expects a file "train.jsonl" with a column named "text".
ds = load_dataset("json", data_files={"train": "train.jsonl"})["train"]

# (If your data is chat-style with {"messages":[...]}, you'd first convert messages
#  into one "text" string per row, matching the format you want at inference.)

# ---- Configure LoRA ----
# The LoRA adapter replaces a small part of the model with trainable layers.
# Common targets for attention layers: q_proj, k_proj, v_proj, o_proj.
# You can start with just ["q_proj","v_proj"]; adding k/o gives more capacity (more VRAM).
lora_cfg = LoraConfig(
    r=8,                         # rank: capacity of the adapter (higher = more capacity/VRAM)
    lora_alpha=16,               # scaling factor for the adapter
    lora_dropout=0.05,           # regularization to avoid overfitting
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# ---- Wrap the base model with LoRA ----
# This leaves the original weights frozen and adds small trainable adapter weights.
model = get_peft_model(model, lora_cfg)

# ---- Training hyperparameters ----
# Keep them small to fit in 16–24GB VRAM.
train_args = TrainingArguments(
    output_dir="lora-out",             # where to save checkpoints/adapters
    per_device_train_batch_size=1,     # fits almost anywhere
    gradient_accumulation_steps=8,     # effective batch size = 1 * 8
    learning_rate=2e-4,                # good starting LR for LoRA
    num_train_epochs=1,                # start with 1 epoch; increase if needed
    fp16=(compute_dtype == torch.float16),  # enable fp16 if that's your dtype
    bf16=(compute_dtype == torch.bfloat16), # enable bf16 if that's your dtype
    logging_steps=10,                  # log every 10 steps
    save_strategy="epoch",             # save at end of each epoch
    optim="paged_adamw_8bit",          # memory-efficient optimizer
    report_to="none"                   # disable wandb/tensorboard by default
)

# ---- Trainer (SFT = supervised fine-tune) ----
# dataset_text_field="text": we tell the trainer which column holds the full
# input+output string the model should learn from.
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    dataset_text_field="text",
    args=train_args
)

# ---- Train! ----
trainer.train()

# ---- Save only the LoRA adapter (small, MBs) and tokenizer ----
# This does NOT save the full base model (GBs). Just the adapter weights.
model.save_pretrained("lora-out")
tok.save_pretrained("lora-out")

print("Training done. LoRA adapter saved to ./lora-out")
