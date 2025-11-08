# !pip install -U \
#   "transformers==4.52.4" \
#   "trl" \
#   "peft==0.15.2" \
#   "accelerate==1.8.1" \
#   "datasets>=2.21.0" \
#   "bitsandbytes>=0.43.2"

# pip install deepspeed
# !pip install rouge_score nltk sacrebleu


import os, random, math, json, time, glob
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from transformers.utils import is_bitsandbytes_available
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer, SFTConfig
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback

MODEL_ID   = "scb10x/llama3.2-typhoon2-t1-3b-research-preview"
CSV_PATH   = "input"
OUTPUT_DIR = "output"

# Set Config
MAX_SEQ_LEN = 4000        
BATCH_SIZE = 2
GRAD_ACCUM = 16
EPOCHS = 2                
LORA_TARGET = ["q_proj","k_proj","v_proj","o_proj"]
LORA_R = 8
LORA_ALPHA = 16
GC_FREE_GB_TH = 6.0 

CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints_hf")     
ADAPTER_SAFETY_DIR = os.path.join(OUTPUT_DIR, "adapter_safety")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(ADAPTER_SAFETY_DIR, exist_ok=True)

TOK_CACHE_DIR = os.path.join(OUTPUT_DIR, "tokenized_ds")

SEED = 42
LR = 1e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.0

LOG_EVERY  = 25
EVAL_STRATEGY = "steps"     
SAVE_STRATEGY = "steps"
EVAL_EVERY = 25 
SAVE_EVERY = 200

MAX_GRAD_NORM = 1.0
USE_EARLY_STOPPING = False
PATIENCE = 4

TIME_LIMIT_HOURS = 200
TIME_LIMIT_SAVE_EVERY_STEPS = 200  

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

USE_DEEPSPEED = False
try:
    import deepspeed  
    USE_DEEPSPEED = True
except Exception:
    USE_DEEPSPEED = False

# Load Data
df = pd.read_csv(CSV_PATH)
need = {"cleaned_text","question_text","answer_text"}
missing = need - set(df.columns)
assert not missing, f"Missing columns: {missing}"
df = df[list(need)].dropna().reset_index(drop=True)

df["__key__"] = (
    df["cleaned_text"].astype(str) + "||" +
    df["question_text"].astype(str) + "||" +
    df["answer_text"].astype(str)
)
df = df.drop_duplicates("__key__").drop(columns="__key__").reset_index(drop=True)

# Set system instructions
SYSTEM_MSG = (
    "คุณคือผู้ช่วย AI ของกองทุนเงินให้กู้ยืมเพื่อการศึกษา (กยศ.) "
    "ตอบคำถามด้วยความสุภาพ กระชับ และอ้างอิงจากข้อมูลหรือบริบทที่มีอยู่เท่านั้น "
    "พร้อมอธิบายเหตุผลประกอบว่าทำไมถึงตอบเช่นนั้น"
    "ตอบให้ตรงกับคำถาม แต่หากข้อมูลไม่เพียงพอ ให้แจ้งผู้ใช้ตามตรงว่าไม่สามารถให้คำตอบได้"
    "อธิบายคำตอบกี่เกี่ยวข้องกับคำถามแบบละเอียด"
)


def row_to_messages(row):
    context = str(row["cleaned_text"])
    q       = str(row["question_text"])
    a       = str(row["answer_text"]).strip()
    user_msg = (
        "ต่อไปนี้คือบริบท (context):\n"
        f"{context}\n\n"
        "คำถาม:\n"
        f"{q}\n\n"
        "โปรดตอบเป็นภาษาไทยและยึดตามบริบทด้านบนเท่านั้น"
    )
    return [
        {"role":"system","content":SYSTEM_MSG},
        {"role":"user","content":user_msg},
        {"role":"assistant","content":a},
    ]

def render_example(row):
    msgs = row_to_messages(row)
    return (
        f"<|system|>\n{msgs[0]['content']}\n"
        f"<|user|>\n{msgs[1]['content']}\n"
        f"<|assistant|>\n{msgs[2]['content']}"
    )

df["text"] = df.apply(render_example, axis=1)

train_df, val_df = train_test_split(
    df[["text"]], test_size=0.1, random_state=SEED, shuffle=True
)

train_ds_raw = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds_raw   = Dataset.from_pandas(val_df.reset_index(drop=True))
raw_data     = DatasetDict({"train": train_ds_raw, "validation": val_ds_raw})

# Load Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def tokenize_batch(ex):
    out = tokenizer(
        ex["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
        return_attention_mask=True
    )
    out["labels"] = out["input_ids"].copy()
    return out

if os.path.exists(TOK_CACHE_DIR):
    data_tok = load_from_disk(TOK_CACHE_DIR)
else:
    data_tok = raw_data.map(tokenize_batch, batched=True, num_proc=4, desc="Tokenizing")
    keep_cols = ["input_ids","attention_mask","labels"]
    data_tok = DatasetDict({
        "train": data_tok["train"].remove_columns([c for c in data_tok["train"].column_names if c not in keep_cols]),
        "validation": data_tok["validation"].remove_columns([c for c in data_tok["validation"].column_names if c not in keep_cols]),
    })
    data_tok.save_to_disk(TOK_CACHE_DIR)


bnb_config = None
if is_bitsandbytes_available():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=DTYPE,
    )

# Load Model 
attn_impl_candidates = ["sdpa", "eager"]  
last_err = None
for attn_impl in attn_impl_candidates:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
            quantization_config=bnb_config if bnb_config is not None else None,
        )
        break
    except Exception as e:
        last_err = e
else:
    raise RuntimeError(f"Failed to load model. Last error: {last_err}")

# Gradient Checkpointing
def get_free_total_gb(device=0):
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return free_bytes / (1024**3), total_bytes / (1024**3)

AUTO_TOGGLE_GC = True
USE_GRADIENT_CHECKPOINTING = False  
if torch.cuda.is_available() and AUTO_TOGGLE_GC:
    free_gb, total_gb = get_free_total_gb(0)
    if free_gb >= GC_FREE_GB_TH:   
        USE_GRADIENT_CHECKPOINTING = False

model.config.use_cache = False
if USE_GRADIENT_CHECKPOINTING:
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
else:
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

if bnb_config is not None:
    model = prepare_model_for_kbit_training(model)

# Set QLoRA
peft_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=LORA_TARGET,
)

# Set DeepSpeed 
ds_config = None
if USE_DEEPSPEED:
    ds_config = {
        "train_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "bf16": {"enabled": bool(USE_BF16)},
        "fp16": {"enabled": (not USE_BF16)},
        "gradient_clipping": MAX_GRAD_NORM,
    }

# TimeLimit Callback 
class TimeLimitCallback(TrainerCallback):
    def __init__(self, max_hours=11.75, adapter_dir=ADAPTER_SAFETY_DIR, save_every_steps=TIME_LIMIT_SAVE_EVERY_STEPS):
        self.max_secs = max_hours * 3600
        self.adapter_dir = adapter_dir
        self.save_every_steps = save_every_steps
        os.makedirs(adapter_dir, exist_ok=True)
    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()
    def _save_adapter(self, model, tokenizer, tag):
        path = os.path.join(self.adapter_dir, tag)
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path, safe_serialization=True)
        tokenizer.save_pretrained(path)
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)
        if model is None or tokenizer is None:
            return control
        if state.global_step and state.global_step % self.save_every_steps == 0:
            self._save_adapter(model, tokenizer, f"step-{state.global_step}")
        if time.time() - self.t0 >= self.max_secs:
            self._save_adapter(model, tokenizer, f"step-{state.global_step}-final")
            control.should_training_stop = True
        return control

def latest_hf_checkpoint_dir(base=CKPT_DIR):
    paths = sorted(glob.glob(os.path.join(base, "checkpoint-*")), key=os.path.getmtime)
    return paths[-1] if paths else None

# Trainer Args
optim_choice = "adamw_torch_fused" if USE_DEEPSPEED else ("paged_adamw_8bit" if is_bitsandbytes_available() else "adamw_torch")

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    logging_steps=LOG_EVERY,

    eval_strategy="steps",   
    save_strategy="steps",           
    save_steps=SAVE_EVERY,
    save_total_limit=2,
    load_best_model_at_end=False,  

    bf16=USE_BF16,
    fp16=(not USE_BF16),
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    max_seq_length=MAX_SEQ_LEN,

    dataset_text_field=None,
    packing=False,

    report_to="none",
    max_grad_norm=MAX_GRAD_NORM,
    ddp_find_unused_parameters=False,
    deepspeed=ds_config,
    optim=optim_choice,
    save_safetensors=True,
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_cfg,
    train_dataset=data_tok["train"],
    eval_dataset=data_tok["validation"],
    args=args,
)

if USE_EARLY_STOPPING and EVAL_STRATEGY != "no":
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=PATIENCE))

trainer.add_callback(TimeLimitCallback(max_hours=TIME_LIMIT_HOURS))

resume_dir = latest_hf_checkpoint_dir(CKPT_DIR)
resume_dir = latest_hf_checkpoint_dir(OUTPUT_DIR) or resume_dir

trainer.train(resume_from_checkpoint=resume_dir)

trainer.model.save_pretrained(os.path.join(ADAPTER_SAFETY_DIR, "final"), safe_serialization=True)
tokenizer.save_pretrained(os.path.join(ADAPTER_SAFETY_DIR, "final"))

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
