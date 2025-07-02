from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner
import os
from sae_lens import (
    SAE,
    ActivationsStore,
    HookedSAETransformer,
    LanguageModelSAERunnerConfig,
    SAEConfig,
    SAETrainingRunner,
    upload_saes_to_huggingface,
)
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_vis import SaeVisConfig, SaeVisData, SaeVisLayoutConfig
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name, test_prompt, to_numpy
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from functools import partial


import sys
layer = int(sys.argv[1])
compnent_id = int(sys.argv[2])
factor = int(sys.argv[3])
dataset_path = sys.argv[4] # "Yusser/m_sae_wiki_tokenized"
is_dataset_tokenized = bool(sys.argv[5]) # 0,1 
save_path = sys.argv[6] # "Yusser/m_sae_wiki_tokenized"



device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
dtype= "float16"

trans_component = ["m","r","a"]
compnent = trans_component[compnent_id]

release=f"llama_scope_lx{compnent}_{factor}x"
sae_id=f"l{layer}{compnent}_{factor}x"

_, cfg_dict, _ = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device=str(device),
)
print(cfg_dict)


os.makedirs("./cache/", exist_ok=True)




#from sae_intrep import *





total_training_steps = 15_000  # probably we should do more
batch_size = 4096
total_training_tokens = total_training_steps * batch_size

lr_warm_up_steps = l1_warm_up_steps = total_training_steps // 10  # 10% of training
lr_decay_steps = total_training_steps // 5  # 20% of training

cfg = LanguageModelSAERunnerConfig(
    #
    # Data generation
    model_name=cfg_dict.get('model_name'),  # our model (more options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html)
    hook_name=cfg_dict.get('hook_name'),
    hook_layer=cfg_dict.get('hook_layer'),
    d_in=cfg_dict.get('d_in'),
    dataset_path=dataset_path,  # tokenized language dataset on HF for the Tiny Stories corpus.
    is_dataset_tokenized=is_dataset_tokenized,
    prepend_bos=True,  # you should use whatever the base model was trained with
    streaming=True,  # we could pre-download the token dataset if it was small.
    train_batch_size_tokens=batch_size, # (int): The batch size for training. This controls the batch size of the SAE Training loop.
    context_size= 512, #cfg_dict.get('context_size'),  # larger is better but takes longer (for tutorial we'll use a short one)


    # Fine Tune SAE
    #from_pretrained_path (str, optional): The path to a pretrained SAE. We can finetune an existing SAE if needed.

    #
    # SAE architecture
    #architecture="gated",
    architecture="jumprelu",
    l1_coefficient=5.0,
    jumprelu_bandwidth=0.001,
    jumprelu_init_threshold=0.001, #cfg_dict.get('jump_relu_threshold'), #0.001,
    
    expansion_factor=factor, # 8
    b_dec_init_method="zeros",
    apply_b_dec_to_input=True,
    normalize_sae_decoder=False,
    scale_sparsity_penalty_by_decoder_norm=True,
    decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    #
    # Activations store
    n_batches_in_buffer=64, # (int): The number of batches in the buffer. When not using cached activations, a buffer in ram is used. The larger it is, the better shuffled the activations will be.
    training_tokens=total_training_tokens, # (int): The number of training tokens.
    store_batch_size_prompts=16, # (int): The batch size for storing activations. This controls how many prompts are in the batch of the language model when generating actiations.
    act_store_device='cpu',
    
    #use_cached_activations=True, #(bool): Whether to use cached activations. This is useful when doing sweeps over the same activations.
    #cached_activations_path='./cache', #(str, optional): The path to the cached activations.
    #
    # Training hyperparameters (standard)
    lr=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # controls how the LR warmup / decay works
    lr_warm_up_steps=lr_warm_up_steps,  # avoids large number of initial dead features
    lr_decay_steps=lr_decay_steps,  # helps avoid overfitting
    #
    # Training hyperparameters (SAE-specific)
    #l1_coefficient=4,
    l1_warm_up_steps=l1_warm_up_steps,
    use_ghost_grads=False,  # we don't use ghost grads anymore
    feature_sampling_window=2000,  # how often we resample dead features
    dead_feature_window=1000,  # size of window to assess whether a feature is dead
    dead_feature_threshold=1e-4,  # threshold for classifying feature as dead, over window
    #
    # Logging / evals
    log_to_wandb=True,  # always use wandb unless you are just testing code.
    wandb_project="multilingual_saes",
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
    #
    # Misc.
    device=str(device),
    seed=42,
    n_checkpoints=5,
    checkpoint_path="./cache/checkpoints",
    dtype="float32", #dtype,

    #model_kwargs = {'dtype': dtype},  #(dict[str, Any]): Additional keyword arguments for the model.
    model_from_pretrained_kwargs = {'dtype': str(dtype)},  #(dict[str, Any]): Additional keyword arguments for the model from pretrained.

)

print("Comment this code out to train! Otherwise, it will load in the already trained model.")
t.set_grad_enabled(True)
runner = SAETrainingRunner(cfg)
sae = runner.run()

os.makedirs(save_path, exist_ok=True)
save_path = os.path.join(save_path, sae.cfg.hook_name)
sae.save_model(save_path)

#hf_repo_id = "Yusser/multilingual_llama3.1-8B_saes"
#sae_id = cfg.hook_name
#upload_saes_to_huggingface({sae_id: sae}, hf_repo_id=hf_repo_id)

#sae = SAE.from_pretrained(release=hf_repo_id, sae_id=sae_id, device=str(device))[0]