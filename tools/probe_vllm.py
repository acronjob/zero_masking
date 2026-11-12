import json, torch
from huggingface_hub import hf_hub_download
from vllm import LLM, SamplingParams
M = "Qwen/Qwen3.5-0.8B"
idx = json.load(open(hf_hub_download(M, "model.safetensors.index.json")))["weight_map"]
ck = sorted(k for k in idx if "gate_proj" in k); print("ckpt gate_proj names:", len(ck), ck[:2])
llm = LLM(M, gpu_memory_utilization=0.25, max_model_len=5120, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
print("vllm model:", llm.apply_model(lambda m: type(m).__name__))
print("vllm gate params:", llm.apply_model(lambda m: [(n, tuple(p.shape), str(p.dtype)) for n, p in m.named_parameters() if "gate_up" in n][:2]))
tok = llm.get_tokenizer()
qs = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
      "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"]
P = [{"prompt_token_ids": tok(tok.apply_chat_template([{"role": "user", "content": q + ' Let\'s think step by step and output the final answer after "####".'}], add_generation_prompt=True, tokenize=False))["input_ids"]} for q in qs]
sp = SamplingParams(temperature=0, max_tokens=200)
gen = lambda: [o.outputs[0].token_ids for o in llm.generate(P, sp, use_tqdm=False)]
a = gen()
ptr_before = llm.apply_model(lambda m: [p.data_ptr() for n, p in m.named_parameters() if "gate_up" in n][0])
from safetensors import safe_open
orig = []
for k in ck:
    with safe_open(hf_hub_download(M, idx[k]), "pt") as f: orig.append((k, f.get_tensor(k)))
llm.apply_model(lambda m: m.load_weights([(k, torch.zeros_like(t)) for k, t in orig]))
llm.reset_prefix_cache()
b = gen()
llm.apply_model(lambda m: m.load_weights([(k, t) for k, t in orig]))
llm.reset_prefix_cache()
c = gen()
ptr_after = llm.apply_model(lambda m: [p.data_ptr() for n, p in m.named_parameters() if "gate_up" in n][0])
print("same ptr after load:", ptr_before == ptr_after)
print("zeroed changes output:", a != b, "| restore matches original:", a == c)
print("orig :", tok.decode(a[0])[:160].replace("\n", " "))
print("zero :", tok.decode(b[0])[:160].replace("\n", " "))
print("eos in output ids?", [o.outputs[0].token_ids[-1] for o in llm.generate(P, SamplingParams(temperature=0, max_tokens=2000), use_tqdm=False)], tok.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
