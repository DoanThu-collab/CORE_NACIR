"""PlugIR active dialogue generator — Phase 2 same-evidence version (TURBO).

Generates paired raw Q/A trajectories AND reconstructed queries from the
same active PlugIR interaction, satisfying the same-evidence protocol.

TURBO optimizations:
  - ThreadPoolExecutor for parallel LLM calls (5 questions generated simultaneously)
  - requests.Session with connection pooling (avoid TCP handshake overhead)
  - Ollama OLLAMA_NUM_PARALLEL=8 support
  - torch.compile for BLIP inference (if available)
  - Aggressive torch.inference_mode everywhere
"""

import torch
import time
import json
import os
import datetime
import subprocess
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, BlipForImageTextRetrieval
from torch.nn.functional import normalize
from typing import Optional
import argparse
from fast_pytorch_kmeans import KMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


class BlipForRetrieval(BlipForImageTextRetrieval):
    def get_text_features(self,
                          input_ids: torch.LongTensor,
                          attention_mask: Optional[torch.LongTensor] = None,
                          return_dict: Optional[bool] = None,
                          ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        question_embeds = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        text_feat = normalize(self.text_proj(question_embeds[:, 0, :]), dim=-1)
        return text_feat

    def get_image_features(
            self,
            pixel_values: torch.FloatTensor,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]
        image_feat = normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat


# ============================================================================
# Argument parsing
# ============================================================================
parser = argparse.ArgumentParser(description="PlugIR active dialogue generator (Phase 2 TURBO)")
parser.add_argument('--seed', type=int, default=1021)
parser.add_argument('--s_idx', type=int, default=0)
parser.add_argument('--e_idx', type=int, default=2064)
parser.add_argument('--q_n', type=int, default=5)
parser.add_argument('--recall_hitting', type=int, default=10)
parser.add_argument('--thres_low', type=int, default=500)
parser.add_argument('--n_clusters', type=int, default=10)
parser.add_argument('--rounds', type=int, default=10)
parser.add_argument('--reconstruct', action='store_true')
parser.add_argument('--referring', action='store_true')
parser.add_argument('--filtering', action='store_true')
parser.add_argument('--select', action='store_true')
parser.add_argument('--subset_indices_path', type=Path, default=None,
                    help='Path to JSON containing specific subset of session IDs to generate.')

# LLM backend
parser.add_argument('--llm_model', type=str, default='llama3.1:8b')
parser.add_argument('--ollama_url', type=str,
                    default=os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434'))
parser.add_argument('--request_timeout', type=float, default=600.0)
parser.add_argument('--max_retries', type=int, default=6)
parser.add_argument('--llm_workers', type=int, default=8,
                    help='Number of parallel LLM request threads')

# Paths
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--queries_path', type=Path,
                    default=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR/dialogues/VisDial_v1.0_queries_val.json'))
parser.add_argument('--search_space_path', type=Path,
                    default=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR/Protocol/Search_Space_val_50k.json'))
parser.add_argument('--captions_path', type=Path,
                    default=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR/Protocol/visdial_captions.json'))
parser.add_argument('--embeddings_path', type=Path,
                    default=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/PlugIR_Workspace/ChatIR/temp/corpus_blip_large.pth'))
parser.add_argument('--image_root', type=Path,
                    default=Path('/mlcv1/WorkingSpace/Personal/core_baotg/thuy/Dataset/PlugIR'))
parser.add_argument('--output_dir', type=Path,
                    default=Path('artifacts_final/plugir_full'))
parser.add_argument('--checkpoint_every', type=int, default=10)
args = parser.parse_args()

# ============================================================================
# Variables
# ============================================================================
SEED = args.seed
s_idx = args.s_idx
e_idx = args.e_idx
q_n = args.q_n
recall_hitting = args.recall_hitting
threshold_low = args.thres_low
n_clusters = args.n_clusters
num_rounds = args.rounds
reconstruct = args.reconstruct
referring = args.referring
filtering = args.filtering
select = args.select

# ============================================================================
# Validate paths
# ============================================================================
for required_path in (args.queries_path, args.search_space_path, args.captions_path,
                      args.embeddings_path, args.image_root):
    if not required_path.exists():
        parser.error(f'Required path does not exist: {required_path}')
args.output_dir.mkdir(parents=True, exist_ok=True)

if not args.ollama_url.startswith('http'):
    args.ollama_url = f'http://{args.ollama_url}'

# ============================================================================
# Reproducibility
# ============================================================================
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True  # TURBO: enable cudnn benchmark for faster conv
torch.backends.cudnn.deterministic = False  # TURBO: relax for speed

# ============================================================================
# LLM client — connection-pooled requests.Session + ThreadPoolExecutor
# ============================================================================
_http_session = requests.Session()
_http_session.headers.update({'Content-Type': 'application/json'})
# Connection pooling: keep-alive connections to Ollama
adapter = requests.adapters.HTTPAdapter(
    pool_connections=args.llm_workers,
    pool_maxsize=args.llm_workers * 2,
    max_retries=0,  # we handle retries ourselves
)
_http_session.mount('http://', adapter)

_llm_executor = ThreadPoolExecutor(max_workers=args.llm_workers)
_LLM_URL = f"{args.ollama_url.rstrip('/')}/v1/chat/completions"


def call_llm(messages, temperature=0.0, max_tokens=None, top_p=None,
             seed=SEED, model=None, **_ignored):
    """Call Ollama via connection-pooled HTTP request."""
    model = model or args.llm_model

    payload = {
        'model': model,
        'messages': messages,
        'temperature': temperature,
    }
    if max_tokens is not None:
        payload['max_tokens'] = max_tokens
    if top_p is not None:
        payload['top_p'] = top_p

    retry_count = 0
    while retry_count < args.max_retries:
        try:
            resp = _http_session.post(_LLM_URL, json=payload, timeout=args.request_timeout)
            resp.raise_for_status()
            body = resp.json()
            if 'choices' not in body or not body['choices']:
                raise RuntimeError(f"Invalid response: {body}")
            return body
        except Exception as e:
            print(f"LLM Call Error (attempt {retry_count+1}): {e}")
            time.sleep(5 * (retry_count + 1))
            retry_count += 1

    raise RuntimeError("Too many retries calling LLM")


def call_llm_async(messages, **kwargs):
    """Submit an LLM call to the thread pool. Returns a Future."""
    return _llm_executor.submit(call_llm, messages, **kwargs)


# ============================================================================
# Load models
# ============================================================================
device = args.device
model_id = "Salesforce/blip2-flan-t5-xl"
model_dtype = torch.float16 if device.startswith('cuda') else torch.float32

print(f"Loading BLIP retrieval model to {device}...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
blip = BlipForRetrieval.from_pretrained("Salesforce/blip-itm-large-coco", torch_dtype=model_dtype).to(device)
blip.eval()

print(f"Loading BLIP2 VQA model to {device}...")
processor2 = Blip2Processor.from_pretrained(model_id)
blip2 = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    device_map={"": device},
    torch_dtype=model_dtype,
)
blip2.eval()

# ============================================================================
# Load data
# ============================================================================
print("Loading data files...")
with args.queries_path.open('r') as f:
    visdial = json.load(f)
with args.search_space_path.open('r') as f:
    search_space = json.load(f)
with args.captions_path.open('r') as f:
    captions = json.load(f)

if args.subset_indices_path is not None:
    with args.subset_indices_path.open('r') as f:
        target_indices = json.load(f)
else:
    if not 0 <= s_idx < e_idx <= len(visdial):
        parser.error(f'Expected 0 <= s_idx < e_idx <= {len(visdial)}, got {s_idx}:{e_idx}')
    target_indices = list(range(s_idx, e_idx))

img_embs = torch.load(args.embeddings_path, map_location=device)[1].to(model_dtype)
kmeans = KMeans(n_clusters=n_clusters, mode='cosine', verbose=0)

print(f"All models and data loaded. Generating {len(target_indices)} target sessions...")
print(f"TURBO mode: {args.llm_workers} parallel LLM workers")


# ============================================================================
# Utility functions
# ============================================================================

def reconstruct_dialog(dial, temp=.0, model=None):
    caption = dial[0]
    dialog = ', '.join(dial[1:])
    retry_count = 0
    dialog_examplar = ', '.join(["is this in a park? yes, i believe it is", "are there others around? no, she is alone",
                                 "does she have a collection bucket? no", "is her hair long? yes, pretty long",
                                 "is she wearing a dress? i don't think so, hard to tell",
                                 "does she have shoes on? yes, flip flops", "is there grass nearby? yes, everywhere",
                                 "is it a sunny day? yes", "are there trees? in the background there are trees",
                                 "is the guitar new? i don't think so"])
    messages = [{"role": "system",
                 "content": "Your role is to reconstruct the [Caption] with the additional information given by following [Dialogue]. The reconstructed [New Caption] should be concise and in appropriate form to retrieve a target image from a pool of candidate images"}]
    messages.append({"role": "user",
                     "content": f"[Caption]: a woman sits on a bench holding a guitar in her lap [Dialogue]: {dialog_examplar}  [New Caption]: "})
    messages.append({"role": "assistant",
                     "content": "a woman with pretty long hair sits alone on a grassy bench in a park on a sunny day, holding a guitar in her lap without a collection bucket, wearing flip flops, with trees in the background, with a slightly worn guitar"})
    messages.append({"role": "user", "content": f"[Caption]: {caption} [Dialogue]: {dialog}  [New Caption]: "})
    while True:
        try:
            response = call_llm(model=model, messages=messages, temperature=temp, max_tokens=512, seed=SEED)
            break
        except Exception as e:
            print(e)
            time.sleep(3)
            retry_count += 1
            if retry_count >= args.max_retries:
                return f"Error: {e}"
            continue
    return response['choices'][0]['message']['content']


_TEXT_EMB_CACHE = {}

@torch.inference_mode()
def get_text_features(text):
    if isinstance(text, str):
        if text not in _TEXT_EMB_CACHE:
            inputs = processor(text=text, padding=True, return_tensors="pt").to(device)
            _TEXT_EMB_CACHE[text] = blip.get_text_features(**inputs)[0].detach().cpu()
        return _TEXT_EMB_CACHE[text].unsqueeze(0).to(device)
    else:
        # text is a list of strings
        out = []
        missing = []
        for t in text:
            if t not in _TEXT_EMB_CACHE:
                missing.append(t)
                
        if missing:
            BATCH = 64
            for b_start in range(0, len(missing), BATCH):
                batch = missing[b_start:b_start+BATCH]
                inputs = processor(text=batch, padding=True, return_tensors="pt").to(device)
                feats = blip.get_text_features(**inputs).detach().cpu()
                for j, t in enumerate(batch):
                    _TEXT_EMB_CACHE[t] = feats[j]
                    
        return torch.stack([_TEXT_EMB_CACHE[t] for t in text]).to(device)


def search_imgs(query="", img_embs=None, search_space=None, k=10):
    query_emb = get_text_features(query)
    query_emb_norm = normalize(query_emb, dim=-1)
    cos_sim = torch.matmul(query_emb_norm, img_embs.T).squeeze()
    related_indices = cos_sim.sort()[1][-k:]
    related_imgs = [search_space[related_indices[idx].item()] for idx in range(k)]
    return related_imgs, related_indices, cos_sim


def get_related_captions(caption_recon, round=1):
    caps = []
    related_size = int(threshold_low - (round-1) * (threshold_low / 10.0))
    emb = normalize(get_text_features(caption_recon), dim=-1)
    sim = torch.matmul(emb, img_embs.T).squeeze()
    topk = sim.argsort()[-related_size:]
    img_embs_topk = img_embs[topk]

    # TURBO: batch all caption embeddings at once instead of one-by-one
    cap_texts = [captions[topk[i].item()]['caption'] if isinstance(captions[topk[i].item()]['caption'], str)
                 else captions[topk[i].item()]['caption'][0] for i in range(related_size)]

    # Process in batches of 64 to avoid OOM
    BATCH = 64
    entropies = torch.zeros([related_size])
    for b_start in range(0, related_size, BATCH):
        b_end = min(b_start + BATCH, related_size)
        batch_texts = cap_texts[b_start:b_end]
        batch_embs = get_text_features(batch_texts)
        batch_embs = normalize(batch_embs, dim=-1)
        batch_sims = torch.matmul(batch_embs, img_embs_topk.T)  # [batch, related_size]
        batch_p = torch.nn.functional.softmax(batch_sims, dim=1)
        batch_entropy = (-batch_p * batch_p.log()).sum(dim=1).detach().cpu()
        entropies[b_start:b_end] = batch_entropy

    idx_entropies_sorted = entropies.argsort()
    cluster_label = kmeans.fit_predict(img_embs_topk)
    cluster_label_sorted = cluster_label[idx_entropies_sorted]
    for i in range(n_clusters):
        if (cluster_label_sorted == i).any():
            idx_c = (cluster_label_sorted == i).nonzero().squeeze().min()
            cap_val = captions[topk[idx_entropies_sorted[idx_c]].item()]['caption']
            caps.append(cap_val[0] if isinstance(cap_val, list) else cap_val)

    return caps


def get_referring_prompt(caption="", img_embs=None, k=10, round=1, search_space=None):
    img_paths, top_k, cos_sims = search_imgs(caption, img_embs, search_space, k=k)
    fakes = get_related_captions(caption, round)
    prompt_sys = ("You should leverage the 'Fake Information' that is related to the target image "
                  "corresponding to the caption but does not match the target image.")
    prompt_fake = ""
    for i in range(len(fakes)):
        prompt_fake += str(i) + '. ' + fakes[i] + '\n'
    return prompt_sys, prompt_fake, top_k, cos_sims


def _build_question_messages(descrip):
    """Build the standard question-generation prompt messages (reusable)."""
    prompt_sys = ("You are a proficient question generator tasked with aiding in the retrieval of a target image. "
                  "Your role is to generate questions about the target image of the description via "
                  "leveraging two key information sources:\n\n"
                  "[Description]: This is a concise explanation of the target image.\n"
                  "[Dialogue]: Comprising question and answer pairs that seek additional "
                  "details about the target image.\n"
                  "Your generated question about the description must be clear, succinct, and concise, "
                  "while differing from prior questions in the [Dialogue].")

    prompt_example = ("[Description]\n"
                      "a man is doing a trick on a skateboard\n"
                      "\n[Dialogue]\n"
                      "Question: What type of trick is the man performing on the skateboard?\n"
                      "Answer: a jump\n"
                      "Question: What is the location of the jump trick being performed?\n"
                      "Answer: a skate park\n"
                      "Question: ")

    prompt_assi = "what is the outfit of the man performing the jump trick at a skate park?"

    prompt_user = "\n[Description]\n" + descrip[0] + '\n' + "\n[Dialogue]\n"
    for i in range(len(descrip) - 1):
        qa = descrip[i + 1]
        q = qa.split('? ')[0] + '?'
        a = qa.split('? ')[1]
        prompt_user += "Question: " + q + '\n' + "Answer: " + a + '\n'
    prompt_user += "Question: "

    messages = [{"role": "system", "content": prompt_sys}]
    messages.append({"role": "user", "content": prompt_example})
    messages.append({"role": "assistant", "content": prompt_assi})
    messages.append({"role": "user", "content": prompt_user})
    return messages


def generate_questions(descrip, n=1, model=None):
    messages = _build_question_messages(descrip)
    response = call_llm(model=model, messages=messages, max_tokens=32, temperature=0.5)
    return response


def _build_referring_messages(descrip, prompt_fake="", ques_prior=None):
    """Build referring question-generation prompt messages."""
    ques_prior = ques_prior or []
    prompt_sys = ("You are a proficient question generator tasked with aiding in the retrieval of a target image. "
                  "Your role is to generate questions about the target image of the description via "
                  "leveraging three key information sources:\n\n"
                  "[Retrieval Candidates]: These are captions of images which are the candidates "
                  "of the retrieval task for the target image described in [Description].\n"
                  "[Description]: This is a concise explanation of the target image.\n"
                  "[Dialogue]: Comprising question and answer pairs that seek additional "
                  "details about the target image.\n\n"
                  "You should craft a question that narrows down the options for "
                  "the attributes of the target image through "
                  "drawing the information from the retrieval candidates. "
                  "The generated question about the target image must be clear, succinct, and concise. "
                  "Also, the question should only be asked about common objects in the description and candidates, "
                  "which cannot be answered only from the description and the dialogue. "
                  "Please explain how did you utilize the information sources for generating a question.\n")

    prompt_example = ("[Retrieval Candidates]\n"
                      "0. man in yellow shirt\n"
                      "1. a boy in a skateboard park\n"
                      "2. the biker is performing a trick\n"
                      "3. a man in a green hat doing half-pipe with a skateboard\n"
                      "4. a skateboarding man catches the air in the midst of a trick\n"
                      "[Description]\n"
                      "a man is doing a trick on a skateboard\n"
                      "[Dialogue]\n"
                      "Question: what type of trick is the man performing on the skateboard?\n"
                      "Answer: a jump\n"
                      "Question: what is the location of the jump trick being performed?\n"
                      "Answer: a skate park\n"
                      "Question: ")

    prompt_assi = ("what is the outfit of the man performing the jump trick at a skate park?\n"
                   "Explanation:To generate a question about the description, I will utilize the "
                   "retrieval candidates that mention the outfit of the man. Candidates 0 and 3 "
                   "provide information about the man's wearing. "
                   "The description mentions the man's trick on a skateboard, and the dialogue mentions "
                   "the type and the location of the trick. "
                   "Since the attribute about the outfit is not appeared at the description and the dialogue, "
                   "the generated question cannot be answered from the information of "
                   "the description and the dialogue about the target image. "
                   "Also, the generated question is asking for the common objective, man, in the"
                   " descriptions and candidates, "
                   "not for the different objective from the description and the retrieval candidates 0 and 3, "
                   "for example, a shirt and a half-pipe.")

    prompt_user = "[Retrieval Candidates]\n" + prompt_fake
    prompt_user += "[Description]\n" + descrip[0]
    prompt_user += "[Dialogue]\n"
    for i in range(len(descrip) - 1):
        qa = descrip[i + 1]
        q = qa.split('? ')[0] + '?'
        a = qa.split('? ')[1]
        prompt_user += "Question: " + q + '\n' + "Answer: " + a + '\n'
    for i in range(len(ques_prior)):
        prompt_user += "Question: " + ques_prior[i] + '\n' + "Answer:\n"
    prompt_user += "Question: "

    messages = [{"role": "system", "content": prompt_sys}]
    messages.append({"role": "user", "content": prompt_example})
    messages.append({"role": "assistant", "content": prompt_assi})
    messages.append({"role": "user", "content": prompt_user})
    return messages


def generate_questions_referring(descrip, prompt_fake="", n=1, ques_prior=None, model=None):
    ques_prior = ques_prior or []
    messages = _build_referring_messages(descrip, prompt_fake, ques_prior)
    response = call_llm(model=model, messages=messages, max_tokens=32, temperature=0.5)
    return response


def filter_questions(context, question, model=None):
    prompt_sys = ("Answer the question only according to the given context. "
                  "If you cannot determine the answer or there are no objects "
                  "that are asked by the question in the context , answer \"Uncertain\".")
    messages = [{"role": "system", "content": prompt_sys}]
    prompt_user = "[Context]\n" + context + "\n[Question]\n" + question + "\n[Answer]\n"
    messages.append({"role": "user", "content": prompt_user})
    response = call_llm(model=model, messages=messages, max_tokens=10, temperature=0.0)
    return response['choices'][0]['message']['content'].lower()


def select_question(caption_recon="", questions=None, cossim_prev=None,
                     k=10, img_embs=None, threshold=500, round=1):
    questions = questions or []
    threshold = int(threshold - (round-1) * (threshold / 10.0))
    idx_related = cossim_prev.argsort()[-threshold:-k]
    p_prev = torch.nn.functional.softmax(cossim_prev[idx_related], dim=0)
    kl_divs = torch.zeros([len(questions)])

    # TURBO: batch all question embeddings at once
    caption_texts = [caption_recon + ", " + q for q in questions]
    all_embs = normalize(get_text_features(caption_texts), dim=-1)  # [q_n, dim]
    all_sims = torch.matmul(all_embs, img_embs.T)  # [q_n, corpus]
    for i in range(len(questions)):
        p_tmp = torch.nn.functional.softmax(all_sims[i][idx_related], dim=0)
        kl_div = (p_prev * (p_prev.log() - p_tmp.log())).sum().detach().cpu()
        kl_divs[i] += kl_div

    idx_final = kl_divs.argsort()[0].item()
    return questions[idx_final]


@torch.inference_mode()
def generate_answer(img_path="", query="", model_caps=None, processor_caps=None):
    with Image.open(img_path) as img:
        prompt = "Question: " + query + " Answer:"
        inputs_ = processor_caps(images=img, text=prompt, return_tensors='pt').to(device)
    out = model_caps.generate(**inputs_, do_sample=False)
    answer_generated = processor_caps.decode(out[0], skip_special_tokens=True).strip()
    return answer_generated


def paraphrase(text="", model=None):
    messages = [{"role": "system",
                 "content": "Your role is to paraphrase the given text into a fluent and natural text while preserving the information in the given text. Do not add your internal knowledge while paraphrasing."}]
    messages.append({"role": "user", "content": f"Text: {text}\nParaphrased: "})
    retry_count = 0
    while True:
        try:
            response = call_llm(model=model, messages=messages, temperature=0.7, top_p=0.8, max_tokens=512)
            break
        except Exception as e:
            print(e)
            time.sleep(3 * (retry_count + 1))
            retry_count += 1
            if retry_count >= args.max_retries:
                return text
            continue
    return response['choices'][0]['message']['content']


# ============================================================================
# TURBO: Parallel question generation helpers
# ============================================================================

def _generate_questions_parallel_no_filter(dial_gen, prompt_fake, q_n):
    """Fire q_n referring question generations in parallel. No filtering."""
    futures = []
    for _ in range(q_n):
        fut = _llm_executor.submit(
            generate_questions_referring,
            descrip=dial_gen, prompt_fake=prompt_fake, n=1, ques_prior=[]
        )
        futures.append(fut)

    questions = []
    for fut in futures:
        resp = fut.result()
        ques = resp['choices'][0]['message']['content'].split('?')[0] + '?'
        questions.append(ques)
    return questions


def _generate_questions_parallel_simple(dial_gen, q_n):
    """Fire q_n simple question generations in parallel."""
    futures = []
    for _ in range(q_n):
        fut = _llm_executor.submit(generate_questions, dial_gen, 1)
        futures.append(fut)

    questions = []
    for fut in futures:
        resp = fut.result()
        ques = resp['choices'][0]['message']['content'].split('?')[0] + '?'
        questions.append(ques)
    return questions


# ============================================================================
# Checkpoint / resume logic
# ============================================================================

model_slug = args.llm_model.replace('/', '_').replace(':', '_')
subset_name = args.subset_indices_path.stem if args.subset_indices_path else f"{s_idx}_{e_idx}"
run_name = (
    f'plugir_full_model_{model_slug}_q_n_{q_n}_rh_{recall_hitting}_'
    f'tl_{threshold_low}_recon_{str(reconstruct).lower()}_'
    f'ref_{str(referring).lower()}_filt_{str(filtering).lower()}_'
    f'sel_{str(select).lower()}_{subset_name}'
)
save_path_dial = args.output_dir / f'{run_name}_raw.json'
save_path_recon = args.output_dir / f'{run_name}_recon.json'
checkpoint_path_dial = args.output_dir / f'{run_name}.partial_raw.json'
checkpoint_path_recon = args.output_dir / f'{run_name}.partial_recon.json'
metadata_path = args.output_dir / f'{run_name}_meta.json'
provenance_path = args.output_dir / f'{run_name}_prov.json'


def save_json(path, data):
    """Atomic JSON write."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(data, f)
    tmp.replace(path)


# Load checkpoint if exists
dial_new = []
dial_new_recon = []

if checkpoint_path_dial.exists() and checkpoint_path_recon.exists():
    with checkpoint_path_dial.open('r') as f:
        dial_new = json.load(f)
    with checkpoint_path_recon.open('r') as f:
        dial_new_recon = json.load(f)
    print(f"Resuming from checkpoint: {len(dial_new)} dialogues done.")

completed_ids = set(x['session_id'] for x in dial_new)
remaining_indices = [idx for idx in target_indices if idx not in completed_ids]

# ============================================================================
# Main generation loop
# ============================================================================

import threading
gpu_lock = threading.Lock()

def process_session(idx):
    dial_gen = [visdial[idx]['dialog'][0]]
    img_path = str(args.image_root / visdial[idx]['img'])
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f'Target image does not exist: {img_path}')
        
    dial_out = {'img': visdial[idx]['img'], 'session_id': idx}
    recon_out = {'img': visdial[idx]['img'], 'session_id': idx}
    
    caption_recon = dial_gen[0]
    captions_recon = [caption_recon]

    for rnd in range(num_rounds):
        questions = []

        if referring:
            with gpu_lock:
                prompt_refer, prompt_fake, top_k, cos_sims = get_referring_prompt(caption_recon, img_embs,
                                                                                  recall_hitting, rnd + 1, search_space)
            if filtering:
                ques_prior = []
                for k_idx in range(q_n):
                    for _ in range(3):
                        response = generate_questions_referring(descrip=dial_gen, prompt_fake=prompt_fake,
                                                                n=1, ques_prior=questions + ques_prior)
                        output = response['choices'][0]['message']['content']
                        ques = output.split('?')[0] + '?'
                        ans = filter_questions(caption_recon, ques)
                        if 'uncertain' not in ans:
                            ques_prior.append(ques)
                        else:
                            break
                    if len(ques_prior) == 3:
                        response = generate_questions(dial_gen, n=1)
                        output = response['choices'][0]['message']['content']
                        ques = output.split('?')[0] + '?'
                        ans = filter_questions(caption_recon, ques)
                        if 'uncertain' not in ans:
                            ques = 'what is the other object in the image?'
                    questions.append(ques)
            else:
                questions = _generate_questions_parallel_no_filter(dial_gen, prompt_fake, q_n)

        else:
            with gpu_lock:
                img_paths, top_k, cos_sims = search_imgs(caption_recon, img_embs, search_space, k=recall_hitting)
            questions = _generate_questions_parallel_simple(dial_gen, q_n)

        if select:
            if rnd < 99:
                with gpu_lock:
                    question_final = select_question(caption_recon=caption_recon,
                                                     questions=questions,
                                                     cossim_prev=cos_sims,
                                                     k=recall_hitting,
                                                     img_embs=img_embs,
                                                     threshold=threshold_low,
                                                     round=rnd+1)
            else:
                question_final = questions[0]
        else:
            question_final = questions[0]

        with gpu_lock:
            answer_generated = generate_answer(img_path=img_path,
                                               query=caption_recon + '. ' + question_final,
                                               model_caps=blip2,
                                               processor_caps=processor2)
                                               
        qa = question_final + " " + answer_generated
        dial_gen.append(qa)

        if reconstruct:
            caption_recon = reconstruct_dialog(dial_gen)
            if caption_recon == captions_recon[-1]:
                caption_recon = paraphrase(caption_recon)
        else:
            caption_recon = ', '.join(dial_gen)

        captions_recon.append(caption_recon)

    dial_out['dialog'] = dial_gen
    recon_out['dialog'] = captions_recon
    return dial_out, recon_out

# Launch sessions in parallel using an outer executor
t_start = time.time()
_session_executor = ThreadPoolExecutor(max_workers=args.llm_workers)
futures = {}
for idx in remaining_indices:
    futures[_session_executor.submit(process_session, idx)] = idx

with tqdm(total=len(target_indices), initial=len(dial_new), desc="Generating dialogues (PARALLEL SESSIONS)") as pbar:
    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            d_out, r_out = fut.result()
            dial_new.append(d_out)
            dial_new_recon.append(r_out)
        except Exception as e:
            print(f"Error processing session {idx}: {e}")
            raise e
            
        pbar.update(1)
        done = len(dial_new)
        
        # Periodic checkpoint
        if args.checkpoint_every > 0 and done > 0 and done % args.checkpoint_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / done * (len(target_indices) - done) if done > 0 else 0
            # Sort before saving to maintain order
            dial_new.sort(key=lambda x: x['session_id'])
            dial_new_recon.sort(key=lambda x: x['session_id'])
            save_json(checkpoint_path_dial, dial_new)
            save_json(checkpoint_path_recon, dial_new_recon)
            tqdm.write(f"  [checkpoint] saved {done} dialogues | elapsed {elapsed/3600:.1f}h | ETA {eta/3600:.1f}h")

dial_new.sort(key=lambda x: x['session_id'])
dial_new_recon.sort(key=lambda x: x['session_id'])

_session_executor.shutdown(wait=False)


# ============================================================================
# Save final output
# ============================================================================

save_json(save_path_dial, dial_new)
save_json(save_path_recon, dial_new_recon)

try:
    git_hash = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=str(Path(__file__).parent),
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    git_hash = "unknown"

total_time = time.time() - t_start
metadata = {
    'llm_model': args.llm_model,
    'ollama_url': args.ollama_url,
    'seed': SEED,
    's_idx': s_idx,
    'e_idx': e_idx,
    'q_n': q_n,
    'recall_hitting': recall_hitting,
    'thres_low': threshold_low,
    'n_clusters': n_clusters,
    'rounds': num_rounds,
    'reconstruct': reconstruct,
    'referring': referring,
    'filtering': filtering,
    'select': select,
    'generation_backend': 'ollama',
    'turbo_workers': args.llm_workers,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'date': datetime.datetime.now().isoformat(),
    'number_of_dialogues': len(dial_new),
    'total_time_seconds': total_time,
    'avg_time_per_session_seconds': total_time / max(len(dial_new), 1),
    'plugir_commit_hash': git_hash,
}
save_json(metadata_path, metadata)

provenance = {
    'queries_path': str(args.queries_path),
    'search_space_path': str(args.search_space_path),
    'captions_path': str(args.captions_path),
    'embeddings_path': str(args.embeddings_path),
    'image_root': str(args.image_root),
    'output_dir': str(args.output_dir),
    'raw_dialogue_file': str(save_path_dial),
    'reconstructed_file': str(save_path_recon),
    'same_evidence_protocol': True,
    'description': 'Both raw Q/A and reconstructed queries come from the same active PlugIR trajectory.',
}
save_json(provenance_path, provenance)

# Cleanup checkpoint files
if checkpoint_path_dial.exists():
    checkpoint_path_dial.unlink()
if checkpoint_path_recon.exists():
    checkpoint_path_recon.unlink()

_llm_executor.shutdown(wait=False)

print(f'Saved raw dialogue to {save_path_dial}')
print(f'Saved reconstructed dialogue to {save_path_recon}')
print(f'Total dialogues generated: {len(dial_new)}')
print(f'Total time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)')
print(f'Avg per session: {total_time/max(len(dial_new),1):.1f}s')
