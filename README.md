# Mistral-7b-instruct Finetuning Code

**Model:** [`abdullah1027/mistral-7b-instruct-finetuned-maleeha-lodhi-style`](https://huggingface.co/abdullah1027/mistral-7b-instruct-finetuned-maleeha-lodhi-style)  
**Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`  
**Fine-tuned by:** Abdullah Usama  
**License:** Apache 2.0 (inherited from base model)

---

## 📌 Model Description

This model is a fine-tuned version of **Mistral-7B-Instruct-v0.3**, adapted using **LoRA (Low-Rank Adaptation)** to generate text in the **opinion-editorial style of Pakistani diplomat and journalist Maleeha Lodhi**.

The training dataset consists of **opinion articles scraped from the last 6 years of Dawn News**, with a focus on **Maleeha Lodhi's own contributions**. The objective is to emulate her analytical tone, vocabulary, and structure when discussing geopolitics, foreign policy, and Pakistani current affairs.

---

## 🧠 Model Details

- **Model Type:** Decoder-only Transformer (Causal Language Model)
- **Fine-Tuning Method:** PEFT (LoRA)
- **Language:** English
- **Domain:** Current affairs, international relations, opinion journalism
- **Training Dataset:** Custom JSONL dataset containing Maleeha Lodhi's Dawn articles  


---

## ✨ Example Uses

### ✅ Direct Use

Use this model for generating texts that mimic the style and tone of Maleeha Lodhi. Ideal for:

- Opinion editorial drafts
- Political commentary templates
- Style-transfer for current affairs writing
- Rhetorical writing exploration

### 🧩 Downstream Use

Potential integrations:

- Educational tools for analyzing political writing styles
- Automated content generation with a specific tone (human-in-the-loop required)
- Style-specific LLMs for editorial teams or research tools

---

## 🚫 Out-of-Scope Uses

Do **not** use this model for:

- Generating factual news reports without human review
- Legal, financial, or medical advice
- Critical decision-making systems
- Generating harmful, biased, or unethical content

---

## ⚠️ Bias, Risks, and Limitations

This model may:

- **Reflect ideological bias**: The model mirrors the tone and stance of Maleeha Lodhi and Dawn’s editorial perspective.
- **Hallucinate facts**: As with any LLM, it is not a reliable source of factual information.
- **Overfit stylistically**: It may perform poorly on topics outside Lodhi's usual domain (e.g. entertainment, tech).
- **Cultural specificity**: Responses may include assumptions or idioms specific to South Asian or Pakistani political discourse.

**Recommendation:** Always use a human-in-the-loop for reviewing outputs, especially when used in real-world or critical applications.

---

## 📂 Training Data & Methodology

- **Source**: Scraped Dawn.com opinion articles from the past 6 years.
- **Filtering**: Articles attributed to Maleeha Lodhi were extracted and structured into JSONL format.
- **Format**: Instruction-style format used for fine-tuning (prompt-response pairs).
- **Fine-tuning Strategy**: LoRA on `mistralai/Mistral-7B-Instruct-v0.3` using QLoRA-compatible PEFT trainer.

The dataset used for fine-tuning is included in this repo for reference and reproducibility.

---

## 🛠️ How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("abdullah1027/mistral-7b-instruct-finetuned-maleeha-lodhi-style")
tokenizer = AutoTokenizer.from_pretrained("abdullah1027/mistral-7b-instruct-finetuned-maleeha-lodhi-style")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "### Instruction:\nWrite an op-ed on the evolving role of Pakistan in regional stability.\n\n### Response:"
result = generator(prompt, max_new_tokens=400, do_sample=True, temperature=0.7)

print(result[0]['generated_text'])

```


