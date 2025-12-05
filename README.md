# ID2223 Lab2 Fine-tuning of LLM

Here is the link to the HuggingFace Space: https://huggingface.co/spaces/Kezovic/iris (both partners, Oliver Westfahl Knezevic and Nils Wiebe Werner, worked on this shared space).
The model used in this Lab is Llama-3.2-1B-Instruct

Note: The preview of notebook doesn't work, so you'd either have to look at the code directly or download it and open with an IDE. Sorry!

# Approaches to improving the performance of the fine-tuned LLM
# Evaluation
To evaluate the approaches tested (to see if they make any improvement to the model) we rely on the lm-eval library which provides automated benchmarks to simplify evaluation.
The benchmarks we chose are: hellaswag, arc_challenge, mmlu_stem and truthfulqa_mc2. These benchmarks focus on reasoning (hellaswag, arc_challenge), reasoning within STEM fields (mmlu_stem) and hallucination (truthfulqa_mc2), these are factors we believe to be relevant when it comes to finetuning an LLM on an instruction dataset for general purpose. 

# Model-centric approaches:

From a model-centric approach there are a multitude of potential ways to improve performance ranging in time and computational requirements.
The main ideas we tested were the following:

Tuning hyperparameters:
We experimented with the following parameters: (Swapping LR Scheduler to cosine, Learning Rates [2e-2, 2e-3, 2e-5], Weight Decay [0.1, 0.001], Rank and Alpha in the LoRA adapters [32, 64]. (Our tests kept the relationship between r and a at 1:1))
Based on the benchmarks we selected for evaluation we did not find any configuration that improved performance. This is a typical constraint with this approach, it generally requires testing quite a large set of configurations to find one that actually improves performance. For the scope of this assignment we deemed it unfeasible to do an extensive grid or random search due to time and computational limitations and simply tried a few different variations (which did not improve the model).


Changing fine-tuning architecture from QLoRA to QDoRA:
We tried using DoRA (Weight-Decay LoRA) which is based on this paper (https://arxiv.org/abs/2402.09353), it is supposed to "enhance both the learning capacity and training stability of LoRA while avoiding any additional inference overhead". The main architectural change this entails is that it splits weights into magnitude and direction, letting the model learn both a low-rank directional update and adjust the weight norms derived from the pretrained weights.
The main drawback from this technique is that training time is significantly slower as there are more computations to run for each step (specifically the normalization of the weight norms). 
From our evaluation we did not see any notable improvement on the benchmarks from this new fine-tuning architecture. (It's possible that better selection of parameters for DoRA would be necessary)


Changing from QLoRA to LoRA to improve the precision of the base model to 16-bit: 
The final approach was to increase the precision of the loaded model and removing quantization noise by using LoRA instead of QLoRA by not loading the model parameters in 4 bit and isntead in full 16 bit precision. This comes at the cost of higher VRAM during training but since our choice of model is very small (1B parameter model) this is a lesser concern (due to inference being on CPU). 
With this change it resulted in improved performance most notably on ARC challenge and HellaSwag and an average improvement of 1.04%. 

| Benchmark     | Base Score | LoRA Score | Improvement |
|---------------|------------|------------|----------------|
| ARC Challenge | 0.3652     | 0.3788     | 0.0136         |
| HellaSwag     | 0.5798     | 0.5916     | 0.0128         |
| mmlu_stem     | 0.3635     | 0.3717     | 0.0092         |
| TruthfulQA    | 0.4367     | 0.4382     | 0.0061         |

# Data-centric approaches:
For data centric approaches, improving the model means providing better or new data that the model can learn from (improving data quality). A generally important point here is that the dataset is tailored to the domain the LLM is going to be used in.

Initially we used one of the alternative datasets (FuseChat) that can be found at: (https://github.com/mlabonne/llm-datasets). The reason for this was firstly because it did not include the multi-turn type of examples that the original dataset included which sometimes caused “both sides” hallucinations where the model would continue both sides of the conversation (which seems to be a larger issue when the model is smaller). It also has generally more short instructions + responses in its data.
However we found that by using the "FuseChat" dataset the resulting fine-tuned model saw minimal improvement, an average 0.435% on the evaluation benchmarks.

| Benchmark     | Base Score | FuseChat Score | FuseChat Improvement |
|---------------|------------|----------------|-------------------|
| ARC Challenge | 0.3652     | 0.3686         | 0.0034            |
| HellaSwag     | 0.5798     | 0.5755         | -0.0043           |
| STEM          | 0.3635     | 0.3758         | 0.0123            |
| TruthfulQA    | 0.4367     | 0.4427         | 0.0060            |



Since our application of the LLM is focused on poetry generation, another interesting alternative we explored was to additionally fine-tune the model on the "Creative Writing" dataset by Nitral-AI (https://huggingface.co/datasets/Nitral-AI/Creative_Writing-ShareGPT). Our hypothesis was that this might improve reasoning of the model, but more importantly would make the model better suited for the specific context of our application.
Swapping to this dataset we saw a larger improvement in the evaluation compared to the Baseline (although heavily schewed towards general reasoning, which given the dataset is expected) with an average improvement of 1.13%

| Benchmark     | Base Score | Nitral Score | Nitral Improvement |
|---------------|------------|-------------|-----------------|
| ARC Challenge | 0.3652     | 0.3763      | 0.0111          |
| HellaSwag     | 0.5798     | 0.6026      | 0.0228          |
| mmlu_stem     | 0.3635     | 0.3657      | 0.0022          |
| TruthfulQA    | 0.4367     | 0.4376      | 0.009           |


Finally we combined the two approaches that saw most success (using LoRA instead of QLoRA and training on the dataset with focus on tcreative writing).
The evaluation showed that this further improved the model performance on the benchmarks, yielding an average of 1.87%.

| Benchmark     | Base Score | Combined Approaches Score | Improvement |
|---------------|------------|---------------------|------------|
| ARC Challenge | 0.3652     | 0.384               | 0.0188     |
| HellaSwag     | 0.5798     | 0.6136              | 0.0338     |
| STEM          | 0.3635     | 0.3822              | 0.0187     |
| TruthfulQA    | 0.4367     | 0.44                | 0.0033     |


# Testing different LLMs for Inference Speed of Application
The model selected was Llama-3.2-1B-Instruct. We tried a couple of different foundational models to see the inference and training time per model. We tested models in the Llama-3 family (1B,3B,8B) since the notebook we adapted was structured to work with this family. In the end we found that inference for the 1B and 3B models was essentially the same for our application (8B being much slower), however the training times were significiantly slower when using the larger models. For this reason we chose to continue using the smallest model since we decided the added time for training was not worth the pay off in the context of the assignment.
