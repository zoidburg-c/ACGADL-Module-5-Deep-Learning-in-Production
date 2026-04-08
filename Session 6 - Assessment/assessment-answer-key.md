# Module 5 Assessment Answer Key: Deep Learning in Production

**Duration:** 1 hour (total)
**Format:** Individual, open-book
**Parts:** Part A (Short-Answer Questions, 20 minutes) + Part B (Practical Component, 40 minutes)

---

## Marks Breakdown

| Section | Weight |
|---------|--------|
| Part A: Short-Answer Questions (7 questions) | 30% |
| Part B: Practical Component (7 steps) | 70% |
| **Total** | **100%** |

### Part B Weighting (as per assessment criteria)

| Criteria | Weight (of total) |
|----------|-------------------|
| Correct model packaging and upload to S3 (Steps 1-3) | 20% |
| Successful endpoint deployment (Step 4) | 25% |
| Correct inference requests and results interpretation (Step 5) | 25% |
| Proper resource cleanup (Step 6) | 15% |
| Clear written explanation (Step 7) | 15% |

---

## Part A: Short-Answer Questions — Answer Key

Each question is worth approximately equal marks within the 30% Part A allocation.

---

### Question 1 — Production Pipelines (Session 1)

> **"What are the key stages of a typical ML production pipeline? Why is moving a model from a Jupyter notebook to a production environment challenging?"**

#### Model Answer

The key stages of a typical ML production pipeline are: (1) data collection, (2) preprocessing, (3) model training, (4) evaluation, (5) packaging, (6) deployment, (7) monitoring, and (8) retraining. Moving a model from a Jupyter notebook to production is challenging because production systems must handle thousands of concurrent requests (scalability), respond in milliseconds (latency), produce consistent results across different environments (reproducibility), and continue performing well as data patterns change over time (model drift). A notebook is a one-off experiment; production requires a reliable, always-on system.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Names at least 4-5 pipeline stages correctly AND explains at least 2 specific challenges (e.g., scalability, latency, reproducibility, model drift). |
| **Partial marks (~50%)** | Names 2-3 pipeline stages OR mentions challenges but without specifics (e.g., "it is hard" without explaining why). |
| **Zero marks** | Cannot name any pipeline stages; provides no meaningful explanation of challenges. |

#### Key Terms/Concepts Required for Full Marks
- At least 4 of: data collection, preprocessing, training, evaluation, packaging, deployment, monitoring, retraining
- At least 2 of: scalability, latency, reproducibility, model drift, reliability, 24/7 availability

---

### Question 2 — Model Optimisation (Session 2)

> **"What is model quantisation, and how does it reduce the size of a deep learning model? What trade-off does quantisation involve?"**

#### Model Answer

Model quantisation reduces the numerical precision of a model's weights and activations. Instead of storing each weight as a 32-bit floating point number (FP32), quantisation converts them to lower precision formats such as 16-bit (FP16), 8-bit integers (INT8), or even 4-bit integers (INT4). This reduces the model size proportionally (e.g., FP32 to INT8 gives approximately 4x size reduction) and can also speed up inference, especially on CPUs. The trade-off is a small loss in model accuracy, since the lower precision introduces rounding errors in the weight values, though in practice this loss is typically less than 1%.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Correctly explains that quantisation reduces numerical precision (e.g., FP32 to INT8), explains how this reduces size (fewer bits per weight), AND identifies the trade-off as accuracy loss. |
| **Partial marks (~50%)** | Mentions reducing precision OR size reduction but not both; or identifies the trade-off vaguely. |
| **Zero marks** | Confuses quantisation with a different technique (e.g., pruning); provides no meaningful explanation. |

#### Key Terms/Concepts Required for Full Marks
- Numerical precision / bit-width reduction (e.g., FP32 to INT8)
- Size reduction (2-4x)
- Trade-off: accuracy loss (small/minimal)

---

### Question 3 — Model Optimisation (Session 2)

> **"Compare pruning and knowledge distillation as model optimisation techniques. In what scenario would you choose one over the other?"**

#### Model Answer

Pruning removes unnecessary or near-zero weights from a trained neural network, making it sparser. It can be unstructured (removing individual weights) or structured (removing entire neurons/filters). Knowledge distillation trains a smaller "student" model to mimic the outputs of a larger "teacher" model, using the teacher's soft probability outputs which contain richer information than hard labels. You would choose pruning when you want to compress an existing model without retraining (or with minimal fine-tuning), and when you have limited compute resources. You would choose knowledge distillation when you need a fundamentally smaller model architecture and have the compute budget to train the student model, or when you want maximum compression.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Correctly describes both techniques, identifies a key difference between them, AND provides a reasonable scenario for choosing one over the other. |
| **Partial marks (~50%)** | Correctly describes one technique but not the other, or describes both without a scenario comparison. |
| **Zero marks** | Cannot distinguish between the two techniques; confuses them with other methods. |

#### Key Terms/Concepts Required for Full Marks
- Pruning: removing weights/neurons, sparsity
- Knowledge distillation: teacher-student, soft labels
- Scenario distinction (e.g., pruning = no retraining needed; distillation = need fundamentally smaller model)

---

### Question 4 — Cloud Deployment (Session 3)

> **"Why are containers (e.g., Docker) important for deploying machine learning models? What specific problem do they solve in the deployment workflow?"**

#### Model Answer

Containers (such as Docker) package a model together with all its dependencies (Python version, libraries, system packages) into a single, portable unit that runs identically on any machine. They solve the "it works on my machine" problem -- where a model that runs perfectly on the data scientist's laptop fails on a different server because of mismatched library versions, missing dependencies, or different operating system configurations. Containers ensure reproducibility and consistency across development, testing, and production environments, and they also enable easy scaling by running multiple container instances.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Explains that containers package code + dependencies together AND identifies the specific problem they solve (environment inconsistency / "works on my machine" problem / dependency management). |
| **Partial marks (~50%)** | Mentions containers or Docker but only vaguely describes the benefit (e.g., "makes deployment easier") without identifying the specific problem. |
| **Zero marks** | Cannot explain what containers do; confuses containers with virtual machines without distinguishing their purpose. |

#### Key Terms/Concepts Required for Full Marks
- Packaging code + dependencies / isolation
- Reproducibility / consistency across environments
- "Works on my machine" problem / dependency conflicts

---

### Question 5 — Cloud Deployment / AWS (Session 3)

> **"What does AWS SageMaker provide for machine learning practitioners? Describe the basic steps required to deploy a trained model to a SageMaker real-time endpoint."**

#### Model Answer

AWS SageMaker is a fully managed machine learning service that provides tools for building, training, and deploying ML models at scale. It handles infrastructure management, auto-scaling, and monitoring. The basic steps to deploy a model to a SageMaker real-time endpoint are: (1) save the trained model and package it as a `model.tar.gz` archive, (2) upload the model artifacts to Amazon S3, (3) create a SageMaker Model object that points to the S3 location and specifies the inference container/image, (4) create an endpoint configuration specifying the instance type and count, and (5) deploy the endpoint, which provisions the infrastructure and makes the model available for real-time inference via an HTTPS API.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Correctly describes what SageMaker provides (managed ML service) AND lists at least 3-4 of the deployment steps in a logical order (package model, upload to S3, create model/endpoint configuration, deploy endpoint). |
| **Partial marks (~50%)** | Describes SageMaker generally but lists only 1-2 deployment steps, or lists steps out of order without clear understanding. |
| **Zero marks** | Cannot describe SageMaker's purpose; provides no deployment steps. |

#### Key Terms/Concepts Required for Full Marks
- Managed ML service / infrastructure management
- At least 3 of: package model (model.tar.gz), upload to S3, create SageMaker Model, endpoint configuration, deploy endpoint
- Real-time inference / API endpoint

---

### Question 6 — On-Premise Deployment (Session 4)

> **"When would an organisation prefer on-premise deployment over cloud deployment for its ML models? Give two examples of situations where on-premise is the better choice."**

#### Model Answer

An organisation would prefer on-premise deployment when it has strict data privacy or regulatory requirements that prevent sending data to external cloud servers, or when it needs to minimise latency for real-time applications where network round-trips to the cloud are too slow. Two examples: (1) A hospital or healthcare provider that processes patient medical records cannot send protected health information to external cloud services due to regulatory compliance (e.g., HIPAA, PDPA). (2) A manufacturing plant that uses ML for real-time quality control on a production line needs sub-millisecond inference, which requires models running on local edge devices rather than relying on internet connectivity to a cloud endpoint.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Identifies at least one valid reason for on-premise deployment AND provides two distinct, reasonable examples with clear justification. |
| **Partial marks (~50%)** | Identifies a valid reason but provides only one example, or provides two examples without clear justification. |
| **Zero marks** | Cannot articulate why on-premise would be preferred; examples are irrelevant or missing. |

#### Key Terms/Concepts Required for Full Marks
- At least one of: data privacy/security, regulatory compliance, latency, cost control (for predictable workloads), no internet dependency
- Two distinct, concrete examples

---

### Question 7 — Retrieval-Augmented Generation (Session 5)

> **"What is Retrieval-Augmented Generation (RAG), and what problem does it solve for large language models? Name the key components of a RAG pipeline."**

#### Model Answer

Retrieval-Augmented Generation (RAG) is a technique that enhances LLMs by first retrieving relevant information from an external knowledge base (such as documents or databases) and then providing that information as context to the LLM when generating a response. RAG solves three key problems: (1) hallucination -- LLMs sometimes fabricate facts, but RAG grounds answers in actual documents; (2) outdated knowledge -- LLMs only know information up to their training cutoff, but RAG can access current documents; (3) no access to private data -- LLMs cannot answer questions about proprietary information unless that data is provided via retrieval. The key components of a RAG pipeline are: a document loader, a text splitter (chunking), an embedding model, a vector database/store, a retriever, and an LLM for generation.

#### Marking Rubric

| Level | Criteria |
|-------|----------|
| **Full marks** | Correctly explains what RAG is (retrieve then generate), identifies at least 2 problems it solves (hallucination, outdated knowledge, no private data access), AND names at least 3-4 key components of the pipeline. |
| **Partial marks (~50%)** | Explains RAG at a high level but identifies only one problem it solves, or names fewer than 3 components. |
| **Zero marks** | Cannot explain what RAG is; confuses RAG with fine-tuning or another technique. |

#### Key Terms/Concepts Required for Full Marks
- Retrieval + generation (two-phase process)
- At least 2 of: hallucination, outdated knowledge, private data access
- At least 3 of: document loader, text splitter/chunking, embeddings, vector store/database, retriever, LLM

---

## Part B: Practical Component — Answer Key

**Task:** Deploy a pre-trained text classification model (DistilBERT sentiment classifier) to AWS SageMaker and run inference.

---

### Step 1 — Load a Pre-Trained Model

> Load a pre-trained DistilBERT sentiment classification model from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`) and its tokenizer. Verify the model works locally by running a test prediction.

#### Expected Code

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Verify locally with a test prediction
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
test_result = classifier("This is a great product!")
print(test_result)
```

#### Expected Output

```
[{'label': 'POSITIVE', 'score': 0.9998}]
```

#### Notes on Partial Credit
- Full credit: Model and tokenizer both load correctly, test prediction runs and produces a sensible result.
- Partial credit: Model loads but student does not verify with a test prediction, or uses a slightly different approach (e.g., manual tokenization + forward pass instead of `pipeline`).
- Zero credit: Model does not load; wrong model name used with no correction.

#### Common Mistakes
- Using `AutoModel` instead of `AutoModelForSequenceClassification` (this loads the base model without the classification head).
- Not installing or importing `transformers` library.
- Typos in the model name.

---

### Step 2 — Package the Model Artifacts

> Save the model and tokenizer files, then package them into a `model.tar.gz` archive in the format required by SageMaker.

#### Expected Code

```python
import os
import tarfile

# Save model and tokenizer to a local directory
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

# Verify saved files
print("Saved files:", os.listdir(model_dir))

# Package into model.tar.gz
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add(model_dir, arcname=".")

print("model.tar.gz created successfully")
print(f"Archive size: {os.path.getsize('model.tar.gz') / (1024*1024):.1f} MB")
```

#### Expected Output

```
Saved files: ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt']
model.tar.gz created successfully
Archive size: ~250-270 MB
```

#### Notes on Partial Credit
- Full credit: Both model and tokenizer are saved, and the tar.gz archive is created correctly with the right structure (files at root, not nested in a subdirectory unless using `arcname`).
- Partial credit: Model is saved but packaging is incorrect (e.g., wrong archive format, files nested incorrectly).
- Zero credit: No attempt at saving or packaging.

#### Common Mistakes
- Forgetting to include the tokenizer files in the archive.
- Not using `arcname="."` in `tar.add()`, which can create a nested directory structure that SageMaker does not expect.
- Using `zip` instead of `tar.gz`.

---

### Step 3 — Upload to S3

> Upload the `model.tar.gz` file to an Amazon S3 bucket using the SageMaker Python SDK or Boto3.

#### Expected Code

```python
import sagemaker
import boto3

# Create a SageMaker session
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()
prefix = "distilbert-sentiment"

# Upload model artifacts to S3
model_artifact_url = sagemaker_session.upload_data(
    path="model.tar.gz",
    bucket=bucket,
    key_prefix=prefix
)

print(f"Model uploaded to: {model_artifact_url}")
```

**Alternative using Boto3:**

```python
import boto3

s3 = boto3.client("s3")
bucket = "your-sagemaker-bucket"
s3_key = "distilbert-sentiment/model.tar.gz"

s3.upload_file("model.tar.gz", bucket, s3_key)
model_artifact_url = f"s3://{bucket}/{s3_key}"
print(f"Model uploaded to: {model_artifact_url}")
```

#### Expected Output

```
Model uploaded to: s3://sagemaker-<region>-<account-id>/distilbert-sentiment/model.tar.gz
```

#### Notes on Partial Credit
- Full credit: Model successfully uploaded to S3 with a valid S3 URI returned.
- Partial credit: Code is correct but upload fails due to AWS permissions/configuration issues. **If the student's code is structurally correct (creates session, specifies bucket, calls upload), award partial credit even if AWS infrastructure is unavailable.**
- Zero credit: No attempt at uploading; code has fundamental errors.

#### Common Mistakes
- Not having the correct IAM permissions for S3 access.
- Hardcoding a bucket name that does not exist.
- Forgetting to create or reference the SageMaker session.

---

### Step 4 — Deploy to a SageMaker Endpoint

> Using the SageMaker Python SDK, create a `HuggingFaceModel` and deploy it to a real-time endpoint. Use an appropriate instance type (e.g., `ml.m5.large`).

#### Expected Code

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

role = sagemaker.get_execution_role()

# Create a HuggingFaceModel
huggingface_model = HuggingFaceModel(
    model_data=model_artifact_url,
    role=role,
    transformers_version="4.37",
    pytorch_version="2.1",
    py_version="py310",
    env={
        "HF_TASK": "sentiment-analysis"
    }
)

# Deploy to a real-time endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="distilbert-sentiment-endpoint"
)

print(f"Endpoint deployed: {predictor.endpoint_name}")
```

#### Expected Output

```
---------!
Endpoint deployed: distilbert-sentiment-endpoint
```

(The `--------!` progress indicator may vary; deployment typically takes 3-8 minutes.)

#### Notes on Partial Credit
- Full credit: Endpoint deploys successfully and is reachable.
- Partial credit: Code is structurally correct (creates HuggingFaceModel with correct parameters, calls `.deploy()` with instance type) but deployment fails due to AWS quota limits, region restrictions, or IAM permission issues. **Award significant partial credit (15-20% out of 25%) if the code structure is correct but AWS infrastructure prevents completion.**
- Partial credit: Student uses a slightly different approach (e.g., `Model` class instead of `HuggingFaceModel`) but the approach is valid.
- Zero credit: No deployment code; fundamentally incorrect approach.

#### Common Mistakes
- Using incompatible version combinations for `transformers_version`, `pytorch_version`, and `py_version`.
- Not specifying the `HF_TASK` environment variable (may cause the endpoint to fail on inference).
- Using an instance type that is not available in the student's region or exceeds their quota.
- Not waiting for the endpoint to be fully deployed before attempting inference.

---

### Step 5 — Run Inference

> Send the following three test sentences to your endpoint and display the predicted sentiment and confidence score for each:
> 1. `"This product is absolutely wonderful and exceeded my expectations."`
> 2. `"The service was terrible and I will never come back."`
> 3. `"The weather today is partly cloudy with a chance of rain."`

#### Expected Code

```python
test_sentences = [
    "This product is absolutely wonderful and exceeded my expectations.",
    "The service was terrible and I will never come back.",
    "The weather today is partly cloudy with a chance of rain."
]

for sentence in test_sentences:
    result = predictor.predict({"inputs": sentence})
    print(f"Input: {sentence}")
    print(f"Result: {result}")
    print("-" * 60)
```

#### Expected Output

```
Input: This product is absolutely wonderful and exceeded my expectations.
Result: [{'label': 'POSITIVE', 'score': 0.9998}]
------------------------------------------------------------
Input: The service was terrible and I will never come back.
Result: [{'label': 'NEGATIVE', 'score': 0.9997}]
------------------------------------------------------------
Input: The weather today is partly cloudy with a chance of rain.
Result: [{'label': 'NEUTRAL' or 'NEGATIVE', 'score': ~0.5-0.7}]
------------------------------------------------------------
```

**Note:** The third sentence is intentionally neutral/ambiguous. DistilBERT-SST2 is trained on a binary sentiment dataset (POSITIVE/NEGATIVE only), so it will classify this as one or the other with lower confidence. This is a good discussion point.

#### Notes on Partial Credit
- Full credit: All three sentences are sent to the endpoint, results are displayed with both label and score, and the student correctly observes the predictions.
- Partial credit: Inference code is correct but endpoint was not deployed (from Step 4 failure). **If Step 4 failed due to AWS issues but the inference code is structurally correct, award partial credit.**
- Partial credit: Student sends fewer than 3 sentences but the approach is correct.
- Zero credit: No inference code; fundamentally wrong API call format.

#### Common Mistakes
- Using the wrong payload format (e.g., not wrapping input in `{"inputs": ...}`).
- Not handling the response format correctly (results are typically a list).
- Confusing `predictor.predict()` with `predictor.invoke()` or other method names.

---

### Step 6 — Clean Up Resources

> Delete all AWS resources you created to avoid ongoing charges.

#### Expected Code

```python
import boto3

# 1. Delete the SageMaker endpoint
predictor.delete_endpoint()
print("Endpoint deleted.")

# 2. Delete the endpoint configuration
sm_client = boto3.client("sagemaker")
sm_client.delete_endpoint_config(
    EndpointConfigName="distilbert-sentiment-endpoint"  # or the auto-generated name
)
print("Endpoint configuration deleted.")

# 3. Delete the SageMaker model
# (The model name can be retrieved from the deployment output)
sm_client.delete_model(ModelName=huggingface_model.name)
print("SageMaker model deleted.")

# 4. Delete the model artifacts from S3
s3 = boto3.client("s3")
s3.delete_object(Bucket=bucket, Key=f"{prefix}/model.tar.gz")
print("S3 artifacts deleted.")

print("\nAll resources cleaned up successfully!")
```

**Simpler alternative (also acceptable):**

```python
# Delete endpoint (this is the most important step to stop charges)
predictor.delete_endpoint(delete_endpoint_config=True)
print("Endpoint and configuration deleted.")

# Delete S3 artifacts
sagemaker_session.delete_s3_object(bucket, f"{prefix}/model.tar.gz")
print("S3 artifacts deleted.")
```

#### Expected Output

```
Endpoint deleted.
Endpoint configuration deleted.
SageMaker model deleted.
S3 artifacts deleted.

All resources cleaned up successfully!
```

#### Notes on Partial Credit
- Full credit: Student deletes the endpoint, endpoint configuration, SageMaker model, and S3 artifacts (all four resources).
- Partial credit: Student deletes the endpoint (most critical for cost) but misses one or more of the other resources.
- Partial credit: If Steps 3-5 failed due to AWS issues and no resources were created, award credit if the student writes correct cleanup code that would work if resources existed.
- Zero credit: No cleanup code; student does not attempt to delete any resources.

#### Common Mistakes
- Forgetting to delete the endpoint configuration (separate from the endpoint itself).
- Not deleting S3 artifacts (these incur storage charges, though small).
- Deleting resources in the wrong order (endpoint must be deleted before endpoint configuration in some SDK versions).

---

### Step 7 — Reflection

> In 2-3 sentences, describe one challenge you encountered during this practical exercise and how you resolved it.

#### Model Answer (example)

"One challenge I encountered was that the SageMaker endpoint took over 5 minutes to deploy, and initially I thought the deployment had failed. I resolved this by checking the endpoint status in the AWS Console and seeing that it was still in the 'Creating' state. I learned that endpoint deployment involves provisioning an EC2 instance, pulling the container image, and loading the model, which takes time."

#### Notes on Partial Credit
- Full credit: Identifies a specific, genuine challenge and describes how they addressed or resolved it. The reflection demonstrates understanding of the deployment process.
- Partial credit: Identifies a challenge but the resolution is vague or the challenge is trivial (e.g., "I had a typo").
- Zero credit: No reflection provided; answer is completely unrelated to the exercise.

#### Common Mistakes
- Writing a generic answer that does not reference anything specific to the exercise.
- Claiming no challenges were encountered (unlikely and suggests lack of engagement).

---

## Grading Notes for Instructors

### AWS Infrastructure Issues

AWS-dependent steps (Steps 3-6) may fail due to reasons outside the student's control:

- **IAM permission errors:** The student's AWS account may not have the correct role or permissions.
- **Service quota limits:** SageMaker endpoint deployment requires instance quota approval, which may not be configured.
- **Region availability:** Some instance types are not available in all regions.
- **Budget/billing issues:** The student's AWS account may have spending limits.

**Guidance:** If a student's code is structurally correct and demonstrates understanding of the deployment workflow, award substantial partial credit even if AWS services are unavailable. Specifically:

| Scenario | Recommended Credit |
|----------|-------------------|
| Code is correct AND AWS steps succeed | Full marks |
| Code is correct BUT AWS steps fail due to infrastructure | 60-75% of the step's marks |
| Code has minor errors (e.g., wrong parameter name) but approach is correct | 40-60% of the step's marks |
| Code is fundamentally wrong or missing | 0% of the step's marks |

### Time Management

- Part A should take approximately 20 minutes (under 3 minutes per question).
- Part B should take approximately 40 minutes, but AWS deployment delays may extend this.
- If AWS deployment is slow, allow students to move to Step 7 (reflection) while waiting.

### Total Marks Summary

| Component | Weight |
|-----------|--------|
| Part A: Q1 | ~4.3% |
| Part A: Q2 | ~4.3% |
| Part A: Q3 | ~4.3% |
| Part A: Q4 | ~4.3% |
| Part A: Q5 | ~4.3% |
| Part A: Q6 | ~4.3% |
| Part A: Q7 | ~4.3% |
| Part B: Steps 1-3 (Packaging + S3) | 20% |
| Part B: Step 4 (Deployment) | 25% |
| Part B: Step 5 (Inference) | 25% |
| Part B: Step 6 (Cleanup) | 15% |
| Part B: Step 7 (Reflection) | 15% |
| **Total** | **~100%** |

---

*Module 5: Deep Learning in Production | SMU Advanced Certificate in Generative AI and Deep Learning*
