# Module 5: Deep Learning in Production

**Programme:** Advanced Certificate in Generative AI and Deep Learning (ACGADL)
**Institution:** Singapore Management University (SMU Academy)
**Duration:** 16 hours across 5 sessions + assessment

## Overview

This module teaches participants how to take trained deep learning models from Jupyter notebooks to real-world production systems. Starting with the fundamentals of ML production pipelines, the module progressively covers model optimisation techniques, cloud deployment with AWS SageMaker, on-premise deployment with Ollama and Apple MLX, and building Retrieval-Augmented Generation (RAG) systems with LangChain and LlamaIndex. By the end of the module, participants will be able to package, deploy, and serve ML models as scalable, production-ready services.

**Primary frameworks:** PyTorch, FastAPI, LangChain, LlamaIndex
**Cloud platform:** AWS (SageMaker, S3, CloudWatch, Bedrock)
**Local LLM tools:** Ollama, Apple MLX
**Domain applications:** Image classification, sentiment analysis, RAG chatbots

## Prerequisites

- Completion of Modules 1-4 of the ACGADL programme
- Python 3.8+ installed locally
- AWS account with the following services enabled: SageMaker, S3, CloudWatch, Bedrock (with model access for Claude or Titan)
- Hardware: 8GB+ RAM recommended; Apple Silicon (M1/M2/M3/M4) recommended for Session 4 (MLX exercises) but not required
- Prior knowledge: basic ML/DL concepts (covered in earlier modules), Python proficiency

## Setup

Before starting, complete the following setup steps:

1. **AWS configuration:** Follow the [AWS Setup Guide](setup/aws-setup-guide.md) to configure your AWS account, IAM roles, and service access.
2. **Local LLM setup:** Follow the [Ollama Setup Guide](setup/ollama-setup-guide.md) to install Ollama and pull the required models.
3. **Python dependencies:** Install all required packages:
   ```bash
   pip install -r setup/requirements.txt
   ```

## Module Structure

| Session | Topic | Duration | Key Topics |
|---------|-------|----------|------------|
| 1 | [Introduction to Production Pipelines](Session%201%20-%20Introduction%20to%20Production%20Pipelines.ipynb) | 1 hour | ML production lifecycle, key challenges (scalability, latency, reproducibility, model drift), model serialisation, building a REST API with FastAPI |
| 2 | [Model Optimisation Techniques](Session%202%20-%20Model%20Optimisation%20Techniques.ipynb) | 1 hour | Quantisation (dynamic, static, QAT), pruning (structured, unstructured), knowledge distillation, size/speed/accuracy trade-offs |
| 3 | [Deploying Models to the Cloud](Session%203%20-%20Deploying%20Models%20to%20the%20Cloud.ipynb) | 4 hours | Docker containers, AWS ecosystem for ML, SageMaker deployment (model packaging, S3 upload, endpoint creation), CloudWatch monitoring, resource cleanup |
| 4 | [Deploying Models On-Premise](Session%204%20-%20Deploying%20Models%20On-premise.ipynb) | 4 hours | On-premise vs cloud trade-offs, running LLMs with Ollama, Apple MLX framework, building a local inference server with FastAPI |
| 5 | [Implementing RAG](Session%205%20-%20Implementing%20RAG.ipynb) | 6 hours | Embeddings, vector databases (FAISS, ChromaDB), LangChain fundamentals, building a complete RAG pipeline, LlamaIndex, AWS Bedrock integration |
| 6 | [Assessment](Session%206%20-%20Assessment.md) | 1 hour | Part A: Short-answer questions (Sessions 1-5), Part B: Practical SageMaker deployment exercise |

## Directory Structure

```
Module 5 Deep Learning in Production/
├── README.md
├── Session 1 - Introduction to Production Pipelines.ipynb
├── Session 2 - Model Optimisation Techniques.ipynb
├── Session 3 - Deploying Models to the Cloud.ipynb
├── Session 4 - Deploying Models On-premise.ipynb
├── Session 5 - Implementing RAG.ipynb
├── Session 6 - Assessment.md
├── setup/
│   ├── aws-setup-guide.md
│   ├── ollama-setup-guide.md
│   └── requirements.txt
├── instructor-guides/
│   ├── session-1-instructor-guide.md
│   ├── session-2-instructor-guide.md
│   ├── session-3-instructor-guide.md
│   ├── session-4-instructor-guide.md
│   ├── session-5-instructor-guide.md
│   └── session-6-assessment-answer-key.md
└── outputs/
    └── session-*-slides/
```

## Quick-Start for Instructors

### 1 Week Before Delivery

- [ ] Complete the full AWS setup using `setup/aws-setup-guide.md` -- ensure IAM roles, S3 buckets, SageMaker access, and Bedrock model access are all configured.
- [ ] Install Ollama following `setup/ollama-setup-guide.md` and pull the required models (e.g., `ollama pull llama3.2:1b`).
- [ ] Run `pip install -r setup/requirements.txt` on the teaching machine to install all Python dependencies.
- [ ] Run through each notebook end-to-end to verify all cells execute successfully in your environment.
- [ ] Verify AWS service quotas support the instance types used in Sessions 3 and 6 (e.g., `ml.m5.large` for SageMaker endpoints).

### Day Before Delivery

- [ ] Verify all AWS services are accessible and running (no service outages or billing holds).
- [ ] Confirm Ollama is running locally (`ollama list` should show the pulled models).
- [ ] Test a few representative notebook cells from each session to catch any library version changes.
- [ ] Prepare the assessment environment: verify the DistilBERT model downloads correctly and SageMaker endpoints can be deployed.
- [ ] Review the relevant instructor guide for the session you are delivering.

### Day of Delivery

- [ ] Open the instructor guide for the session (`instructor-guides/session-N-instructor-guide.md`) and follow the pre-session checklist.
- [ ] Ensure all participants have AWS credentials configured and can access the required services.
- [ ] For Sessions 3-5, remind participants to **delete all AWS resources** at the end of the session to avoid ongoing charges.
- [ ] Keep the assessment answer key (`instructor-guides/session-6-assessment-answer-key.md`) ready for Session 6 grading.

## Session Flow

Each session follows a consistent structure:

- **Theory section** (20-60 minutes depending on session) -- Concepts delivered via slides and in-notebook markdown explanations.
- **Hands-on exercises** (40-180 minutes depending on session) -- Guided coding exercises in Jupyter notebooks with full solutions provided.
- **Recap** -- Summary of key concepts and preview of the next session.

## AWS Cost Estimate

| Resource | Estimated Cost |
|----------|---------------|
| SageMaker endpoint (ml.m5.large) | ~$0.115/hr while active |
| S3 storage (model artifacts) | ~$0.023/GB/month (negligible) |
| CloudWatch monitoring | Free tier covers basic usage |
| Bedrock inference (Session 5) | ~$0.001/1K tokens |

**Reminder:** Always shut down SageMaker endpoints when not in use. Remind participants to run the cleanup cells at the end of Sessions 3 and 6. A single forgotten endpoint can incur significant charges overnight.

## Total Module Duration

| Component | Hours |
|-----------|-------|
| Session 1: Introduction to Production Pipelines | 1 |
| Session 2: Model Optimisation Techniques | 1 |
| Session 3: Deploying Models to the Cloud | 4 |
| Session 4: Deploying Models On-Premise | 4 |
| Session 5: Implementing RAG | 6 |
| Session 6: Assessment | 1 |
| **Total** | **17 hours** |

---

*Module 5: Deep Learning in Production | SMU Advanced Certificate in Generative AI and Deep Learning*
