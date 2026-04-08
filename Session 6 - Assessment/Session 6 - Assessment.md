# Module 5 Assessment: Deep Learning in Production

**Duration:** 1 hour (total)
**Format:** Individual, open-book
**Parts:** Part A (Short-Answer Questions) + Part B (Practical Component)

---

## Part A: Short-Answer Questions (20 minutes)

Answer all questions in 2-4 sentences each.

---

**Question 1 — Production Pipelines (Session 1)**

What are the key stages of a typical ML production pipeline? Why is moving a model from a Jupyter notebook to a production environment challenging?

---

**Question 2 — Model Optimisation (Session 2)**

What is model quantisation, and how does it reduce the size of a deep learning model? What trade-off does quantisation involve?

---

**Question 3 — Model Optimisation (Session 2)**

Compare pruning and knowledge distillation as model optimisation techniques. In what scenario would you choose one over the other?

---

**Question 4 — Cloud Deployment (Session 3)**

Why are containers (e.g., Docker) important for deploying machine learning models? What specific problem do they solve in the deployment workflow?

---

**Question 5 — Cloud Deployment / AWS (Session 3)**

What does AWS SageMaker provide for machine learning practitioners? Describe the basic steps required to deploy a trained model to a SageMaker real-time endpoint.

---

**Question 6 — On-Premise Deployment (Session 4)**

When would an organisation prefer on-premise deployment over cloud deployment for its ML models? Give two examples of situations where on-premise is the better choice.

---

**Question 7 — Retrieval-Augmented Generation (Session 5)**

What is Retrieval-Augmented Generation (RAG), and what problem does it solve for large language models? Name the key components of a RAG pipeline.

---

## Part B: Practical Component (40 minutes)

**Task:** Deploy a pre-trained text classification model (DistilBERT sentiment classifier) to AWS SageMaker and run inference.

Complete the following steps in a Jupyter notebook:

---

**Step 1 — Load a Pre-Trained Model**

Load a pre-trained DistilBERT sentiment classification model from Hugging Face (`distilbert-base-uncased-finetuned-sst-2-english`) and its tokenizer. Verify the model works locally by running a test prediction.

---

**Step 2 — Package the Model Artifacts**

Save the model and tokenizer files, then package them into a `model.tar.gz` archive in the format required by SageMaker.

---

**Step 3 — Upload to S3**

Upload the `model.tar.gz` file to an Amazon S3 bucket using the SageMaker Python SDK or Boto3.

---

**Step 4 — Deploy to a SageMaker Endpoint**

Using the SageMaker Python SDK, create a `HuggingFaceModel` and deploy it to a real-time endpoint. Use an appropriate instance type (e.g., `ml.m5.large`).

---

**Step 5 — Run Inference**

Send the following three test sentences to your endpoint and display the predicted sentiment and confidence score for each:

1. `"This product is absolutely wonderful and exceeded my expectations."`
2. `"The service was terrible and I will never come back."`
3. `"The weather today is partly cloudy with a chance of rain."`

---

**Step 6 — Clean Up Resources**

Delete all AWS resources you created to avoid ongoing charges:

- Delete the SageMaker endpoint
- Delete the endpoint configuration
- Delete the SageMaker model
- Delete the model artifacts from S3

---

**Step 7 — Reflection**

In 2-3 sentences, describe one challenge you encountered during this practical exercise and how you resolved it.

---

### Assessment Criteria

| Criteria | Weight |
|---|---|
| Correct model packaging and upload to S3 | 20% |
| Successful endpoint deployment | 25% |
| Correct inference requests and results interpretation | 25% |
| Proper resource cleanup | 15% |
| Clear written explanation (Step 7) | 15% |

---

*This is an open-book assessment. You may refer to course materials, documentation, and online resources.*
