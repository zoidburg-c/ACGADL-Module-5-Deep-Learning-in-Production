# Instructor Guide: Session 3 - Deploying Models to the Cloud

**Module 5: Deep Learning in Production**
**Duration:** 4 hours (60 min theory + 180 min hands-on)

---

## Session Overview

This is the **heaviest AWS dependency session** in the entire programme. Students will take a pre-trained ResNet-18 PyTorch model through the full cloud deployment lifecycle: packaging model artifacts, writing inference scripts, understanding Docker containers, deploying to a live Amazon SageMaker endpoint, testing inference, monitoring with CloudWatch, and cleaning up resources.

The session is designed around SageMaker's built-in PyTorch container, which avoids the need for students to build and push custom Docker images. However, the Dockerfile exercise (Exercise 2) ensures students understand what happens under the hood.

**Critical instructor responsibility:** All AWS infrastructure must be verified working **before** class begins. A single IAM permission issue or quota limit can derail the entire hands-on portion.

---

## Learning Objectives

By the end of this session, students will be able to:

1. **Explain** why cloud deployment is necessary to move ML models from development to production
2. **Describe** the role of Docker containers in ensuring reproducible ML deployments
3. **Identify** the key AWS services (S3, ECR, SageMaker, CloudWatch, IAM) involved in a model deployment pipeline and how they interact
4. **Package** a PyTorch model as a `model.tar.gz` artifact with a custom inference script (`model_fn`, `input_fn`, `predict_fn`, `output_fn`)
5. **Deploy** a real-time inference endpoint on Amazon SageMaker using the built-in PyTorch container
6. **Test** a live endpoint by sending inference requests and interpreting predictions
7. **Monitor** endpoint health and performance metrics using Amazon CloudWatch
8. **Apply** cost-conscious practices by deleting endpoints and cleaning up AWS resources after use

---

## Pre-Session Checklist

This checklist is **critical**. Complete every item at least 24 hours before class. A single missing permission or quota limit will block students during the hands-on exercises.

### AWS Account and Permissions

- [ ] AWS account with active billing verified (not a suspended or new account with pending verification)
- [ ] IAM role for SageMaker created with the following managed policies attached:
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
  - `AmazonEC2ContainerRegistryFullAccess`
  - `CloudWatchFullAccess`
- [ ] IAM role trust policy includes `sagemaker.amazonaws.com` as a trusted entity
- [ ] If students use individual accounts: verify each student's IAM role ARN is ready and shared before class
- [ ] If using a shared account: verify concurrent endpoint limits (default is 2-4 per instance type)

### Service Quotas

- [ ] Quota for `ml.m5.large` for endpoint usage verified in the AWS Service Quotas console
  - Navigate to: Service Quotas > Amazon SageMaker > `ml.m5.large for endpoint usage`
  - Default quota is typically 2-4 instances; request an increase if running a large class
- [ ] **Fallback instance type:** Verify quota for `ml.t2.medium` as a backup (slower but functional; ~$0.065/hr)
- [ ] If quota increase is pending: prepare a pre-recorded video of a successful deployment as backup

### S3 Bucket

- [ ] Verify that the default SageMaker S3 bucket exists (format: `sagemaker-{region}-{account-id}`) or that students have permission to create one
- [ ] Confirm no S3 Block Public Access settings prevent SageMaker from reading model artifacts

### Local Environment

- [ ] Docker Desktop installed and running (required for understanding Exercise 2, though students will not build images)
- [ ] Python 3.8+ with `boto3`, `sagemaker`, `torch`, `torchvision`, `flask`, and `requests` installed
- [ ] AWS CLI configured or environment variables set (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`)

### Connectivity Test

Run this command to verify AWS connectivity:

```bash
python -c "import boto3; sts = boto3.client('sts'); print(sts.get_caller_identity())"
```

Expected output: a JSON object containing `Account`, `UserId`, and `Arn`. If this fails, AWS credentials are not configured correctly.

### Additional Verification

```bash
python -c "
import sagemaker
session = sagemaker.Session()
print(f'Default bucket: {session.default_bucket()}')
print(f'Region: {session.boto_region_name}')
"
```

### Pre-Recorded Fallback Materials

- [ ] Screen recording of a successful endpoint deployment (covering the 5-10 minute provisioning wait)
- [ ] Screen recording of CloudWatch metrics dashboard with live endpoint data
- [ ] Screenshots of the SageMaker console showing endpoint status transitions (Creating -> InService)

---

## Timing Breakdown

### Theory (60 minutes)

| Time | Topic | Duration | Notebook Section |
|------|-------|----------|-----------------|
| 0:00 - 0:05 | Session introduction and learning objectives | 5 min | Cell 0 |
| 0:05 - 0:20 | Why deploy to the cloud? Cloud vs on-premise trade-offs | 15 min | Section 1.1 |
| 0:20 - 0:35 | Containers and Docker: concepts, Dockerfiles, ML-specific challenges | 15 min | Section 1.2 |
| 0:35 - 0:50 | AWS for ML deployment: S3, ECR, SageMaker, CloudWatch, IAM overview | 15 min | Sections 1.3-1.4 |
| 0:50 - 1:00 | MLOps basics: monitoring, data drift, model degradation | 10 min | Section 1.5 |

### Hands-On (180 minutes)

| Time | Activity | Notebook Section | Duration |
|------|----------|-----------------|----------|
| 1:00 - 1:10 | Setup: install dependencies, verify AWS credentials | Setup cells | 10 min |
| 1:10 - 1:40 | Exercise 1: Load ResNet-18, write inference.py, package model.tar.gz, upload to S3 | Exercise 1 | 30 min |
| 1:40 - 2:10 | Exercise 2: Write and understand a Dockerfile for ML inference | Exercise 2 | 30 min |
| **2:10 - 2:20** | **BREAK (10 minutes)** | | **10 min** |
| 2:20 - 3:05 | Exercise 3: Deploy to SageMaker (includes 5-10 min endpoint provisioning wait) | Exercise 3 | 45 min |
| 3:05 - 3:35 | Exercise 4: Test the live endpoint with sample inputs and real images | Exercise 4 | 30 min |
| 3:35 - 3:50 | Exercise 5: Monitor with CloudWatch (logs, metrics, alarms) | Exercise 5 | 15 min |
| 3:50 - 4:00 | Exercise 6: Cleanup all AWS resources and local files, recap | Exercise 6 | 10 min |

> **Break timing:** Schedule the break at the 2-hour mark, after the Dockerfile exercise and before the SageMaker deployment. This gives students a mental reset before the most intensive (and most error-prone) part of the session. If Exercise 1 runs long, shift the break to after Exercise 1 instead.

> **Endpoint provisioning wait (5-10 min):** During Exercise 3, after students call `pytorch_model.deploy()`, there is a 5-10 minute wait while SageMaker provisions the instance. Use this time productively -- see the "What to Discuss While Waiting" section below.

---

## Key Talking Points

### Section 1.1: Why Deploy to the Cloud?

- Open with the problem statement: "You have trained a model on your laptop. It works. Now what? Your laptop cannot serve 1000 concurrent users, cannot stay online 24/7, and does not have the compute for fast GPU inference."
- **Analogy:** "Training a model is like developing a recipe in your home kitchen. Deployment is like opening a restaurant. You need industrial-grade equipment, a reliable location, and the ability to serve many customers at once."
- Walk through the deployment patterns comparison table (always-on endpoints, serverless, batch). Emphasise that today we focus on always-on endpoints -- the most common pattern for real-time applications.
- **Cost awareness -- plant the seed early:** "A single ml.m5.large endpoint costs about $0.13 per hour. That is $3.12 per day or about $94 per month if left running. Always delete endpoints when you are done."
- **Discussion prompt:** "When would you choose cloud deployment over running the model on your own servers? What are the trade-offs?" (Lead into the comparison table in the notebook.)

### Section 1.2: Containers and Docker

- **Analogy:** "A Docker container is like a shipping container. It does not matter whether you are shipping televisions or oranges -- the container standardises the packaging so any truck, ship, or crane can handle it. Similarly, a Docker container standardises your software so any cloud provider can run it."
- Walk through the key Docker terms table (Dockerfile, Image, Container, Registry) using the food analogy from the notebook.
- Explain why ML models are *especially* difficult to deploy without containers: CUDA versions, PyTorch versions, Python versions, system libraries. "Has anyone spent a day debugging a `CUDA version mismatch` error? Containers eliminate that entirely."
- Show the sample Dockerfile and walk through each line. Do not rush this -- students need to understand the layered structure (base image, dependencies, code, entrypoint).
- **Key clarification:** "We will write a Dockerfile in Exercise 2 to understand the concepts, but for the actual SageMaker deployment in Exercise 3, we will use a built-in container. SageMaker maintains official PyTorch containers so you do not have to build your own."

### Section 1.3-1.4: AWS Services and SageMaker Deep Dive

- Draw or display the architecture diagram from the notebook showing how S3, ECR, SageMaker, and CloudWatch interact. Trace the data flow: "Your model artifacts go to S3. Your container image goes to ECR (or we use a built-in one). SageMaker pulls both, provisions compute, and creates an endpoint. CloudWatch collects logs and metrics."
- Walk through the five-step SageMaker deployment workflow: package model, create SageMaker model, create endpoint config, create endpoint, send requests.
- **Emphasise the `model.tar.gz` convention:** "SageMaker has a very specific expected format. Your model weights and inference code must be packaged as a tar.gz archive with a specific directory structure. Getting this wrong is one of the most common deployment failures."
- **Discussion prompt:** "Why does SageMaker separate the model artifacts from the container? What advantage does this give you?" (Answer: you can update the model without rebuilding the container, and you can use the same container for different models.)

### Section 1.5: MLOps Basics

- Keep this section high-level and motivational. "Deploying a model is not the end -- it is the beginning of a new set of problems."
- Explain the three categories of monitoring: infrastructure metrics (is the server healthy?), model quality metrics (is the model still accurate?), and data quality metrics (has the input distribution changed?).
- **Analogy:** "Deploying a model without monitoring is like launching a satellite and throwing away the radio. You need to know if something goes wrong."
- Preview what students will do in Exercise 5: basic CloudWatch monitoring, not full Model Monitor setup.

---

## Live Demo Suggestions

### Demo 1: AWS Credential Verification (during Setup, ~3 min)

Run the `boto3.client('sts').get_caller_identity()` check live. Show what a successful response looks like. Then briefly show what happens when credentials are missing or expired -- this helps students diagnose their own issues.

### Demo 2: Model Packaging (during Exercise 1, ~5 min)

Walk through the `inference.py` file live. Explain each handler function:
- `model_fn`: "This is called once when SageMaker starts your container. It loads the model weights from disk."
- `input_fn`: "This is called for every request. It converts the raw HTTP request body into a PyTorch tensor."
- `predict_fn`: "This runs the model on the preprocessed input."
- `output_fn`: "This converts the model output back to JSON for the HTTP response."

Show the `model.tar.gz` structure: "model.pth at the root, inference.py inside a code/ directory. SageMaker is very particular about this structure."

### Demo 3: Dockerfile Walkthrough (during Exercise 2, ~5 min)

Open the generated Dockerfile and walk through it line by line. Highlight the multi-stage pattern: base image, system dependencies, Python dependencies, application code, health check, entrypoint. "This is exactly what SageMaker's built-in container does for you, but understanding this helps you debug issues when things go wrong."

### Demo 4: SageMaker Deployment (during Exercise 3, ~10 min)

This is the critical demo. Walk through the `PyTorchModel` configuration parameters one by one, then call `deploy()`. While the endpoint provisions (5-10 minutes), use the "What to Discuss While Waiting" plan below.

Show the SageMaker console in a browser tab:
1. Navigate to SageMaker > Endpoints to show the status transitioning from "Creating" to "InService"
2. Click into the endpoint to show the instance type, creation time, and URL

### Demo 5: Live Inference (during Exercise 4, ~5 min)

Send a test prediction to the live endpoint and show the response. Then send a real image (download one from the internet) and show the top-5 ImageNet predictions. "This is what your users would experience -- an HTTP request goes in, a prediction comes back in milliseconds."

### Demo 6: CloudWatch Dashboard (during Exercise 5, ~5 min)

Open CloudWatch in the AWS console. Navigate to the log group for the endpoint and show the inference logs. Then show the metrics (Invocations, ModelLatency, OverheadLatency). "These metrics tell you if your endpoint is healthy and performing well."

### What to Discuss While Waiting for Endpoint Provisioning (5-10 min)

When `pytorch_model.deploy()` is running, the class will be waiting. Use this time productively:

1. **Architecture review (3 min):** "While we wait, let us trace what is happening behind the scenes. SageMaker is: (a) registering our model object, (b) creating an endpoint configuration that specifies the instance type, (c) provisioning an ml.m5.large instance, (d) pulling the PyTorch container image, (e) downloading our model.tar.gz from S3, (f) starting the inference server, and (g) running a health check before marking the endpoint as InService."

2. **Cost discussion (2 min):** "This ml.m5.large instance costs about $0.13 per hour. If you were running this in production with auto-scaling, what would your monthly bill look like? What if you needed a GPU instance like ml.g4dn.xlarge at $0.74 per hour?"

3. **Show the SageMaker console (2 min):** Open the endpoint in the AWS console and show the status transitioning. "Notice the endpoint status says 'Creating'. This is normal. SageMaker is doing a lot of work behind the scenes."

4. **Fallback questions (3 min):** "While we wait, any questions about the model packaging or Dockerfile exercises? Anything unclear about the inference.py handler functions?"

### CRITICAL Fallback Plans

| Scenario | Fallback |
|----------|----------|
| Endpoint provisioning fails due to quota limits | Switch to `ml.t2.medium` instance type ($0.065/hr, slower but works). Update the `deploy()` call: `instance_type="ml.t2.medium"` |
| Endpoint provisioning fails for unknown reasons | Show pre-recorded video of successful deployment. Walk through the SageMaker console screenshots. |
| Docker not installed on student machines | Exercise 2 is write-and-understand only -- students write the Dockerfile but do not build it. No Docker required for Exercises 3-6. |
| AWS credentials expired mid-session | Run `aws sts get-session-token` to refresh. If using SageMaker Studio, restart the kernel. |
| S3 upload fails | Check bucket permissions. Verify the default SageMaker bucket exists. Try creating a new bucket with a unique name. |
| Multiple students deploy simultaneously and hit concurrent endpoint limits | Stagger deployments in groups of 2-3. Or have students work in pairs sharing one endpoint. |

---

## Common Student Questions and Answers

### "Why is the endpoint taking so long to provision?"

SageMaker is doing significant work behind the scenes during those 5-10 minutes: allocating an EC2 instance, pulling the Docker container image (which can be several gigabytes for PyTorch), downloading your model artifacts from S3, loading the model into memory, starting the inference server, and running health checks. This is a one-time cost -- once the endpoint is InService, individual inference requests take milliseconds.

In production, you would provision endpoints ahead of time and keep them running. For development and testing, the wait is unavoidable but only happens once per deployment.

### "How much will this cost?"

The `ml.m5.large` instance we use in this session costs approximately **$0.12-$0.13 per hour**. For the duration of this class (roughly 2 hours of endpoint uptime), that is about $0.25. If you forget to delete the endpoint and leave it running for a month, that becomes approximately $86-$94.

For reference, common SageMaker instance costs (ap-southeast-1 region):
- `ml.t2.medium`: ~$0.065/hr (CPU only, good for testing)
- `ml.m5.large`: ~$0.13/hr (CPU, 2 vCPUs, 8 GB RAM)
- `ml.m5.xlarge`: ~$0.26/hr (CPU, 4 vCPUs, 16 GB RAM)
- `ml.g4dn.xlarge`: ~$0.74/hr (1 NVIDIA T4 GPU)
- `ml.g5.xlarge`: ~$1.41/hr (1 NVIDIA A10G GPU)

**Always delete endpoints when you are done.** Set up AWS billing alerts as a safety net.

### "What if I get a quota limit error?"

AWS imposes default quotas on SageMaker instance types to prevent accidental large-scale spending. If you see an error like `ResourceLimitExceeded`, try these steps:

1. **Immediate fix:** Switch to a different instance type. Change `instance_type="ml.m5.large"` to `instance_type="ml.t2.medium"` in the `deploy()` call. This is a smaller, cheaper instance that works for our ResNet-18 model.

2. **Request a quota increase:** Go to the AWS Service Quotas console, search for "SageMaker", find the specific instance type, and click "Request quota increase". Increases for small instances are usually approved within minutes to hours.

3. **Check for existing endpoints:** You may have hit the quota because a previous endpoint is still running. Go to SageMaker > Endpoints in the console and delete any endpoints you no longer need.

### "Can I use a GPU instance instead?"

Yes, but it is not necessary for our ResNet-18 model, which is small enough to run on CPU. GPU instances make sense when:
- Your model is large (e.g., a transformer with billions of parameters)
- You need low latency for batch predictions
- You are serving multiple models on the same instance

For this session, `ml.m5.large` (CPU) is sufficient and keeps costs low. If you want to experiment with GPU deployment, try `ml.g4dn.xlarge` (NVIDIA T4 GPU, ~$0.74/hr).

### "Why do we write a Dockerfile if we are not building it?"

Understanding Docker is essential for three reasons:

1. **Debugging:** When a SageMaker deployment fails, the error logs reference container behaviour. If you understand Dockerfiles, you can diagnose problems like missing dependencies or incorrect file paths.

2. **Customisation:** The built-in SageMaker containers work for standard use cases. For anything non-standard (custom preprocessing, multiple models, non-Python dependencies), you need to build your own container.

3. **Portability:** The Dockerfile knowledge transfers directly to other platforms (Google Cloud Vertex AI, Azure ML, Kubernetes). In Session 4, we will use Docker directly for on-premise deployment.

---

## Troubleshooting Table

| # | Problem | Symptom | Cause | Solution |
|---|---------|---------|-------|----------|
| 1 | **Docker not running** | `Cannot connect to the Docker daemon` or Docker commands fail | Docker Desktop is not started, or Docker service is not running | Start Docker Desktop. On Linux: `sudo systemctl start docker`. On macOS/Windows: launch Docker Desktop from Applications. Verify with `docker info`. Note: Docker is only needed if students attempt to build the container in Exercise 2 -- Exercises 3-6 do not require Docker locally. |
| 2 | **ECR push failure (auth expired)** | `no basic auth credentials` or `denied: Your authorization token has expired` when pushing to ECR | ECR authentication tokens expire after 12 hours | Re-authenticate: `aws ecr get-login-password --region {region} \| docker login --username AWS --password-stdin {account}.dkr.ecr.{region}.amazonaws.com`. This is only relevant if students build custom containers. |
| 3 | **SageMaker quota exceeded** | `ResourceLimitExceeded: An error occurred... Account-level service limit 'ml.m5.large for endpoint usage' is 0` | Default quota for the instance type is 0 or already consumed | Switch to `ml.t2.medium` as fallback. Request quota increase in AWS Service Quotas console. Check for and delete any existing endpoints consuming quota. |
| 4 | **Endpoint creation timeout** | Endpoint stays in "Creating" status for more than 15 minutes, or transitions to "Failed" | Container fails health check, model loading error, or infrastructure issue | Check CloudWatch logs: navigate to `/aws/sagemaker/Endpoints/{endpoint-name}` log group. Common sub-causes: (a) `inference.py` has a syntax error, (b) model file path mismatch, (c) missing Python dependencies. Delete the failed endpoint and redeploy after fixing. |
| 5 | **IAM permission denied** | `AccessDeniedException` or `is not authorized to perform` errors | The SageMaker execution role is missing required policies | Verify the IAM role has `AmazonSageMakerFullAccess`, `AmazonS3FullAccess`, and `CloudWatchFullAccess` policies. Check the role's trust relationship includes `sagemaker.amazonaws.com`. Use the IAM Policy Simulator to test specific actions. |
| 6 | **S3 bucket name conflict** | `BucketAlreadyOwnedByYou` or `BucketAlreadyExists` error when creating a bucket | S3 bucket names are globally unique; someone else already has that name | Use the SageMaker default bucket (`sagemaker.Session().default_bucket()`), which includes the account ID and region. Or create a bucket with a unique suffix (e.g., timestamp or random string). |
| 7 | **Model artifact upload failure** | `NoSuchBucket`, `AccessDenied`, or upload hangs | Bucket does not exist, wrong region, or insufficient S3 permissions | Verify the bucket exists in the correct region: `aws s3 ls s3://{bucket-name}`. Check that the IAM role has `s3:PutObject` permission. Ensure the `model.tar.gz` file was created correctly (not empty): `tar tzf model.tar.gz` should list `model.pth` and `code/inference.py`. |
| 8 | **CloudWatch logs not appearing** | No log group or log streams visible for the endpoint | Endpoint has not received any requests yet, IAM role missing CloudWatch permissions, or logs have not propagated | Wait 1-2 minutes after sending requests for logs to appear. Verify the log group name matches the pattern `/aws/sagemaker/Endpoints/{endpoint-name}`. Ensure the IAM role has `CloudWatchFullAccess`. Try refreshing the CloudWatch console. |
| 9 | **`model.tar.gz` structure incorrect** | Endpoint fails with `ModuleNotFoundError: No module named 'inference'` or `model.pth not found` | The archive does not match SageMaker's expected directory structure | The correct structure is: `model.pth` at the root and `code/inference.py` inside a `code/` directory. Verify with `tar tzf model.tar.gz`. Repackage if incorrect. Common mistake: nesting files inside an extra directory. |
| 10 | **Inference returns 500 error** | `ModelError: An error occurred (ModelError) when calling the InvokeEndpoint operation` | Error in `inference.py` handler functions (input parsing, model execution, or output formatting) | Check CloudWatch logs for the Python traceback. Common sub-causes: (a) input tensor shape mismatch, (b) JSON serialisation error, (c) model expects different input format than what was sent. Fix `inference.py`, repackage, re-upload, and redeploy. |
| 11 | **boto3 / sagemaker version mismatch** | `AttributeError`, `TypeError`, or unexpected API behaviour | Outdated or incompatible SDK versions | Upgrade both: `pip install --upgrade boto3 sagemaker`. The notebook was tested with `sagemaker>=2.200.0`. Check versions with `python -c "import boto3, sagemaker; print(boto3.__version__, sagemaker.__version__)"`. |
| 12 | **Endpoint stuck in "Updating" or "Deleting"** | Cannot delete or modify the endpoint | Previous operation is still in progress | Wait for the operation to complete (check status in SageMaker console). If stuck for more than 30 minutes, contact AWS Support or use the AWS CLI: `aws sagemaker describe-endpoint --endpoint-name {name}` to check status. |

---

## Cleanup Reminder

**This section is non-negotiable.** Dedicate the final 10 minutes of class to cleanup and physically verify that every student has deleted their endpoint.

### Cleanup Checklist (project on screen)

Tell students to run the cleanup cells in Exercise 6 and verify each step:

1. **Delete the SageMaker endpoint** -- this stops the billing clock
2. **Delete the endpoint configuration** -- removes the instance type mapping
3. **Delete the SageMaker model object** -- removes the model registration
4. **Delete the S3 model artifact** -- removes the `model.tar.gz` from S3
5. **Clean up local files** -- remove `model.pth`, `model.tar.gz`, and the `docker_example/` directory

### Verification Command

Have students run this after cleanup to confirm:

```python
import boto3
sm = boto3.client('sagemaker')
endpoints = sm.list_endpoints()['Endpoints']
print(f"Active endpoints: {len(endpoints)}")
for ep in endpoints:
    print(f"  - {ep['EndpointName']} ({ep['EndpointStatus']})")
```

If the list is empty or contains no endpoints from this session, cleanup is complete.

### Instructor Final Check

After class, log into the AWS console and verify no student endpoints are still running. If any are found, delete them immediately to avoid charges.

---

## Transition to Session 4

> "Today we deployed a model to the cloud using AWS SageMaker. We packaged a ResNet-18 model, wrote inference handlers, created a live endpoint, tested it with real requests, and monitored it with CloudWatch. This is the standard deployment pattern used by thousands of companies worldwide.
>
> But cloud deployment is not the only option. Some organisations cannot use the cloud -- they have data privacy requirements, regulatory constraints, or simply want to avoid recurring cloud costs. Others want to run models on edge devices, in hospitals, or in factories where internet connectivity is unreliable.
>
> In **Session 4: Deploying Models On-Premise**, we will flip the script entirely. Instead of relying on managed services like SageMaker, we will deploy models using Docker and Docker Compose on your own machine -- the same infrastructure you would use on a company server. We will set up NVIDIA GPU support for on-premise inference, build a complete inference API from scratch, and compare the trade-offs between cloud and on-premise deployment.
>
> The Docker knowledge from today's Exercise 2 will be directly relevant -- in Session 4, you will actually build and run those containers.
>
> Before next session, make sure Docker Desktop is installed and running on your machine. If you have an NVIDIA GPU, install the NVIDIA Container Toolkit as well."

---

## Session-Specific Instructor Notes

### Managing the AWS Wait Times

This session has two significant wait points that require planning:

1. **Endpoint provisioning (5-10 min):** After `deploy()` is called in Exercise 3. Use the "What to Discuss While Waiting" plan from the Live Demo Suggestions section.

2. **CloudWatch metric propagation (1-2 min):** After sending test requests in Exercise 4, metrics may take 1-2 minutes to appear in CloudWatch. Use this time to explain the difference between real-time logs and aggregated metrics.

### Cost Management for Large Classes

If you have 20+ students each deploying their own endpoint:
- Total cost per hour: 20 x $0.13 = $2.60/hr (manageable)
- Concurrent endpoint limit may be an issue: default is 2-4 per instance type per account
- **Solution for shared accounts:** Have students work in pairs, or stagger deployments in groups of 5

### Instance Type Fallback Strategy

If `ml.m5.large` is unavailable due to quota limits:

| Preference | Instance Type | Cost/hr | Notes |
|-----------|--------------|---------|-------|
| 1st choice | `ml.m5.large` | ~$0.13 | Default for this session, 2 vCPUs, 8 GB RAM |
| 2nd choice | `ml.t2.medium` | ~$0.065 | Cheaper, burstable CPU, sufficient for ResNet-18 |
| 3rd choice | `ml.m4.large` | ~$0.12 | Older generation, similar performance to m5.large |
| Last resort | Pre-recorded demo | $0.00 | Show the video, discuss the concepts, and have students complete the notebook with simulated outputs |

### Key Files Students Create

By the end of the session, students should have created:
- `model.pth` -- saved ResNet-18 model weights
- `code/inference.py` -- SageMaker inference handler script with `model_fn`, `input_fn`, `predict_fn`, `output_fn`
- `model.tar.gz` -- packaged model artifact uploaded to S3
- `docker_example/Dockerfile` -- ML inference Dockerfile (understanding exercise, not built)
- `docker_example/app.py` -- Flask inference server (understanding exercise)
- `docker_example/requirements.txt` -- Python dependencies for the Docker container
