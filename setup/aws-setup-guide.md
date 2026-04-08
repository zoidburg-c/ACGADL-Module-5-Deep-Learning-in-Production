# AWS Setup Guide for Module 5: Deep Learning in Production

This guide walks you through setting up the AWS environment needed for all hands-on exercises in Module 5.

## 1. AWS Account

You need an AWS account with billing enabled. If using an organisational account, ensure you have permissions for:
- **Amazon SageMaker** (model hosting and endpoints)
- **Amazon S3** (storing model artifacts)
- **Amazon Bedrock** (foundation model access for Session 5)
- **Amazon ECR** (Elastic Container Registry, for Docker images in Session 3)
- **Amazon CloudWatch** (monitoring deployed endpoints)
- **AWS IAM** (role and policy creation)

## 2. Create an IAM User / Role

### Option A: IAM User (for local development)

If you are running notebooks on your own machine:

1. Go to **IAM** in the AWS Console
2. Click **Users** > **Create user**
3. Attach the following managed policies:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
   - `AmazonBedrockFullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
   - `CloudWatchFullAccess`
4. Create an access key under **Security credentials** > **Access keys**
5. Configure your local CLI:
   ```bash
   aws configure
   # Enter your Access Key ID, Secret Access Key, and preferred region
   ```

### Option B: SageMaker Execution Role (for SageMaker Studio / Notebooks)

If you are running notebooks inside SageMaker:

1. Go to **IAM** in the AWS Console
2. Create a new role with:
   - Trusted entity: **SageMaker**
   - Policies:
     - `AmazonSageMakerFullAccess`
     - `AmazonS3FullAccess`
     - `AmazonBedrockFullAccess`
     - `AmazonEC2ContainerRegistryFullAccess`
     - `CloudWatchFullAccess`
3. Name the role: `SageMakerExecutionRole-Module5`
4. Note the **Role ARN** -- you will need it in the Session 3 notebook

## 3. Request Bedrock Model Access

Bedrock models require explicit access approval before use. Session 5 (Implementing RAG) uses Bedrock as an optional LLM and embedding provider.

1. Go to **Amazon Bedrock** in the AWS Console
2. Navigate to **Model access** in the left sidebar
3. Click **Manage model access**
4. Enable access for:
   - **Anthropic Claude 3 Haiku** -- Used as the LLM in Session 5 RAG exercises
   - **Anthropic Claude 3.5 Sonnet** -- Optional higher-quality alternative
   - **Amazon Titan Text Embeddings V2** -- Used for embeddings in Session 5
5. Click **Save changes**
6. Wait for approval (usually instant for Amazon Titan; Anthropic models may take a few minutes)

> **Note:** Bedrock is optional for Module 5. All exercises provide a free local alternative using **Ollama** (see the Ollama Setup Guide). You only need Bedrock if you want to use AWS-hosted models.

## 4. Choose Your AWS Region

Select a region that supports both SageMaker and Bedrock. Recommended regions:

| Region | SageMaker | Bedrock (Claude) | Bedrock (Titan Embeddings) |
|--------|-----------|-------------------|---------------------------|
| `us-east-1` (N. Virginia) | Yes | Yes | Yes |
| `us-west-2` (Oregon) | Yes | Yes | Yes |
| `ap-southeast-1` (Singapore) | Yes | Limited | Yes |

We recommend **`us-east-1`** for the broadest service availability.

## 5. SageMaker Instance Recommendations

Different sessions have different compute requirements:

| Session | Activity | Recommended Instance | Notes |
|---------|----------|---------------------|-------|
| Session 1 | Local exercises (FastAPI) | No AWS instance needed | Runs on your laptop |
| Session 2 | Model optimisation | No AWS instance needed | Runs on your laptop (CPU is fine) |
| Session 3 | Deploying to SageMaker | **ml.m5.large** (endpoint) | 2 vCPU, 8 GB RAM; used for the deployed inference endpoint |
| Session 4 | On-premise deployment | No AWS instance needed | Runs locally with Ollama |
| Session 5 | RAG with Bedrock (optional) | No AWS instance needed | Bedrock is serverless; no instance to manage |

> **Key point:** The only session that requires a running SageMaker instance is **Session 3**, which deploys a PyTorch model to an `ml.m5.large` endpoint. All other sessions run locally.

## 6. Request Quota Increase for ml.m5.large

SageMaker endpoint instances require sufficient quota. New AWS accounts may have a quota of zero for `ml.m5.large`.

1. Go to **Service Quotas** in the AWS Console
2. Search for **Amazon SageMaker**
3. Find the quota: **`ml.m5.large for endpoint usage`**
4. If the current value is 0, click **Request increase on account level**
5. Request a new value of **1** (one instance is sufficient)
6. Wait for approval (typically approved within 1-2 business days)

> **Tip:** Submit quota requests early -- ideally a few days before the Session 3 class -- as approval is not instant.

## 7. Set Up Billing Alerts

Protect yourself from unexpected charges:

1. Go to **AWS Budgets** in the AWS Console (or search "Budgets")
2. Click **Create budget**
3. Choose **Cost budget** > **Monthly**
4. Set the budget amount to **$50** (provides a comfortable buffer)
5. Configure alert thresholds:
   - Alert at **50%** ($25) -- informational
   - Alert at **80%** ($40) -- warning
   - Alert at **100%** ($50) -- critical
6. Enter your email address for notifications

### Cost Estimates

| Resource | Estimated Cost | When |
|----------|---------------|------|
| SageMaker ml.m5.large endpoint | ~$0.12/hour | Session 3 only |
| S3 storage | < $0.01 | Negligible for model artifacts |
| Bedrock Claude 3 Haiku | ~$0.001 per query | Session 5 only (optional) |
| Bedrock Titan Embeddings | ~$0.0001 per query | Session 5 only (optional) |
| CloudWatch | Free tier | Session 3 monitoring |

**Estimated total cost for Module 5: $20-40** (assuming you delete endpoints promptly after exercises).

> **Important:** The single biggest cost driver is leaving a SageMaker endpoint running. Always run the cleanup cells at the end of Session 3 to delete your endpoint.

## 8. Upload Course Materials

If using SageMaker Studio:

1. Open a terminal in SageMaker Studio
2. Upload or clone the course notebooks into your home directory
3. Install additional dependencies:
   ```bash
   pip install -r requirements.txt
   ```

If running locally:

1. Ensure you have Python 3.9+ installed
2. Install dependencies:
   ```bash
   pip install -r setup/requirements.txt
   ```
3. Configure AWS credentials:
   ```bash
   aws configure
   ```

## 9. Verify Setup

Run the following in a notebook cell or Python script to verify everything works:

```python
import boto3
import sagemaker

# Check SageMaker session
session = sagemaker.Session()
print(f"SageMaker session region: {session.boto_region_name}")
print(f"Default S3 bucket: {session.default_bucket()}")

# Check IAM role (SageMaker Studio only)
try:
    role = sagemaker.get_execution_role()
    print(f"Execution role: {role}")
except ValueError:
    print("Not running in SageMaker Studio (this is fine for local development)")

# Check Bedrock access (optional, for Session 5)
try:
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    print("Bedrock client created successfully")
except Exception as e:
    print(f"Bedrock not available: {e}")
    print("(This is OK if you plan to use Ollama instead)")

# Check S3 access
s3 = boto3.client("s3")
print(f"S3 client created successfully")

print("\nAWS setup verification complete!")
```

## 10. Cost Management Tips

- **Delete SageMaker endpoints immediately** after completing Session 3 exercises. Endpoints are billed per hour, even when idle.
- **Stop SageMaker notebook instances** when not in use (if using SageMaker Studio).
- **Monitor your AWS Budgets** dashboard regularly during the module.
- **Use the cleanup cells** provided at the end of each notebook -- they delete endpoints and remove temporary resources.
- **Bedrock costs are minimal** -- each API call costs fractions of a cent, so Session 5 usage will be negligible.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Bedrock model access denied | Check Model access in the Bedrock console; ensure your region supports the model |
| SageMaker quota exceeded for ml.m5.large | Request a quota increase via Service Quotas (see Section 6) |
| S3 permission denied | Verify your IAM user/role has `AmazonS3FullAccess` |
| `NoCredentialProviders` error | Run `aws configure` and enter your access key and secret key |
| SageMaker endpoint creation timeout | Check CloudWatch logs; ensure the endpoint instance type has available quota |
| Bedrock throttling errors | Reduce request rate; consider using a smaller model (Haiku instead of Sonnet) |
| Docker permission denied (Session 3) | Ensure Docker Desktop is installed and running; add your user to the `docker` group on Linux |
| ECR push fails | Run `aws ecr get-login-password` and ensure your IAM user has ECR permissions |
