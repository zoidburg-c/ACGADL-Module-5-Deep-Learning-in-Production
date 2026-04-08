# Instructor Guide: Session 1 - Introduction to Production Pipelines

**Module 5: Deep Learning in Production**
**Duration:** 1 hour (20 min theory + 40 min hands-on)
**Prerequisites:** Modules 1-4 completed (Python, Deep Learning Basics, CNNs/NLP, Transformers/LLMs)

---

## Learning Objectives

By the end of this session, students will be able to:

1. **Explain** what "production" means for machine learning models and how it differs from experimentation
2. **Identify** the eight stages of an ML production pipeline and describe the role of each
3. **Analyze** key production challenges (scalability, latency, reproducibility, model drift) and their implications
4. **Implement** model serialization using PyTorch's `state_dict()` approach
5. **Build** a simple REST API endpoint for model serving using FastAPI

---

## Pre-Session Checklist

- [ ] Python 3.8+ installed and verified on the instructor machine
- [ ] All packages installed: `torch`, `torchvision`, `fastapi`, `uvicorn`, `Pillow`, `requests`
- [ ] Run `uvicorn --version` to confirm it is available and working
- [ ] Run the entire notebook end-to-end at least once to cache the ResNet18 weights (~44 MB download)
- [ ] Verify the FastAPI app starts correctly: save `app.py`, run `uvicorn app:app --port 8000`, and test the `/` and `/predict` endpoints
- [ ] Prepare a sample image file (e.g., `dog.jpg`) on the desktop for the live API demo
- [ ] Have a browser tab ready at `http://localhost:8000/docs` for the Swagger UI demo
- [ ] **Fallback:** Pre-record a 2-minute terminal screencast of `uvicorn` starting and a `curl` request succeeding, in case the live server fails
- [ ] **Fallback:** Take screenshots of the Swagger UI `/predict` response for use if the browser demo does not work
- [ ] Confirm internet access for downloading ImageNet class labels from GitHub

---

## Timing Breakdown

### Theory (20 minutes)

| Time | Topic | Notes |
|------|-------|-------|
| 0:00 - 0:03 | Welcome and session objectives | Frame Module 5: moving from building models to deploying them |
| 0:03 - 0:08 | What does "production" mean for ML? (Section 1.1) | Notebook vs production table; real-world examples |
| 0:08 - 0:14 | The ML production pipeline (Section 1.2) | Walk through all 8 stages; distinguish training from serving |
| 0:14 - 0:18 | Key challenges in production (Section 1.3) | Scalability, latency, reproducibility, model drift |
| 0:18 - 0:20 | Module 5 roadmap and transition to hands-on (Section 1.4) | Preview Sessions 2-5; tell students to open the notebook |

### Hands-On (40 minutes)

| Time | Activity | Notebook Section |
|------|----------|------------------|
| 0:20 - 0:27 | Exercise 1: Load a pre-trained ResNet18 model | Cells 10-12 |
| 0:27 - 0:35 | Exercise 2: Build the prediction function and test with a real image | Cells 13-20 |
| 0:35 - 0:42 | Exercise 3: Save and load the model (serialization) | Cells 21-26 |
| 0:42 - 0:55 | Exercise 4: Write and inspect the FastAPI app; instructor live demo | Cells 27-33 |
| 0:55 - 1:00 | Recap, key takeaways, and transition to Session 2 | Cell 34 |

---

## Key Talking Points

### Section 1.1: What Does "Production" Mean?

- **Core concept:** Production means your model is part of a live system that real users or applications interact with -- it must work reliably, at scale, 24/7. It is no longer a one-off experiment inside a Jupyter notebook.
- **Real-world analogy:** Think of the difference between a chef cooking a single meal at home versus running a restaurant kitchen. The recipe (model) is the same, but the restaurant needs consistent quality, speed, the ability to handle a rush of orders, and the ability to recover when something goes wrong.
- **Discussion prompt:** "Can anyone name an ML model they interacted with today? What production qualities do you think it needed?" (Guide toward: Netflix recommendations, spam filters, autocomplete, face unlock.)

### Section 1.2: The ML Production Pipeline

- **Core concept:** Production is not a single step -- it is a pipeline of eight connected stages: data collection, preprocessing, training, evaluation, packaging, deployment, monitoring, and retraining. The pipeline is a loop, not a line.
- **Real-world analogy:** It is like a manufacturing assembly line. Each station does one job well and passes the result to the next. If the quality check (monitoring) finds a defect, the line loops back for retooling (retraining).
- **Key distinction to emphasize:** Model training is a batch process (hours/days, occasional). Model serving is a real-time process (milliseconds, constant). Production systems are primarily about serving. Make sure students internalize this difference.
- **Discussion prompt:** "In today's hands-on, we skip data collection and training. Which pipeline stages do we actually cover?" (Answer: evaluation/inference, packaging, and the first step of deployment.)

### Section 1.3: Key Production Challenges

- **Core concept:** Four main challenges -- scalability (handling many requests), latency (fast responses under 200ms), reproducibility (same input yields same output), and model drift (performance degrades over time as real-world data changes).
- **Real-world analogy for model drift:** A spam filter trained in 2023 slowly becomes less effective as spammers invent new techniques in 2025. The model does not crash -- it just silently gets worse. This is why monitoring is critical.
- **Discussion prompt:** "Which of these four challenges do you think is the hardest to detect? Why?" (Guide toward model drift, because the model does not raise an error -- it just becomes quietly less accurate.)

### Section 1.4: Module 5 Roadmap

- **Core concept:** Each session builds on the previous one. Session 1 gives you the foundation (local API). Session 2 adds optimisation techniques. Sessions 3-4 move to cloud and on-premise deployment. Session 5 covers RAG pipelines.
- **Emphasize the progression:** "By the end of Module 5, you will be able to take a trained model and deploy it as a real, accessible service. That is the skill that turns a data scientist into a machine learning engineer."

### Exercise 1: Loading a Pre-trained Model

- **Key point to hammer home:** Always call `model.eval()` before using a model for inference. This disables dropout and uses running statistics for batch normalization. Forgetting this is one of the most common production bugs.
- **Discussion prompt:** "Why do we use a pre-trained model instead of training from scratch?" (Saves time, better accuracy, transfer learning, production-ready.)

### Exercise 2: The Prediction Function

- **Core concept:** The load-preprocess-infer-postprocess pattern is universal. Every production ML system follows this exact flow, whether it is classifying images, translating text, or detecting fraud.
- **Key point:** The preprocessing must exactly match what was used during training. If the model was trained on 224x224 images with specific normalization values, inference must use those same values. Mismatched preprocessing is a silent accuracy killer.

### Exercise 3: Model Serialization

- **Core concept:** Serialization is the bridge between training and serving. You train once, save the model artifact, and deploy that file to your serving infrastructure.
- **Key point:** Always use `torch.save(model.state_dict(), ...)` in production, not `torch.save(model, ...)`. The state dictionary approach is more portable and gives you explicit control over the model architecture.
- **Discussion prompt:** "Why might you want to keep multiple saved model versions?" (Rollback if new model performs worse, A/B testing, auditing.)

### Exercise 4: FastAPI Application

- **Core concept:** A REST API is the standard way to make models accessible to other applications. The client sends an HTTP request, the server runs the model, and the server sends back an HTTP response.
- **Key points to emphasize:**
  - The model loads once when the server starts, not on every request. This is critical for performance.
  - FastAPI auto-generates interactive documentation at `/docs` -- this is extremely useful for testing and for frontend developers who need to integrate with your API.
  - The response is structured JSON, which any application (mobile, web, backend) can parse.

---

## Live Demo Suggestions

### Demo 1: Running the FastAPI Server (during Exercise 4, ~8 minutes)

1. **Setup:** After the notebook writes `app.py`, switch to a terminal window (keep the notebook visible on one side if you have a wide screen or second monitor).
2. **Start the server:** Run `uvicorn app:app --host 0.0.0.0 --port 8000`. Narrate what you see: the startup log, the "Uvicorn running on..." message.
3. **Health check:** In a second terminal tab, run `curl http://localhost:8000/`. Show the JSON response. Explain: "This is how load balancers check if your service is alive."
4. **Swagger UI:** Open `http://localhost:8000/docs` in a browser. Walk through the interactive documentation. Click on the POST `/predict` endpoint, click "Try it out", upload a sample image, and execute. Show the JSON response with the top-5 predictions.
5. **Python client (optional):** If time permits, show a quick Python `requests.post()` call to demonstrate programmatic access.
6. **Stop the server:** Press Ctrl+C and explain that in production, this would be managed by a process supervisor or container orchestrator.

**Fallback plan:** If `uvicorn` fails to start (port conflict, import error, missing dependency):
- First, try a different port: `uvicorn app:app --port 8001`
- If that fails, show the pre-recorded terminal screencast
- If neither works, show the screenshots of the Swagger UI and walk through what the response would look like. Emphasize: "The pattern is what matters -- the code is correct, and you will run this on your own machine after class."

### Demo 2: Prediction Consistency Check (during Exercise 3, ~2 minutes)

1. After saving and loading the model, run the verification cell live.
2. Highlight the "VERIFIED: Loaded model produces IDENTICAL predictions!" message.
3. Ask: "What would you do if the predictions did NOT match?" (Answer: check that `model.eval()` was called, verify the save/load paths, check for version mismatches.)

---

## Common Student Questions and Answers

### "Why FastAPI instead of Flask?"

FastAPI is built on modern Python features (async/await, type hints) and provides automatic request validation, automatic API documentation (Swagger UI), and significantly better performance than Flask for I/O-bound workloads. Flask is still widely used and perfectly fine, but FastAPI has become the preferred choice for new ML serving projects because of its built-in features and speed. In production, both are typically placed behind a reverse proxy like Nginx anyway.

### "Can I serve multiple models from the same API?"

Yes. You can load multiple models at startup and create separate endpoints for each (e.g., `/predict/resnet`, `/predict/vgg`). Alternatively, you can accept a model name as a parameter and route to the appropriate model. In production, many teams use a model registry to manage which models are loaded and versioned.

### "How is this different from deploying on AWS SageMaker?"

What we built today is a local development server -- it runs on your machine only. SageMaker (covered in Session 3) handles everything we are doing manually: it loads your model, wraps it in an API, provisions servers, handles scaling, and manages HTTPS. Think of today's exercise as understanding the mechanics that SageMaker automates for you.

### "What happens if the server crashes while processing a request?"

With our simple setup, the request is lost. In production, you use: (1) process managers like `gunicorn` that restart crashed workers automatically, (2) multiple server replicas behind a load balancer so one crash does not take down the whole service, and (3) request queues that buffer requests and retry on failure. We will cover some of these patterns in later sessions.

### "Why do we save just the state_dict instead of the whole model?"

Saving the entire model with `torch.save(model, ...)` uses Python's `pickle` module, which embeds references to the exact class definition and module paths. If you refactor your code or change file names, loading breaks. The `state_dict()` approach saves only the weight tensors as a dictionary, so you can load them into any compatible architecture regardless of how your code is organized. This is more portable and is the recommended approach for production.

---

## Troubleshooting Table

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `ModuleNotFoundError: No module named 'fastapi'` | FastAPI not installed in the active Python environment | Run `pip install fastapi uvicorn` in the same environment the notebook is using. Verify with `python -c "import fastapi"`. |
| `uvicorn: command not found` | `uvicorn` not on the system PATH, or installed in a different virtual environment | Run `pip install uvicorn` or use `python -m uvicorn app:app --port 8000` instead. |
| `Address already in use` when starting uvicorn | Another process is using port 8000 | Use a different port: `uvicorn app:app --port 8001`. Or find and kill the existing process: `lsof -i :8000` then `kill <PID>`. |
| `RuntimeError: Error(s) in loading state_dict` when loading saved model | Model architecture mismatch between save and load, or the file is corrupted | Ensure you create the model with `models.resnet18()` (no weights) before calling `load_state_dict()`. Verify the file was saved correctly by checking file size (~44 MB). |
| ResNet18 weights download hangs or times out | Slow or restricted internet connection | Download the weights file manually from the PyTorch model hub and place it in the torch cache directory (`~/.cache/torch/hub/checkpoints/`). Alternatively, have a USB drive with the cached weights ready. |
| `PIL.UnidentifiedImageError` when testing prediction | Image URL is broken or the downloaded file is not a valid image | Try a different image URL. For local files, verify the format with `file your_image.jpg`. Ensure `.convert("RGB")` is called since some images may be RGBA or grayscale. |
| Predictions differ between original and loaded model | `model.eval()` was not called on the loaded model | Always call `loaded_model.eval()` after `load_state_dict()`. Without it, dropout and batch norm behave differently, producing inconsistent outputs. |
| FastAPI `/predict` endpoint returns 422 Unprocessable Entity | Request is not sending the file correctly (wrong field name or content type) | Ensure the `curl` command uses `-F "file=@image.jpg"` (not `-d`). In Python `requests`, use `files={"file": open("image.jpg", "rb")}`. The field name must match the parameter name in the endpoint function. |

---

## Transition to Next Session

> "Today we built a working mini production pipeline: we loaded a pre-trained model, created a prediction function, serialized the model to disk, and wrapped it all in a FastAPI application. You now understand the core pattern behind every model serving system in production.
>
> But we have a problem. Our ResNet18 model is 44 MB and runs on CPU. What if we need to deploy to a mobile phone with limited memory? What if we need sub-50ms latency? What if we are paying by the millisecond for cloud GPU time and want to cut costs?
>
> In Session 2, we tackle **model optimisation** -- techniques like quantisation, pruning, and knowledge distillation that shrink models and speed up inference while keeping accuracy nearly intact. We will take the same ResNet18 we used today and see how much smaller and faster we can make it. The goal: remove redundancy without sacrificing what matters."
