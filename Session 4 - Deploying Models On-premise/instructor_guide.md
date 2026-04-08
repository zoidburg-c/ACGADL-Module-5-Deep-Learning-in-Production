# Session 4: Deploying Models On-Premise -- Instructor Guide

**Module 5 | Advanced Certificate in Generative AI and Deep Learning | SMU Academy**

---

## Session Overview

| Item | Detail |
|---|---|
| **Duration** | 4 hours total |
| **Theory** | 60 minutes |
| **Hands-on** | 180 minutes (Jupyter notebook) |
| **Domain** | On-premise ML deployment |
| **Frameworks** | Ollama, Apple MLX, FastAPI |
| **Model** | Llama 3.2 (1B) via Ollama; mlx-community models via MLX |
| **Pre-requisite** | Sessions 1--3 of Module 5 completed |

**Special Note:** This session has **two parallel tracks** based on student hardware. Read the Platform Split Management section carefully before the session.

---

## Learning Objectives

By the end of this session, learners will be able to:

1. **Explain** why and when on-premise deployment is preferred over cloud-based deployment, citing data privacy, latency, and cost considerations
2. **Compare** cloud vs on-premise deployment trade-offs across dimensions such as setup effort, scaling, data privacy, and cost
3. **Deploy** a large language model locally using Ollama and interact with it via its REST API
4. **Demonstrate** Apple MLX inference on Apple Silicon hardware, including loading quantised models and measuring generation speed
5. **Build** a local inference server using FastAPI that wraps Ollama with logging, rate limiting, and concurrent request handling
6. **Evaluate** hardware and framework options for on-premise deployment using a structured decision framework

---

## Pre-Session Checklist

### Instructor Preparation (Do 24 hours before)

- [ ] Install Ollama on your machine (`curl -fsSL https://ollama.com/install.sh | sh` on Linux/Mac, or download from ollama.com)
- [ ] Pull the model: `ollama pull llama3.2:1b` (approximately 1.3 GB download)
- [ ] Verify Ollama is running: `curl http://localhost:11434/api/tags` should return a JSON response listing the model
- [ ] Install Python packages: `pip install requests fastapi uvicorn psutil aiohttp`
- [ ] If you have an Apple Silicon Mac: `pip install mlx mlx-lm`
- [ ] Run through the entire notebook end-to-end to verify all cells execute
- [ ] Prepare a backup terminal with Ollama running and a pre-generated response ready, in case of live demo issues
- [ ] Check classroom WiFi bandwidth -- students will need to download ~1.3 GB for the Ollama model

### Student Pre-Session Instructions (Send 48 hours before)

Send students the following instructions:

> **Before Session 4, please complete these steps:**
>
> 1. **Install Ollama** from https://ollama.com -- it is available for Mac, Linux, and Windows
> 2. **Pull the model** by running: `ollama pull llama3.2:1b` (this downloads ~1.3 GB)
> 3. **Check your chip**: On Mac, go to Apple menu > About This Mac. If it says M1, M2, M3, or M4, you have Apple Silicon
> 4. **Install Python packages**: `pip install requests fastapi uvicorn psutil aiohttp`
> 5. **Apple Silicon Mac users only**: also run `pip install mlx mlx-lm`
>
> If you have trouble with any step, come to class 15 minutes early and we will help you.

### Hardware Identification

- [ ] Survey class for Apple Silicon Mac users (needed for Track A/B split decision)
- [ ] Identify any students on Windows or Intel Mac who may need extra help with Ollama installation
- [ ] Ensure at least one fallback laptop has both Ollama and MLX working for demos

---

## Platform Split Management

### Decision Flowchart: Which Track to Use

```
                    Survey the class:
              "Who has an Apple Silicon Mac?"
                         |
            +------------+------------+
            |                         |
     Majority have              Minority have
     Apple Silicon              Apple Silicon
     (>50% of class)            (<50% of class)
            |                         |
            v                         v
   Run both tracks           +--------+--------+
   in parallel               |                 |
   (full split)         Only 1-2           3-5 students
                        students           have Apple Silicon
                             |                 |
                             v                 v
                      Run Ollama track    Run both tracks
                      for everyone        but assign a TA
                      (simplest option)   or advanced student
                      Demo MLX briefly    to lead Track A
                      on instructor Mac
```

### When to Split and When to Merge

| Time | Activity | Track |
|---|---|---|
| 0:00 -- 1:00 | Theory: on-premise concepts, hardware, frameworks | **MERGED** -- everyone together |
| 1:00 -- 1:05 | Hardware survey, form groups, explain parallel tracks | **TRANSITION** |
| 1:05 -- 2:05 | Part A (Exercises 1--2): Ollama hands-on | **MERGED** -- everyone does this together |
| 2:05 -- 2:15 | Break | -- |
| 2:15 -- 3:15 | Part B (Exercises 3--4): Apple MLX hands-on | **SPLIT** -- Track A does MLX; Track B reviews Ollama advanced features or reads through MLX code |
| 3:15 -- 3:20 | **Checkpoint:** Both tracks discuss results and compare Ollama vs MLX | **MERGED** |
| 3:20 -- 3:55 | Part C (Exercises 5--6): FastAPI server + model management | **MERGED** -- everyone uses Ollama backend |
| 3:55 -- 4:00 | Recap and transition to Session 5 | **MERGED** |

### Managing Two Parallel Tracks

**Before the split:**
- Have both tracks' notebook sections clearly bookmarked on your instructor machine
- Assign a TA or advanced student as "Track A lead" if available
- Tell Track B students: "While Track A works through MLX exercises, you will review the MLX code and explanations so you understand the concepts, even though you cannot run them"

**During the split (Part B):**
- Alternate attention between groups every 10--15 minutes
- Start with Track A (MLX users) to get them unblocked on setup, then shift to Track B
- Track B students should read the MLX cells and their explanations, and can optionally explore additional Ollama features (e.g., pulling a second model like `gemma2:2b`)

**Checkpoint moment (3:15):**
- Ask Track A: "What generation speed (tokens/sec) did you see with MLX?"
- Ask Track B: "What generation speed did you see with Ollama?"
- Facilitate a brief comparison discussion -- this reinforces why hardware choice matters

**Students who want to try both tracks:**
- Encourage this as a post-session exercise
- Apple Silicon students can install Ollama and try Track B exercises at home
- Track B students can read through the MLX code to understand the concepts
- Note: Track A exercises will simply print "MLX not available" messages on non-Apple Silicon hardware -- they will not crash

---

## Timing Breakdown

### Theory (60 minutes)

| Time | Section | Topic | Key Talking Points |
|---|---|---|---|
| 0:00 -- 0:05 | Opening | Title + Learning Objectives | Welcome, session context within Module 5. Frame the shift from cloud to on-premise. "In Sessions 1--3, we used cloud services. Today, we bring the model home." |
| 0:05 -- 0:20 | 1.1 | Why Deploy On-Premise? | Walk through the three drivers: data privacy/regulation, latency, and cost at scale. Use the cloud vs on-premise comparison table. Ask: "Can anyone think of a real scenario where sending data to the cloud is not an option?" |
| 0:20 -- 0:30 | 1.2 | Linux/Nvidia Server Deployment | Cover the standard industry stack: Nvidia GPUs, CUDA, Docker, serving frameworks (Triton, vLLM, TGI, TorchServe). Emphasise this is the production gold standard but not what we use in the hands-on today. |
| 0:30 -- 0:40 | 1.3 | Apple Silicon for ML | Explain unified memory architecture. Draw the diagram contrasting traditional CPU/GPU memory separation vs Apple Silicon shared memory. Introduce MLX and Ollama. |
| 0:40 -- 0:48 | 1.4 | Distributed Inference | Briefly cover model parallelism vs data parallelism. Show the memory requirements table (1B = 2 GB, 8B = 16 GB, 70B = 140 GB). Note this is advanced but important context. |
| 0:48 -- 0:55 | 1.5 | Choosing Your Deployment Target | Walk through the decision flowchart. Show the summary of options table. Stress: "Start simple with Ollama, scale up when needed." |
| 0:55 -- 1:00 | Transition | Hardware Survey + Track Assignment | Survey the class for Apple Silicon. Assign tracks. Direct everyone to open the notebook for Part A. |

### Hands-On (180 minutes)

| Time | Notebook Section | Exercises | Track | Instructor Actions |
|---|---|---|---|---|
| 1:00 -- 1:05 | Setup | Package installation | Both | Walk through together. Reassure non-Apple-Silicon students that MLX install failure is expected and OK. |
| 1:05 -- 1:30 | Part A | Exercise 1: Install and run Ollama | Both | Verify everyone has Ollama running. Help troubleshoot. Key moment: first time students see an LLM running on their own machine. |
| 1:30 -- 2:00 | Part A | Exercise 2: Ollama as a local API server | Both | Demonstrate the REST API, OpenAI-compatible endpoint, and streaming. Ask: "Why does the OpenAI-compatible API format matter?" |
| 2:00 -- 2:15 | Break | -- | -- | Use this time to set up MLX demo if needed. |
| 2:15 -- 2:45 | Part B | Exercises 3a--3d: MLX basics + LLM inference | **SPLIT** | Track A runs MLX exercises. Track B reads through MLX code and explores additional Ollama features. |
| 2:45 -- 3:10 | Part B | Exercise 4: MLX vs PyTorch comparison | **SPLIT** | Track A runs the benchmark. Track B continues exploring or begins Part C early. |
| 3:10 -- 3:20 | Checkpoint | Compare results across tracks | **MERGED** | Both tracks share their generation speeds and memory usage observations. |
| 3:20 -- 3:45 | Part C | Exercise 5: Build a local inference server | Both | Everyone builds the FastAPI server. This uses Ollama as the backend, so it works for all students. |
| 3:45 -- 3:55 | Part C | Exercise 6: Model management | Both | List, inspect, and manage models. Discuss disk usage implications. |
| 3:55 -- 4:00 | Recap | Key takeaways + Session 5 preview | Both | Summarise. Transition to RAG. |

---

## Key Talking Points

### Connecting to Earlier Sessions

This session marks a **strategic pivot** in Module 5. Repeatedly connect to previous sessions:

- "In Sessions 1--3, we relied on cloud providers to host our models. Today we explore the alternative: what if the model runs on hardware you control?"
- "Remember the cloud API latency from Session 2? Today we eliminate the network round-trip entirely."
- "The containerisation concepts from Session 3 apply directly here -- Nvidia Container Runtime and Docker are used for on-premise GPU deployments too."

### The Central Theme: Control vs Convenience

Frame the entire session around the trade-off: **"On-premise gives you control. Cloud gives you convenience. When does control matter more?"**

Walk through the scenarios:
- **Healthcare:** Patient data under PDPA/HIPAA cannot leave the hospital network. You must bring the model to the data.
- **Trading firms:** Proprietary algorithms and market data must stay in-house. Even milliseconds of network latency matter.
- **High-volume inference:** If you are processing millions of documents daily, paying per-API-call becomes very expensive compared to owning the hardware.

### The "Docker for LLMs" Analogy

When introducing Ollama, use this analogy:

> "Think of Docker: before Docker, deploying software meant worrying about OS versions, library conflicts, and configuration. Docker packaged everything into a container. Ollama does the same for LLMs. Instead of worrying about model formats, quantisation, and GPU drivers, you just run `ollama pull llama3.2:1b` and it works."

### Unified Memory -- Why Apple Silicon Matters

Many students will not understand why Apple Silicon is relevant for ML. Use concrete numbers:

> "An Nvidia RTX 4090 -- a $1,600 GPU -- has 24 GB of VRAM. That is your ceiling for model size. A MacBook Pro with an M4 Max chip can have 128 GB of unified memory. That means you can load a 70B parameter model in 4-bit quantisation on a laptop. You cannot do that on any consumer Nvidia GPU."

Caveat immediately: "But Apple Silicon is slower for training and for high-throughput production serving. It is best for local development, prototyping, and privacy-sensitive edge deployment."

### The OpenAI-Compatible API Pattern

When covering Exercise 2b, emphasise the practical significance:

> "Ollama supports the OpenAI chat completions API format. This is not a coincidence -- it is a deliberate design decision. It means you can write your application against the OpenAI API, test locally with Ollama, and deploy to either cloud or on-premise without changing your application code. The only thing that changes is the base URL."

This is a key architectural insight that connects to production deployment patterns.

### FastAPI Server -- Why Not Just Use Ollama Directly?

Students will ask why Exercise 5 wraps Ollama in another API layer. Explain:

> "In production, you never expose your model runtime directly to users. You put a server in front that handles authentication, rate limiting, logging, request validation, and response formatting. Ollama's API is great for development, but a production deployment needs these additional layers."

---

## Live Demo Suggestions

### Demo 1: First LLM on Your Machine (During Exercise 1)

**Setup:** Open a terminal alongside the notebook.

**Demo steps:**
1. Run `ollama run llama3.2:1b` in the terminal
2. Type a question and show the model generating a response in real time
3. Point out: "This is running entirely on your machine. No internet needed. No API key. No cloud bill."
4. Show the streaming output and note the tokens/second

**Fallback plan:** If Ollama is not responding, show a pre-recorded terminal session. Prepare a screen recording of the demo running successfully.

### Demo 2: Switching Between OpenAI and Ollama (During Exercise 2)

**Setup:** Show two code snippets side by side.

**Demo steps:**
1. Show a request to OpenAI's API (if you have a key) or a mock of it
2. Show the same request to Ollama's OpenAI-compatible endpoint
3. Highlight that only the `base_url` changes
4. Ask: "What does this pattern enable for your deployment strategy?"

**Fallback plan:** Use the notebook's Exercise 2b cell directly. The code is self-explanatory even without a live OpenAI comparison.

### Demo 3: MLX Speed (During Part B, Apple Silicon only)

**Setup:** Have Exercise 3c ready to run.

**Demo steps:**
1. Load a small MLX model on your Apple Silicon Mac
2. Generate text and show the tokens/second
3. Compare with Ollama's speed on the same machine
4. Show Activity Monitor to demonstrate unified memory usage

**Fallback plan:** If MLX installation fails or the model download is slow, show pre-captured output. Include screenshots in a backup slide.

### Demo 4: Load Testing the FastAPI Server (During Exercise 5d)

**Setup:** Have the FastAPI server running before the demo.

**Demo steps:**
1. Run the concurrent requests cell
2. Show how the server handles multiple simultaneous requests
3. Point to the logging output showing request tracking
4. Discuss: "What would happen if 100 users hit this simultaneously? How would you scale?"

**Fallback plan:** If the server fails to start, walk through the server code conceptually and show the expected output from the test cells.

---

## Common Student Questions

### "Can I run GPT-4 or Claude locally?"

**Answer:** No. GPT-4 and Claude are proprietary models that are only available through their respective cloud APIs. You cannot download their weights. However, there are strong open-source alternatives you can run locally -- Llama 3.2 (which we use today), Mistral, Gemma, and Phi are all available through Ollama. For many tasks, these open models are competitive with proprietary ones, especially at the 7B--70B parameter range.

### "How much RAM do I need to run a useful model?"

**Answer:** It depends on the model size and quantisation level. As a practical guide:
- **8 GB RAM:** Can run 1--3B parameter models (like Llama 3.2 1B or Phi-3 Mini)
- **16 GB RAM:** Can run 7--8B parameter models in 4-bit quantisation
- **32 GB RAM:** Can run 13B parameter models comfortably
- **64+ GB RAM:** Can run 30--70B parameter models in 4-bit quantisation

Apple Silicon unified memory is especially efficient here because all of it is accessible to the GPU. On a traditional system, only the GPU VRAM counts for model loading.

### "Is Ollama production-ready?"

**Answer:** Ollama is excellent for development, prototyping, and small-scale deployments. For high-throughput production serving, you would typically use vLLM (optimised for LLM throughput with techniques like PagedAttention and continuous batching) or Nvidia Triton (for multi-model serving with dynamic batching). Think of Ollama as the "development server" and vLLM/Triton as the "production server" -- analogous to Flask's development server vs Gunicorn/Nginx in web development.

### "Why is my model responding slowly on CPU?"

**Answer:** LLM inference is computationally intensive. Without a GPU or Apple Silicon's neural engine, inference runs on CPU, which is significantly slower. For Llama 3.2 1B on CPU, you might see 5--15 tokens/second, whereas a GPU or Apple Silicon can achieve 30--60+ tokens/second. If speed is critical, consider using a smaller model (like a 0.5B parameter model) or using quantisation (4-bit instead of 8-bit) to reduce the computational load. For production use without a GPU, consider a cloud API instead.

### "What is quantisation and why does it help?"

**Answer:** Quantisation reduces the precision of model weights from 32-bit or 16-bit floating point to 8-bit or 4-bit integers. This reduces memory requirements by 2--8x and speeds up inference, with only a small quality degradation. Think of it like JPEG compression for images -- you lose some fidelity, but the file is much smaller and loads much faster. Ollama models are typically pre-quantised to 4-bit, which is why Llama 3.2 1B is only ~1.3 GB instead of the ~2 GB it would be at full precision.

---

## Troubleshooting Table

| Issue | Likely Cause | Solution |
|---|---|---|
| `ollama: command not found` | Ollama not installed or not in PATH | Reinstall from https://ollama.com. On Mac, ensure the Ollama application is running (check the menu bar). On Linux, verify the install script completed successfully. |
| Ollama model pull hangs or is very slow | Limited bandwidth or large download (~1.3 GB for llama3.2:1b) | Check internet connection. If classroom WiFi is slow, have a USB drive with the model blob pre-downloaded. Alternatively, use a mobile hotspot. Run `ollama pull` with a wired connection if possible. |
| `ConnectionRefusedError` when calling `localhost:11434` | Ollama server not running | On Mac: open the Ollama application from Applications folder (look for the llama icon in the menu bar). On Linux: run `ollama serve` in a separate terminal. On Windows: start the Ollama application. |
| `mlx` or `mlx-lm` fails to install | Not on Apple Silicon (Intel Mac, Linux, or Windows) | This is expected. These packages only work on Apple Silicon Macs (M1/M2/M3/M4). Students should follow Track B (Ollama) instead and read through the MLX code for conceptual understanding. |
| MLX model download is slow or fails | Large model download from Hugging Face | Use the smallest available model (e.g., `mlx-community/Llama-3.2-1B-Instruct-4bit`). If download fails, check Hugging Face is accessible. Consider pre-downloading the model: `huggingface-cli download mlx-community/Llama-3.2-1B-Instruct-4bit`. |
| FastAPI server fails to start (`Address already in use`) | Port 8000 is already in use by another process | Kill the existing process: `lsof -ti:8000 | xargs kill -9` (Mac/Linux) or change the port in the server code to 8001. Check if a previous run of the notebook left a server running. |
| `ModuleNotFoundError: No module named 'fastapi'` | Python packages not installed | Run the setup cell at the top of the notebook: `pip install requests fastapi uvicorn psutil aiohttp`. Ensure the correct Python environment/kernel is active in Jupyter. |
| Server starts but `/chat` endpoint returns errors | Ollama not running or model not pulled | Verify Ollama is running (`curl http://localhost:11434/api/tags`). Verify the model is available (`ollama list`). If the model name in the server code does not match the pulled model, update the server code accordingly. |
| `torch` not found during Exercise 4 (MLX vs PyTorch) | PyTorch not installed | Install with `pip install torch`. If installation is slow, students can skip this exercise -- the MLX benchmark alone is sufficient to demonstrate performance. |
| Memory usage warning or system slowdown during inference | Large model consuming most system RAM | Close other applications. If using a model larger than recommended for the hardware, switch to a smaller model. On Apple Silicon with unified memory, the system may become sluggish if the model consumes > 80% of RAM. |

---

## Session 5 Transition

At the end of this session, preview Session 5 (Retrieval-Augmented Generation):

> "Today we learned how to run models on your own hardware and serve them through APIs. But there is a limitation we have not addressed: the model only knows what it was trained on. It cannot answer questions about your company's documents, your private data, or recent information. In Session 5, we tackle this with **Retrieval-Augmented Generation (RAG)** -- a technique that gives LLMs access to your own documents and knowledge base at query time."

Key points to tease:

- **The architecture:** A retrieval system (like a search engine) finds relevant documents, and the LLM uses them to generate informed answers
- **The connection to today's session:** RAG works on any deployment target we covered -- cloud APIs, Ollama on your laptop, or a production GPU server
- **The practical value:** RAG is one of the most widely deployed LLM patterns in enterprise settings, because it lets you use LLMs with proprietary data without retraining the model
- **What we will build:** A working RAG pipeline that retrieves from a document collection and generates grounded answers

> "The skills you learned today -- running models locally, building API servers, managing models -- are the foundation for everything in Session 5. Your local Ollama setup will be the LLM backend for the RAG system we build next time."

---

## Files in This Session

| File | Purpose |
|---|---|
| `Session 4 - Deploying Models On-premise.ipynb` | Hands-on Jupyter notebook (all exercises) |
| `instructor-guides/session-4-instructor-guide.md` | This file |
