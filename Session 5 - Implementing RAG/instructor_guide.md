# Instructor Guide: Session 5 - Implementing RAG (Retrieval-Augmented Generation)

**Module 5: Deep Learning in Production**
**Duration:** 6 hours (120 min theory + 240 min hands-on)

---

## Session Overview

This is the **longest and most hands-on session** in the module. Students will build a complete Retrieval-Augmented Generation (RAG) system from scratch, progressing from embedding fundamentals through vector stores to a fully functional question-answering pipeline. The session uses Ollama with llama3.2:1b as the primary LLM backend, sentence-transformers for embeddings, and ChromaDB/FAISS as vector stores. AWS Bedrock is provided as a fallback for students who encounter issues with the local setup.

**Energy management is critical.** Six hours is a long session. Build in micro-breaks, alternate between lecture and hands-on work, and keep the energy high by having students run code frequently rather than watching extended demos.

---

## Learning Objectives

By the end of this session, students will be able to:

1. **Explain** why RAG addresses the key limitations of standalone LLMs (hallucination, outdated knowledge, no private data access)
2. **Generate** text embeddings using sentence-transformers and interpret cosine similarity scores between documents
3. **Design** a document chunking strategy by selecting appropriate chunk size and overlap parameters for a given use case
4. **Build** vector stores using both FAISS and ChromaDB and perform similarity-based retrieval
5. **Construct** end-to-end RAG pipelines using LangChain components (document loaders, text splitters, embeddings, retrievers, and chains)
6. **Evaluate** RAG system behaviour by testing in-scope and out-of-scope queries and analysing source attribution
7. **Compare** LangChain and LlamaIndex as alternative frameworks for RAG implementation
8. **Configure** Ollama-based local LLM inference and identify when to switch to AWS Bedrock as an alternative backend

---

## Pre-Session Checklist

This session has the most complex setup requirements in the module. **Verify every item below before class begins.** A single missing dependency can stall a student for 30+ minutes.

### Critical (Must Work Before Class)

| Item | How to Verify | If It Fails |
|------|---------------|-------------|
| **Ollama installed and running** | `curl http://localhost:11434` should return `Ollama is running` | Install from https://ollama.ai; on Mac run `brew install ollama` then `ollama serve` |
| **llama3.2:1b model pulled** | `ollama list` should show `llama3.2:1b` (~1.3 GB) | Run `ollama pull llama3.2:1b` -- do this on fast wifi, NOT on classroom wifi |
| **Python packages installed** | `pip install langchain langchain-community langchain-aws chromadb faiss-cpu sentence-transformers pypdf llama-index llama-index-llms-ollama llama-index-embeddings-huggingface boto3 matplotlib numpy` | Run the install cell at the top of the notebook |
| **sentence-transformers model cache** | Run `from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')` -- should load without downloading | First run downloads ~90 MB; pre-run this to cache it |
| **knowledge_base directory** | The notebook creates this directory automatically in Exercise 5 | If students jump ahead, remind them to run cells sequentially |

### Optional (For Bedrock Fallback)

| Item | How to Verify | If It Fails |
|------|---------------|-------------|
| **AWS CLI configured** | `aws sts get-caller-identity` returns account info | Run `aws configure` with valid credentials |
| **Bedrock model access** | Check AWS Console > Bedrock > Model Access for Claude Haiku and Titan Embeddings | Request access in the console; approval is usually instant for Haiku |

### Instructor Machine Preparation

- [ ] Run the entire notebook end-to-end once to confirm it works
- [ ] Verify Ollama starts automatically on boot or start it manually
- [ ] Pre-download the sentence-transformers model so the first embedding cell runs instantly during demo
- [ ] Prepare a backup plan: have the Bedrock code cells ready to uncomment if Ollama fails

---

## Timing Breakdown

### Segment 1: RAG Theory and Architecture (45 min)
**Notebook sections: 1.1 - 1.3**

| Time | Topic | Activity |
|------|-------|----------|
| 0:00 - 0:05 | Session introduction and objectives | Set the stage: this is the capstone technical session |
| 0:05 - 0:15 | Why LLMs need help (hallucination, stale knowledge, no private data) | Interactive: ask students for real-world examples of each limitation |
| 0:15 - 0:25 | What is RAG? The open-book exam analogy | Draw the RAG pipeline diagram on the board |
| 0:25 - 0:35 | RAG vs fine-tuning: when to use which | Reference the comparison table in the notebook |
| 0:35 - 0:45 | Embeddings: meaning fingerprints | Transition to hands-on with a teaser: "Let's see if the computer can tell that 'cat on mat' and 'kitten on rug' mean the same thing" |

### Segment 2: Embeddings and Similarity (45 min)
**Notebook sections: 1.3 - 1.4, Exercise 1 - 2**

| Time | Topic | Activity |
|------|-------|----------|
| 0:45 - 0:55 | Embedding models overview and cosine similarity | Brief lecture with the similarity table |
| 0:55 - 1:10 | Exercise 1: Generate embeddings and visualise similarity heatmap | Students run code; discuss the heatmap results together |
| 1:10 - 1:30 | Exercise 2: Build a FAISS vector store with country facts | Students build, query, and experiment with the search function |

### Segment 3: Vector Stores and ChromaDB (45 min)
**Notebook sections: 1.5, Exercise 3**

| Time | Topic | Activity |
|------|-------|----------|
| 1:30 - 1:40 | Vector database concepts: similarity search, distance metrics, popular options | Lecture with the comparison table |
| 1:40 - 1:55 | Exercise 3: ChromaDB with metadata filtering | Students run code; emphasise the metadata filtering capability |
| 1:55 - 2:15 | Document chunking theory: chunk size, overlap, separators | Use the "organising your notes" analogy; draw chunking diagrams |

### Segment 4: LangChain Fundamentals (45 min)
**Notebook sections: 1.6 - 1.8, Exercise 4 - 5**

| Time | Topic | Activity |
|------|-------|----------|
| 2:15 - 2:25 | LangChain architecture: components, chains, and the RAG flow diagram | Walk through the component table |
| 2:25 - 2:40 | Exercise 4: LangChain basics with Ollama -- first LLM call and prompt template | **Checkpoint:** every student must get a response from Ollama here |
| 2:40 - 3:00 | Exercise 5: Document loading and splitting with different chunk sizes | Students create the knowledge_base directory and experiment |

### Break (15 min)
| Time | Activity |
|------|----------|
| 3:00 - 3:15 | **Break.** Encourage students to stretch and move around. This is the halfway point of a 6-hour session. |

### Segment 5: Building the RAG Pipeline (45 min)
**Notebook section: Exercise 6**

| Time | Topic | Activity |
|------|-------|----------|
| 3:15 - 3:25 | Recap and preview: putting all the pieces together | Quick whiteboard recap of the full pipeline |
| 3:25 - 3:45 | Exercise 6: Build the complete RAG chain (Steps 1-5) | Walk through each step; ensure students understand the chain composition |
| 3:45 - 4:00 | Exercise 6: Test with in-scope and out-of-scope questions (Step 6) | Discuss why the model correctly refuses out-of-scope questions |

### Segment 6: Improving and Evaluating RAG (45 min)
**Notebook section: Exercise 7**

| Time | Topic | Activity |
|------|-------|----------|
| 4:00 - 4:15 | Exercise 7, Experiment 1: Effect of chunk size on retrieval quality | Students run the comparison; discuss which chunk size works best and why |
| 4:15 - 4:30 | Exercise 7, Experiment 2: Source attribution | Emphasise transparency and traceability in production RAG systems |
| 4:30 - 4:45 | Discussion: RAG system design decisions (chunk size, top-k, overlap, prompt engineering) | Facilitate a group discussion on trade-offs |

### Segment 7: LlamaIndex and Alternative Approaches (45 min)
**Notebook section: Exercise 8, Part D**

| Time | Topic | Activity |
|------|-------|----------|
| 4:45 - 4:55 | LlamaIndex overview: philosophy, comparison with LangChain | Use the comparison table from the notebook |
| 4:55 - 5:15 | Exercise 8: Build the same RAG system with LlamaIndex | Students see how much simpler the LlamaIndex API is |
| 5:15 - 5:30 | Side-by-side comparison: discuss when to choose LangChain vs LlamaIndex | LangChain = flexibility; LlamaIndex = simplicity for data-focused RAG |

### Segment 8: Bedrock Fallback Demo and Wrap-Up (30 min)
**Notebook sections: 1.8, Recap**

| Time | Topic | Activity |
|------|-------|----------|
| 5:30 - 5:40 | AWS Bedrock demo: swap Ollama for ChatBedrock in the RAG chain | Show the 2-line code change; discuss cloud vs local trade-offs |
| 5:40 - 5:50 | Session recap: RAG architecture summary, RAG vs fine-tuning table | Walk through the recap section of the notebook |
| 5:50 - 6:00 | Transition to Session 6 (Assessment) and Q&A | Set expectations for the assessment |

---

## Key Talking Points

### Why LLMs Need RAG

- **The hallucination problem:** LLMs generate plausible-sounding text based on patterns, not facts. They do not "know" anything -- they predict the next token. When they lack information, they fabricate it confidently.
- **Analogy:** "Imagine hiring a brilliant new employee who graduated top of their class but has never read a single document about your company. They will give you eloquent, confident answers -- but they will be making things up. RAG is like giving that employee access to the company wiki before they answer."
- **The three limitations** are the core motivation: hallucination, outdated knowledge, no private data. Every RAG architecture decision traces back to solving one of these.

### What is RAG?

- **The open-book exam analogy:** Fine-tuning is like studying for a closed-book exam -- the model memorises information during training. RAG is like an open-book exam -- the model looks up information at query time.
- **Key insight:** RAG separates knowledge storage from reasoning. The LLM provides the reasoning capability; the vector store provides the knowledge. This separation is powerful because you can update knowledge without retraining.
- **When NOT to use RAG:** If you need the model to change its style, tone, or behaviour (not its knowledge), fine-tuning is more appropriate. RAG and fine-tuning are complementary, not competing techniques.

### Embeddings

- **Analogy -- "Meaning fingerprints":** Just as a fingerprint uniquely identifies a person, an embedding uniquely identifies the meaning of a piece of text. Two sentences with the same meaning will have nearly identical "fingerprints," even if the words are completely different.
- **Why this matters:** Traditional keyword search fails when users use different words than the documents. Embeddings enable semantic search -- finding documents by meaning, not just word matching.
- **The numbers:** Each sentence becomes a vector of 384 numbers (for MiniLM). These numbers are not human-interpretable, but the distances between vectors are meaningful.

### Document Chunking

- **Analogy -- "Organising your notes":** Imagine you have a 100-page textbook and someone asks you a question. You do not hand them the entire book. You find the most relevant page or paragraph. Chunking is the process of deciding how to break the book into searchable pieces.
- **The Goldilocks problem:** Chunks too small lose context ("the answer spans two chunks and neither is complete"). Chunks too large dilute relevance ("this chunk is 5 pages and only one sentence is relevant"). Finding the right size is experimental.
- **Overlap as insurance:** Chunk overlap ensures that information at boundaries is not lost. If an important sentence falls at the border between two chunks, the overlap captures it in both.

### Vector Similarity

- **Analogy -- "Finding the most relevant pages":** When you search, the vector database computes how "close" your question's embedding is to each stored chunk's embedding. The closest chunks are the most semantically relevant.
- **Cosine similarity:** Think of it as the angle between two arrows. If they point in the same direction (similarity = 1.0), the texts mean the same thing. If they are perpendicular (similarity = 0.0), the texts are unrelated.
- **Top-k retrieval:** We retrieve the k most similar chunks. Choosing k is a trade-off: too few and you miss relevant context; too many and you include noise that confuses the LLM.

### LangChain vs LlamaIndex

- **LangChain = toolkit:** It provides individual components (loaders, splitters, embeddings, retrievers, chains) that you assemble yourself. More flexible, but more code to write.
- **LlamaIndex = opinionated framework:** It provides a streamlined pipeline with sensible defaults. Fewer lines of code for standard RAG, but less flexibility for custom workflows.
- **Recommendation:** Start with LlamaIndex for quick prototypes; move to LangChain when you need fine-grained control over the pipeline.

---

## Live Demo Suggestions

### Demo 1: Embedding Similarity Heatmap (Exercise 1)
- **What to show:** Run the heatmap cell and highlight how "cat sat on mat" and "kitten resting on rug" have high similarity despite no shared words, while "Python programming language" is distant from everything else.
- **Interactive twist:** Ask students to suggest two sentences and predict their similarity before running the code.
- **Fallback:** If the model download is slow, have pre-computed embeddings saved and load them from a pickle file.

### Demo 2: FAISS Search with Surprising Queries (Exercise 2)
- **What to show:** Search for "kangaroos" and watch it find Australia. Then search for something abstract like "ancient civilisations" and see it find Egypt.
- **Interactive twist:** Let students type queries and predict which country will rank first.
- **Fallback:** If FAISS fails to install, use ChromaDB for both exercises (it has a simpler installation).

### Demo 3: RAG Chain In Action (Exercise 6)
- **What to show:** Ask an in-scope question ("What is Acme's pricing?") and get a grounded answer. Then ask an out-of-scope question ("What is the weather?") and show the model correctly refusing to answer.
- **Interactive twist:** Have a student ask a tricky borderline question to test the system's limits.
- **Fallback:** If Ollama is not responding, switch to Bedrock by uncommenting the ChatBedrock lines. Show this swap as a teaching moment about backend flexibility.

### Demo 4: Chunk Size Comparison (Exercise 7)
- **What to show:** Run the same query with chunk sizes 200, 500, and 1000. Show how smaller chunks are more precise but may miss context, while larger chunks capture more but dilute relevance.
- **Fallback:** This demo does not depend on the LLM -- it only uses embeddings and retrieval. It should work even if Ollama is down.

---

## Common Student Questions and Answers

### "Why not just fine-tune the model instead of using RAG?"

Fine-tuning and RAG solve different problems. Fine-tuning changes the model's behaviour, style, or specialised knowledge by retraining its weights -- think of it as long-term learning. RAG provides the model with up-to-date, specific information at query time -- think of it as looking up a reference. Fine-tuning is expensive (requires GPU training), slow to update (must retrain for new information), and offers no source attribution. RAG is cheap, instantly updatable (just add or remove documents), and transparent (you can see which documents informed the answer). In practice, many production systems use both: fine-tune for domain style and behaviour, then RAG for factual knowledge retrieval.

### "How do we handle documents that are too large?"

This is exactly why we have document chunking. Large documents are split into smaller pieces (typically 200-1500 characters) with overlap between chunks to preserve context at boundaries. The key parameters are chunk size (how big each piece is) and chunk overlap (how much adjacent pieces share). For very large document collections, you can also use hierarchical chunking: first split into sections by headings, then split sections into paragraphs. The right chunk size depends on your use case -- shorter chunks for precise factual retrieval, longer chunks for questions that require broader context.

### "What if the retrieved context is irrelevant?"

This is a real challenge in RAG systems. Several strategies help: (1) Increase the number of retrieved chunks (higher top-k) so the relevant one is more likely to be included. (2) Use a re-ranking step after initial retrieval to score chunks more carefully. (3) Improve your chunking strategy so that relevant information is self-contained within chunks. (4) Add metadata filtering to narrow the search space (e.g., only search documents from a specific category). (5) Use a better embedding model -- larger models like all-mpnet-base-v2 capture meaning more accurately. (6) Include an instruction in your prompt telling the LLM to say "I don't know" if the context is insufficient, which the notebook demonstrates.

### "Can we use this with any LLM?"

Yes, that is one of the key strengths of RAG. The retrieval pipeline (embedding, vector store, similarity search) is completely independent of the LLM. You can swap the LLM without changing anything else. In the notebook, we use Ollama with llama3.2:1b locally and show how to switch to AWS Bedrock with Claude in just two lines of code. You could also use OpenAI GPT-4, Google Gemini, or any other LLM. The only requirement is that the LLM can accept a text prompt with the retrieved context included.

### "How does this compare to traditional search (like Elasticsearch)?"

Traditional search is keyword-based: it matches exact words and uses techniques like TF-IDF to rank results. RAG uses semantic search: it matches meaning via embeddings. This means RAG can find documents about "canine nutrition" when you search for "dog food," which keyword search would miss. However, traditional search is faster for very large corpora, more predictable, and easier to debug. Many production systems use a hybrid approach: keyword search to narrow candidates, then semantic re-ranking for the final results. ChromaDB and other vector databases are adding hybrid search capabilities that combine both approaches.

### "How many documents can a RAG system handle?"

It depends on the vector database. In-memory solutions like FAISS can handle millions of vectors on a single machine. ChromaDB running locally is practical for tens of thousands to low millions of documents. For larger scales, managed cloud vector databases like Pinecone, Weaviate, or Amazon OpenSearch Serverless can handle billions of vectors with horizontal scaling. The embedding step is the bottleneck for ingestion: embedding a million documents with sentence-transformers takes hours on a CPU but minutes on a GPU.

### "Is it safe to put sensitive company data in a vector store?"

This is an important production consideration. The vector store contains embeddings (numerical representations) and optionally the original text. If someone gains access to the vector store, they can read the original documents. For sensitive data: (1) Use a self-hosted vector database, not a cloud service. (2) Encrypt the vector store at rest and in transit. (3) Implement access controls so only authorised users can query. (4) Consider whether the embeddings themselves leak information -- research shows that embeddings can sometimes be reverse-engineered to recover approximate original text. In our notebook, ChromaDB runs entirely in-process with no network exposure, which is the safest option for learning.

---

## Troubleshooting

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Ollama not running** | `ConnectionRefusedError` or `Connection error` when invoking the LLM | Start Ollama: on Mac run `ollama serve` in a terminal (or launch the Ollama app). Verify with `curl http://localhost:11434`. If the port is blocked, check for firewall rules. |
| **Model not found (not pulled)** | Error message: `model 'llama3.2:1b' not found` | Run `ollama pull llama3.2:1b` in a terminal. This downloads ~1.3 GB. Do NOT attempt this on slow classroom wifi -- have students download it before class or provide a USB drive with the model files. |
| **ChromaDB installation fails** | `pip install chromadb` errors, often related to `chroma-hnswlib` or C++ build tools | On Mac: `xcode-select --install` to get build tools. On Windows: install Visual C++ Build Tools. Alternative: use `pip install chromadb --no-binary :all:` or skip ChromaDB exercises and use FAISS only. |
| **sentence-transformers download timeout** | Hangs or times out when loading `SentenceTransformer('all-MiniLM-L6-v2')` for the first time | The first load downloads ~90 MB from Hugging Face. Pre-download before class: run the import once on good wifi. If blocked by a corporate firewall, download the model manually and load from a local path. |
| **Out of memory during embedding** | `MemoryError` or system becomes unresponsive when embedding large document collections | Reduce batch size in `model.encode()` by adding `batch_size=32` (default is 64). For very large collections, embed in batches and add to the vector store incrementally. The exercises use small document sets, so this should only occur if students add their own large documents. |
| **knowledge_base directory not found** | `FileNotFoundError: knowledge_base` when loading documents | The notebook creates this directory in Exercise 5 (cell that writes `company_faq.txt` and `product_manual.txt`). Ensure students run cells sequentially. If the directory was cleaned up by the final cell, re-run the creation cell. |
| **Bedrock access denied** | `AccessDeniedException` when using `ChatBedrock` or `BedrockEmbeddings` | Check three things: (1) AWS credentials are configured (`aws configure`), (2) Bedrock model access is enabled in the AWS Console (Bedrock > Model Access > Request Access for Claude Haiku and Titan Embeddings), (3) the IAM user/role has the `bedrock:InvokeModel` permission. Model access requests are usually approved instantly. |
| **Slow inference on CPU-only machines** | LLM responses take 30-60+ seconds with Ollama | This is expected with llama3.2:1b on CPU. Reassure students that this is normal for local inference. To speed up: (1) Ensure no other heavy processes are running. (2) Close unnecessary browser tabs and applications. (3) If still too slow, switch to Bedrock which runs on AWS infrastructure. The 1b parameter model was chosen specifically because it is the smallest practical option. |
| **FAISS import error** | `ModuleNotFoundError: No module named 'faiss'` | Ensure `faiss-cpu` (not `faiss-gpu`) was installed: `pip install faiss-cpu`. On some systems, you may need to restart the Jupyter kernel after installation. |
| **LangChain import errors or deprecation warnings** | `ImportError` or `LangChainDeprecationWarning` messages | LangChain's API changes frequently. Ensure versions match: `pip install langchain>=0.3 langchain-community>=0.3 langchain-aws>=0.2`. Deprecation warnings can be ignored for the exercises but note them for students so they know the ecosystem evolves rapidly. |
| **LlamaIndex timeout on Ollama** | `TimeoutError` when LlamaIndex tries to query Ollama | Increase the timeout: `LlamaOllama(model="llama3.2:1b", request_timeout=300)`. LlamaIndex has a shorter default timeout than LangChain, so CPU inference may exceed it. |

---

## Bedrock Fallback: Step-by-Step Swap

If Ollama fails for a student and cannot be quickly fixed, they can switch to AWS Bedrock for the LLM component. The retrieval pipeline (embeddings, vector store, chunking) remains unchanged -- only the LLM backend changes.

### What to Change

**Exercise 4 (LangChain Basics) -- swap the LLM:**
```python
# BEFORE (Ollama):
from langchain_community.llms import Ollama
llm = Ollama(model="llama3.2:1b")

# AFTER (Bedrock):
from langchain_aws import ChatBedrock
llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
```

**Exercise 8 (LlamaIndex) -- swap the LLM setting:**
```python
# BEFORE (Ollama):
from llama_index.llms.ollama import Ollama as LlamaOllama
Settings.llm = LlamaOllama(model="llama3.2:1b", request_timeout=120)

# AFTER (Bedrock):
# Note: Requires llama-index-llms-bedrock package
from llama_index.llms.bedrock import Bedrock
Settings.llm = Bedrock(model="anthropic.claude-3-haiku-20240307-v1:0")
```

**Everything else stays the same.** The embeddings still use sentence-transformers locally. The vector store still uses FAISS/ChromaDB locally. Only the final generation step changes from a local LLM to a cloud LLM.

### Key Talking Point for the Swap

> "Notice that we only changed two lines of code to switch from a local open-source model to a cloud-hosted commercial model. This is one of the strengths of frameworks like LangChain and LlamaIndex -- they abstract the LLM backend so your RAG pipeline is portable. In production, you might start with a local model for development and switch to a cloud model for deployment."

---

## Transition to Session 6

> "Today we built something practical and powerful -- a complete RAG system that can answer questions grounded in real documents. We started with the fundamentals: how embeddings capture meaning, how vector databases enable semantic search, and how chunking strategies affect retrieval quality. Then we assembled a full pipeline using LangChain, tested it with LlamaIndex, and saw how easily we can swap LLM backends between local and cloud.
>
> This session ties together many of the production concepts from earlier in the module. In Session 1, we learned about model serving. In Session 2, we deployed models to SageMaker. In Sessions 3 and 4, we worked with MLOps and monitoring. Today's RAG system is the kind of application you would build on top of all that infrastructure.
>
> In Session 6, we bring everything together with the module assessment. The assessment has two parts: short-answer questions covering the concepts from Sessions 1 through 5, and a practical component where you will deploy a model to AWS SageMaker. Review the RAG concepts from today -- understand the pipeline architecture, know when to use RAG versus fine-tuning, and be able to explain the role of each component (embeddings, vector store, retriever, LLM). If you can explain today's system to someone who has never seen it, you are well prepared."
