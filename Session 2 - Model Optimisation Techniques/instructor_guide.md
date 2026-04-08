# Instructor Guide: Session 2 - Model Optimisation Techniques

**Module 5: Deep Learning in Production**
**Duration:** 1 hour (20 min theory + 40 min hands-on)
**Prerequisites:** Session 1 completed (students understand production pipelines, model serialization, and FastAPI serving); basic PyTorch knowledge from Modules 1-4

---

## Learning Objectives

By the end of this session, students will be able to:

1. **Evaluate** why model optimisation is necessary for production deployment across different hardware targets
2. **Apply** dynamic and static quantisation to a PyTorch model and measure the impact on size and speed
3. **Implement** unstructured pruning on convolutional layers and analyze the resulting sparsity patterns
4. **Compare** the trade-offs between quantisation, pruning, and knowledge distillation using empirical benchmarks
5. **Determine** which optimisation technique to apply given a specific deployment scenario (cloud, mobile, edge)

---

## Pre-Session Checklist

- [ ] All packages installed: `torch`, `torchvision`, `matplotlib`, `Pillow`, `tabulate`, `numpy`
- [ ] Run the entire Session 2 notebook end-to-end to confirm all cells execute without errors
- [ ] Verify that ResNet18 pre-trained weights are cached locally (should be from Session 1; if not, run `models.resnet18(weights=models.ResNet18_Weights.DEFAULT)` once)
- [ ] Note the baseline benchmark numbers from your instructor machine -- these will differ from student machines, so be prepared to discuss why (CPU type, background processes, OS)
- [ ] **Matplotlib check:** Verify that plots render correctly. If projecting to a screen, increase the default font size by adding `plt.rcParams.update({'font.size': 14})` at the top of the notebook so charts are legible from the back of the room
- [ ] **Projector-friendly colors:** The notebook uses `steelblue`, `darkorange`, `#2196F3`, `#4CAF50`, `#FF9800`, `#9C27B0` -- these are high-contrast and projector-safe. If your projector washes out colors, mention the numeric labels on each bar so students can still read the charts
- [ ] **Fallback:** Prepare pre-computed benchmark results (screenshot or markdown table) in case inference timing varies wildly or cells take too long on underpowered student machines
- [ ] **Fallback:** Have the final side-by-side comparison table (Exercise 5) ready as a slide or printed handout in case students do not reach it in time
- [ ] Confirm the quantisation backend matches your platform: `x86` for Intel/AMD, `qnnpack` for ARM (Apple Silicon Macs may need `qnnpack`)

---

## Timing Breakdown

### Theory (20 minutes)

| Time | Topic | Notes |
|------|-------|-------|
| 0:00 - 0:02 | Welcome, recap Session 1 | Bridge from "we built a serving pipeline" to "now we optimise the model inside it" |
| 0:02 - 0:06 | Why optimise models? (Section 1.1) | Production constraints table; the accuracy-vs-speed trade-off; deployment scenarios |
| 0:06 - 0:11 | Quantisation (Section 1.2) | FP32 to INT8; dynamic vs static vs QAT; pros and cons |
| 0:11 - 0:15 | Pruning (Section 1.3) | Unstructured vs structured; the "close to zero" observation; company reorganisation analogy |
| 0:15 - 0:18 | Knowledge distillation (Section 1.4) | Teacher-student framework; soft labels; DistilBERT example |
| 0:18 - 0:20 | Comparison of all three techniques (Section 1.5) and transition to hands-on | Summary table; "these are complementary, not competing" |

### Hands-On (40 minutes)

| Time | Activity | Notebook Section |
|------|----------|------------------|
| 0:20 - 0:25 | Helper functions + Exercise 1: Baseline model metrics | Cells 11-18 |
| 0:25 - 0:32 | Exercise 2: Dynamic quantisation | Cells 20-25 |
| 0:32 - 0:40 | Exercise 3: Static quantisation (walkthrough) | Cells 27-36 |
| 0:40 - 0:48 | Exercise 4: Pruning and weight distribution visualisation | Cells 38-45 |
| 0:48 - 0:55 | Exercise 5: Side-by-side comparison (table + charts + consistency check) | Cells 47-52 |
| 0:55 - 1:00 | Recap, decision flowchart, and transition to Session 3 | Cell 54 |

---

## Key Talking Points

### Section 1.1: Why Optimise Models?

- **Core concept:** Your model works great on your GPU workstation, but production environments have hard constraints -- memory limits, latency budgets, cost ceilings, and energy/battery limitations. Optimisation removes redundancy to meet these constraints.
- **Real-world analogy:** Shipping a product is like packing for a flight. Your full wardrobe (the original model) does not fit in carry-on luggage (a mobile device). You keep the essentials and leave the rest behind. The goal is to arrive looking just as good with a much smaller bag.
- **Key table to walk through:** The deployment scenarios table (cloud, mobile, edge, browser). Ask students: "Which of these is the most constrained? Why?" (Answer: edge devices like drones or IoT sensors -- tiny memory, no GPU, battery-powered.)
- **Discussion prompt:** "Has anyone tried running a large model on a laptop CPU and found it too slow? What did you do about it?"

### Section 1.2: Quantisation

- **Core concept:** Quantisation reduces the numerical precision of weights and activations. Instead of 32-bit floats, we use 8-bit integers. This cuts model size by roughly 4x and often speeds up inference, especially on CPUs.
- **Real-world analogy:** Think of reducing image quality from RAW (50 MB) to JPEG (5 MB). You lose some fine detail, but for most purposes the image looks identical. Similarly, going from FP32 to INT8 loses some numerical precision, but the model's predictions barely change.
- **Key insight to emphasize:** The errors from reduced precision mostly cancel out across millions of weights. This is not intuitive -- walk through the example in the notebook where a weight of 0.7834521 becomes approximately 0.78 in INT8. For a single weight the error is tiny; across the whole model, the aggregate effect is minimal.
- **Three types to distinguish:** Dynamic (simplest, one line of code), Static (better gains, needs calibration data), QAT (best accuracy, requires retraining). For today, we implement dynamic and static. QAT is mentioned for awareness.
- **Discussion prompt:** "Why might INT8 quantisation be faster than FP32 on a CPU?" (Answer: CPUs have specialized integer arithmetic units; INT8 operations use less memory bandwidth; more values fit in cache.)

### Section 1.3: Pruning

- **Core concept:** Many weights in a trained neural network are very close to zero and contribute almost nothing to the output. Pruning removes these unnecessary weights.
- **Real-world analogy:** Pruning a neural network is like pruning a tree. You cut away the dead branches (near-zero weights) so the tree (model) can focus its energy on the branches that matter. The tree stays healthy and often grows better.
- **Unstructured vs structured distinction:** Unstructured pruning sets individual weights to zero (high sparsity but irregular memory patterns). Structured pruning removes entire neurons, filters, or channels (lower sparsity but genuine hardware speedup). Use the "company reorganisation" analogy from the notebook: unstructured = removing tasks from everyone's to-do list; structured = eliminating entire departments.
- **Important caveat to mention:** Unstructured pruning alone does not reduce stored model size because zeros still take up space in the tensor. Students will see this in Exercise 4. Real gains come from sparse storage formats, structured pruning, or combining pruning with quantisation.
- **Discussion prompt:** "If 50% of weights are near zero, what does that tell us about the model's capacity relative to the task?"

### Section 1.4: Knowledge Distillation

- **Core concept:** Train a smaller "student" model to mimic a larger "teacher" model. The key insight is that the teacher's soft probability outputs (e.g., "85% cat, 10% dog, 4% car, 1% truck") contain richer information than hard labels ("cat"), because they encode similarity relationships between classes.
- **Real-world analogy:** An experienced doctor (teacher) does not just tell a medical student (student) the diagnosis. They explain: "It is probably condition A, but it shares some symptoms with condition B, and you can rule out condition C because..." This richer teaching signal helps the student learn faster and more effectively.
- **Concrete example:** DistilBERT -- 40% smaller than BERT, 60% faster, retains 97% of BERT's accuracy. It is one of the most widely deployed NLP models in production precisely because of this favorable trade-off.
- **Note for students:** We do not implement distillation hands-on because it requires a full training loop, but many models students will use in production (DistilBERT, TinyLlama, etc.) were created this way.

### Section 1.5: Comparison and When to Combine

- **Core concept:** These techniques are complementary, not competing. A common production pipeline is: Train a large model, distil it into a smaller student, prune the student, then quantise the result. Each step compounds the compression.
- **Key decision framework:** Start with quantisation (low effort, high reward). If that is not enough, add pruning. If you need a fundamentally smaller architecture, use knowledge distillation.
- **Discussion prompt:** "If you were deploying an image classifier to a Raspberry Pi with 1 GB of RAM, which techniques would you use and in what order?" (Guide toward: all three, starting with distillation to get a tiny architecture, then pruning, then quantisation.)

### Exercise-Specific Talking Points

- **Exercise 1 (Baseline):** Point out that "random input" predictions are meaningless content-wise, but they serve as a consistent reference for comparing optimised models. The important thing is whether the optimised model produces the same top-1 class, not whether that class is correct.
- **Exercise 2 (Dynamic Quantisation):** Highlight that for ResNet-18, dynamic quantisation mainly affects the final `fc` layer. Since most computation is in convolutional layers, gains are modest. This is a feature of the technique, not a failure -- dynamic quantisation shines on Transformer and LSTM models with large linear layers.
- **Exercise 3 (Static Quantisation):** This is the most complex exercise. Walk students through each of the four steps (prepare, configure, calibrate, convert) carefully. Emphasize that the calibration dataset should be representative of real inference data. We use random tensors for simplicity, but in production you would use a held-out validation set.
- **Exercise 4 (Pruning):** When the weight distribution histogram appears, draw attention to the massive spike at zero in the pruned model. This is a powerful visual. Ask: "Where did those zero weights come from?" (Answer: they were the smallest-magnitude weights, removed by L1 pruning.)
- **Exercise 5 (Comparison):** This is the payoff. Let the comparison table and bar charts tell the story. Ask students to identify which technique gave the best trade-off for size vs. speed.

### Matplotlib Visualisation Notes

- **Projector visibility:** The notebook uses high-contrast colors (`steelblue`, `darkorange`, `#2196F3`, `#4CAF50`, `#FF9800`, `#9C27B0`) that work well on most projectors. If your projector washes out colors, point to the numeric labels above each bar.
- **Font size:** If the back row cannot read the axis labels, add `plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16})` before the plotting cells.
- **Dark mode:** If students are using dark-mode Jupyter themes, the default `matplotlib` white background will still render correctly. No changes needed.

---

## Live Demo Suggestions

### Demo 1: Dynamic Quantisation in One Line (during Exercise 2, ~3 minutes)

1. **Setup:** Have the notebook at Exercise 2, Step 1.
2. **Walk through:** Before running the cell, highlight that `quantize_dynamic()` is a single function call. Type or point to each argument: the model copy, the layer type specification (`{nn.Linear}`), and the target dtype (`torch.qint8`).
3. **Run the cell** and show the output comparing the original `fc` layer with the quantised version. Point out the `DynamicQuantizedLinear` layer type in the output.
4. **Run the measurement cell** and narrate: "Notice the size reduction is modest -- only a few percent. That is because ResNet18's size is dominated by convolutional layers, and dynamic quantisation only touches Linear layers."
5. **Key takeaway line:** "Dynamic quantisation is your one-line quick win. It takes 10 seconds to apply and zero data to calibrate. Always try this first."

### Demo 2: Static Quantisation Walkthrough (during Exercise 3, ~5 minutes)

1. **Walk through all four steps** at a measured pace. This is the most conceptually dense exercise.
2. **At Step 1 (QuantStub/DeQuantStub wrapper):** Explain: "We are telling PyTorch where the quantised domain begins and ends. Input comes in as FP32, gets quantised, flows through the model in INT8, and gets dequantised back to FP32 at the output."
3. **At Step 3 (calibration):** Emphasize: "In production, these 20 random tensors would be 20 real images from your validation set. The observers are recording the typical range of activation values so the quantiser knows how to map floats to integers."
4. **At Step 5 (results):** Compare with dynamic quantisation numbers. Static should show significantly better size reduction (roughly 4x). Ask: "Why is static quantisation so much more effective here?" (Answer: it quantises the convolutional layers too.)

### Demo 3: Pruning Weight Distribution Visualisation (during Exercise 4, ~3 minutes)

1. **Run the histogram cell** and let the visualization render.
2. **Point to the left chart (original):** "A nice bell curve centered near zero. This is typical for well-trained neural networks."
3. **Point to the right chart (pruned):** "See that enormous spike at zero? Those are the 50% of weights we just removed. They were the smallest-magnitude weights -- the ones contributing least to the model's predictions."
4. **Ask:** "But look at the file size comparison. Did pruning actually reduce the file size?" (Answer: no, because zeros still take up space in the dense tensor format. This motivates sparse formats and combining pruning with quantisation.)

**Fallback plan for all demos:** If benchmark numbers vary wildly on student machines (common with CPU timing), reassure students: "The absolute numbers depend on your hardware. Focus on the relative differences -- the ratios between baseline and optimised models should be consistent across machines." If a student's machine is too slow to run all exercises in time, share the pre-computed comparison table so they can still participate in the discussion.

---

## Common Student Questions and Answers

### "If quantisation is so easy and effective, why does not everyone just quantise every model?"

Most production teams do quantise. The reason it is not universal is that not all model architectures quantise equally well. Models with very few Linear layers (like small CNNs) see limited benefit from dynamic quantisation. Some operations (like certain attention mechanisms) may lose noticeable accuracy when quantised. Additionally, GPU inference does not always benefit from INT8 quantisation as much as CPU inference does, because GPUs are already optimized for FP32/FP16 math. The key is to always benchmark on your specific model and hardware.

### "Why did the pruned model not get any smaller on disk?"

Unstructured pruning sets weights to zero, but they are still stored as zeros in the tensor -- the tensor shape does not change. To get real size benefits from pruning, you need: (1) sparse tensor formats like CSR/CSC that compress zero values, (2) structured pruning that removes entire filters and genuinely shrinks the tensor dimensions, or (3) a combination of pruning and quantisation where the zeros compress very efficiently. Think of it this way: a book with many blank pages is not lighter than a book with all pages filled.

### "What accuracy loss should I expect from these techniques?"

For quantisation (FP32 to INT8), typical accuracy loss on image classification tasks is less than 1%. For pruning at 50% sparsity without fine-tuning, expect 1-3% accuracy loss. For knowledge distillation, it depends on the size gap between teacher and student, but well-designed student models typically retain 95-99% of the teacher's accuracy. The key insight is that you should always measure on your specific task -- never assume. Run your test set through both the original and optimised models and compare.

### "Can I combine all three techniques on the same model?"

Yes, and this is exactly what production teams do for maximum compression. A typical pipeline is: (1) train a large teacher model, (2) distil into a smaller student, (3) prune the student to remove redundant weights, (4) fine-tune the pruned model to recover accuracy, (5) quantise the result. Each step compounds the size reduction. For example, distillation might give 3x compression, pruning another 2x, and quantisation another 4x, for a total of roughly 24x compression.

### "How do I choose the right pruning percentage?"

Start conservative (20-30%) and increase gradually while monitoring accuracy on your validation set. Plot a pruning-vs-accuracy curve: prune at 10%, 20%, 30%, ... 90% and measure accuracy at each level. Most models show a "knee" in the curve where accuracy drops sharply -- stay below that knee. For many models, 50-70% pruning with fine-tuning maintains nearly baseline accuracy. Without fine-tuning (as we did today), keep pruning below 50% to be safe.

---

## Troubleshooting Table

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `RuntimeError: Didn't find engine for operation quantized::linear_dynamic` | Quantisation backend mismatch with the CPU architecture | Check your platform: use `torch.backends.quantized.engine = 'x86'` on Intel/AMD, or `'qnnpack'` on ARM (including Apple Silicon). Run `print(torch.backends.quantized.supported_engines)` to see available options. |
| Static quantisation cell produces `NotImplementedError` or `RuntimeError` during `torch.quantization.convert()` | Some layer types do not support quantisation, or the model was not properly prepared | Ensure `torch.quantization.prepare()` was called before `convert()`. Check that `model.eval()` was called. Some custom layers may need manual quantisation configuration. |
| Inference timing results are inconsistent (high standard deviation) | Background processes consuming CPU, or thermal throttling on a laptop | Close other applications, especially browsers and IDEs. Increase the number of warm-up runs from 5 to 20. Run on AC power (not battery) to avoid throttling. If still inconsistent, increase `num_runs` to 100 for more stable averages. |
| `matplotlib` plots do not render in the notebook | Missing `%matplotlib inline` magic, or `matplotlib` backend issue | Add `%matplotlib inline` at the top of the notebook. If using VS Code, ensure the Jupyter extension is up to date. As a fallback, save plots to PNG files with `plt.savefig('plot.png')` and display them. |
| `ModuleNotFoundError: No module named 'tabulate'` | `tabulate` package not installed | Run `pip install tabulate`. This is a lightweight formatting library with no dependencies. |
| Pruning cell raises `AttributeError: weight_orig` | Pruning was already applied to the same layer (double-pruning) | Restart the kernel and re-run from the beginning. Pruning modifies the module in place, so running the pruning cell twice on the same model object causes this error. |
| `ImportError: cannot import name 'QuantStub'` | Older version of PyTorch that does not support the quantisation API | Upgrade PyTorch: `pip install torch --upgrade`. The quantisation API requires PyTorch 1.8+. Check version with `torch.__version__`. |
| Apple Silicon Mac shows no speed improvement from quantisation | PyTorch's INT8 quantisation is not fully optimised for ARM/Apple Silicon | Use the `qnnpack` backend instead of `x86`. Note that on Apple Silicon, the baseline FP32 performance is already quite fast due to the efficient CPU architecture, so relative gains may be smaller. Emphasize that the size reduction is still valuable for deployment scenarios. |

---

## Transition to Next Session

> "Today we learned how to make models smaller and faster without retraining them. Quantisation gave us a nearly 4x size reduction in one line of code. Pruning showed us that half the weights in our model were barely contributing. And knowledge distillation -- while we did not implement it today -- is how models like DistilBERT achieve production-ready performance at a fraction of the size.
>
> But we have been working entirely on our local machines. The model runs on your laptop, serves requests on localhost, and is accessible to exactly one person -- you. That is not production.
>
> In Session 3, we take the next step: **deploying models to the cloud**. We will take a model like the ones we optimised today and deploy it to AWS SageMaker, where it gets a real HTTPS endpoint that anyone on the internet can call. We will cover containerisation, endpoint configuration, auto-scaling, and monitoring. The gap between your laptop and a globally accessible ML service is smaller than you think -- and Session 3 will close it."
