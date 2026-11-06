# Viva Guide — LSD-XAI

## 1. Before the viva (10 minutes)

1. Plug in the laptop charger. The model runs on the GPU.
2. Open a terminal in the project folder and run `./start.sh`.
3. Open http://localhost:5173 in the browser.
4. Do one test prediction, so the models are loaded into memory and the first
   demo prediction is fast.
5. Keep these open in tabs: the web app, `outputs/ablation.md`, and one XAI
   figure such as `outputs/xai/hybrid_attention_c3/sample_04.png`.
6. Backup demo if the web app fails: `.venv/bin/python app.py --run hybrid_attention_c3`
   (Gradio).

## 2. Demo script (5–7 minutes)

| Step | Do | Say |
|---|---|---|
| 1 | Open **About** | "LSD causes skin nodules in cattle. Our pipeline has six stages: acquisition, preprocessing, dataset preparation, hybrid model, classification, and explainable AI." |
| 2 | Point to the architecture card | "EfficientNet-B0 learns local texture such as nodules. DeiT-Tiny, a Vision Transformer, learns global context through self-attention. We fuse both with concatenation or cross-attention." |
| 3 | Point to *Preventing train/test leakage* | "Public LSD datasets contain augmented copies of the same photo. We grouped near-duplicates with perceptual hashing and kept each group in one split, so our test score is honest." |
| 4 | Open **Diagnose**, select **3-class**, click an LSD sample, click **Analyze** | "The model predicts LSD with high confidence. Here are the probabilities for each class." |
| 5 | Point to Grad-CAM and LIME | "Grad-CAM shows where the CNN branch looked. LIME shows which image regions support the prediction. Both focus on the nodules." |
| 6 | Click **Explain deeper (SHAP)** | "SHAP gives pixel-level contributions based on Shapley values." |
| 7 | Click a healthy sample, switch to **Binary**, Analyze | "On a healthy animal the model predicts healthy." |
| 8 | Open **Results** | "This is the ablation study: CNN only, ViT only, and two hybrid fusions, plus the 3-class model. The best model reaches 93.9% accuracy and 0.990 ROC-AUC." |
| 9 | Select a run in the gallery | "Confusion matrix, ROC curve and training curves for each run." |

### Which sample to click

The sample strip shows 9 test images in this order. Results were checked on
both models:

| Sample | True class | 3-class model | Binary model |
|---|---|---|---|
| 1–3 | Healthy | Healthy (≥ 99.9%) | Healthy (92–97%) |
| 4 | LSD | **Healthy (99.1%) — a miss** | LSD (96.3%) |
| 5–6 | LSD | LSD (100%) | LSD (93–97%) |
| 7–9 | Foot-and-mouth | Other (100%) | LSD (61–81%) — expected, binary has no "other" class |

Use samples **5 or 6** for the LSD demo. Sample 4 is a good honest example if
an examiner asks about errors: the 3-class model misses about 1 in 10 LSD
images (LSD recall 0.90). Samples 7–9 on the binary model show why the 3-class
model is needed: a binary model forces every skin disease into "LSD".

## 3. Key numbers

| Item | Value |
|---|---|
| Images | 4,268 (1,991 healthy, 1,531 LSD, 746 foot-and-mouth) |
| Split | 70 / 15 / 15, grouped by duplicate, seed 42 |
| Best model | Hybrid cross-attention, 3-class: 93.9% accuracy, F1 0.945, AUC 0.990 |
| Best binary | CNN only 94.1%; best hybrid (concat, tuned) 93.4% |
| Model size | Hybrid 10.3 M parameters |
| Training time | About 27 s per epoch on an RTX 4050 (6 GB), at most 15 epochs |

## 4. Likely questions and answers

**Q: Why a hybrid of CNN and Transformer?**
A: CNNs are strong at local patterns such as nodule texture. Transformers relate
all image patches to each other, so they capture global context such as how
many nodules there are and where. The hybrid combines both views.

**Q: Your CNN-only model beats the hybrid in the binary task. Why use the hybrid?**
A: With about 2,500 training images, the 10 M-parameter hybrid overfits faster
than the 4 M-parameter CNN. Tuning (lower backbone learning rate, label
smoothing) closed most of the gap. In the harder 3-class task the hybrid gives
our best result (F1 0.945). An honest ablation that shows this trade-off is part
of the contribution.

**Q: What is the difference between concat and cross-attention fusion?**
A: Concat joins the two feature vectors side by side. Cross-attention lets each
CNN feature (query) look at all Transformer tokens (keys and values) and take
the relevant global information before pooling.

**Q: Why is your accuracy lower than published papers (98–99%)?**
A: Many datasets contain augmented copies of the same image. If copies fall in
both training and test data, the model is tested on images it has already seen.
We grouped 402 near-duplicate images with perceptual hashing and kept each
group in one split, so our numbers are lower but trustworthy.

**Q: What is perceptual hashing?**
A: pHash converts an image to a 64-bit fingerprint from its low-frequency DCT
components. Visually similar images get similar hashes. We treat images within
a Hamming distance of 4 bits as duplicates.

**Q: Your proposal had 5 classes. Why 3?**
A: The severity classes (early, moderate, severe) need expert labels, and no
public dataset provides them. Inventing labels would make the results
meaningless. We used the classes the data supports and list severity grading as
future work.

**Q: Explain Grad-CAM.**
A: Take the gradient of the predicted class score with respect to the last
convolutional feature maps, average it per channel to get weights, take the
weighted sum of the maps and apply ReLU. The result is a heatmap of the regions
that increased the class score.

**Q: Explain LIME.**
A: Split the image into superpixels, create many copies with random superpixels
hidden, get the model's prediction for each copy, and fit a simple linear model.
The superpixels with the largest positive weights are the explanation.

**Q: Explain SHAP.**
A: SHAP assigns each input a contribution based on Shapley values from
cooperative game theory: the average change in the output when that input is
added, over all combinations. We use GradientExplainer, which approximates this
with expected gradients over background images.

**Q: Why does one Grad-CAM map highlight a corner of the image?**
A: Some images contain watermarks. The model may partly use such shortcuts. This
is exactly what explainable AI is for: it exposes the bias. Future work is
background removal or cattle segmentation.

**Q: Which preprocessing did you apply?**
A: RGB conversion, median filter (kernel 3) for noise, resize to 224 × 224,
normalisation with ImageNet statistics, and training-only augmentation:
rotation, flip, random crop (zoom), brightness and contrast.

**Q: How did you handle class imbalance?**
A: Class-weighted cross-entropy, stratified splitting, and macro-averaged
metrics so that every class counts equally.

**Q: Why EfficientNet-B0 and DeiT-Tiny?**
A: Both are accurate for their size and fit a 6 GB GPU together, with mixed
precision and batch size 32.

**Q: Can it be used by farmers?**
A: It is a research prototype, not a diagnostic tool. A field version needs
field images, veterinary validation and a mobile application.

**Q: What would you do with more time?**
A: Severity labels with veterinarians, more skin diseases, k-fold
cross-validation, an external test set, and a mobile application.
