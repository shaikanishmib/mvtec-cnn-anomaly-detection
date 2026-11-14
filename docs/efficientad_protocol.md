# EfficientAD: reference and pilot protocol

Primary source: Batzner, Heckler and König, WACV 2024,
[paper](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html).
CVF returned 403 during preparation; the [author's arXiv v3](https://arxiv.org/html/2303.14535v3)
provided full text, equations, Algorithms 1–2 and Tables 6–8.

The small PDN has four convolutions and two average pools: channels
3→128→256→256→384, kernels 4,4,3,4. Its 33-pixel receptive field supplies local descriptors.
Paper padding yields 64×64 features from 256×256 images. The medium variant uses
256/512 intermediate channels and extra 1×1 layers; it increases arithmetic,
activation memory and capacity. No medium pilot has been run. Parameter count
alone does not determine latency: early downsampling is central to PDN efficiency.

The teacher is distilled on ImageNet from pretrained WideResNet-101 feature
descriptors. It is frozen during anomaly learning. Distillation and anomaly
learning are distinct stages; an arbitrary pretrained backbone is not a PDN.
Teacher channel means and population standard deviations are fitted only to
fitting normals. The student is a fresh PDN with 768 output channels: 384 local
teacher predictions and 384 autoencoder predictions.

For normalized teacher t, student heads s1,s2 and autoencoder a:

- D=(t−s1)²; Lhard=mean(D[D≥quantile(D,.999)]).
- Lpenalty=mean(s1(P)²) for legitimate unrelated external images P.
- Lae=mean((t(Iaug)−a(Iaug))²).
- Lstae=mean((a(Iaug)−s2(Iaug))²).
- Ltotal=Lhard+Lpenalty+Lae+Lstae. Both trainable branches receive Lstae gradients.

The global autoencoder compresses 256×256 to a 64×1×1 bottleneck and decodes to
384×64×64. This supplies whole-image context, while the local branch sees small
patches. The pilot uses fixed brightness 1.1 augmentation to make the before/after
probe comparable; paper training randomly chooses brightness, contrast or saturation
with coefficient U(.8,1.2). This difference is disclosed, not selected by test scores.

Local map=channel mean((t−s1)²); global map=channel mean((a−s2)²).
Resize each to image resolution, then normalize each with development-normal
q90/q99.5: Mhat=.1(M−q90)/(q99.5−q90). Combine by equal average; image score is
the maximum. Scores are not clipped and larger means more anomalous. Degenerate
map quantiles raise an error. Reserved calibration normals later fit strict `score > threshold`;
development map scaling and final threshold fitting are different operations.

## Assets and licenses

Selected checkpoint: [unofficial implementation](https://github.com/nelson1425/EfficientAD),
revision `fcab5146f84ae17597044ad5ddf1656ccf805401`, `models/teacher_small.pth`.
Repository declares Apache-2.0; local license copy is in
`docs/third_party/efficientad/LICENSE`. No separate checkpoint license/model card
or independently verifiable original training log was found. It is a public
third-party checkpoint, not an author-released or independently reproduced teacher.
Its accompanying pretraining code specifies torchvision WideResNet101_2
IMAGENET1K_V1, layer2/layer3 descriptors and ImageNet inputs. These are claimed
provenance, not independently attested facts about the downloaded tensor bytes.
Record SHA256 and exact bytes in pilot artifacts; do not redistribute weights.

MVTec AD remains local under CC BY-NC-SA 4.0. No ImageNet images are present in the
project and none were downloaded. ImageNet access requires its own terms and
legitimate acquisition. The optional penalty is implemented but omitted in this
pilot, making it an EfficientAD-S no-penalty engineering variant. Reusing MVTec
test images as penalty inputs is forbidden. Another legitimate path would be
licensed unrelated images with a frozen manifest; a newly distilled teacher on a
small licensed corpus would be an explicitly bounded alternative, not equivalent
to ImageNet pretraining. Neither alternative has been executed.

## Budget and gate

The pilot budget and proposed 12-fit development budget were recorded in
`docs/milestone_status.md` before training. No development or final benchmark is
started by the pilot script. Paper training uses 70,000 steps; 32 pilot steps only
test engineering viability. Extrapolations must include a thermal/runtime margin
and exclude any claim of converged anomaly detection. The complete milestone
requires later development, calibration, 45 frozen category-seed evaluations,
failure analysis, nine metrics and exact six-model comparison. No such results
are available at this stage.
