# Licensing, provenance and attribution review

This is a rights inventory, not a new license grant or a legal opinion. No
project-wide license has been added. Checked sources and retained notices must
be read separately from the code's technical reproducibility.

## Supported statements

| Material | Evidence and supported scope |
|---|---|
| Original repository source | No root project license is currently declared. Public visibility permits viewing/forking under the platform's terms; it does not establish general reuse or commercial rights. |
| MVTec AD images/masks | The [dataset author's page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) states CC BY-NC-SA 4.0. Its attribution, noncommercial and share-alike terms apply to dataset use. Raw images/masks are not included here. |
| Dataset-derived banks and outputs | Not distributed. Do not assume fitted artifacts remove dataset restrictions or grant commercial use; obtain appropriate rights before redistribution/deployment. |
| ResNet18 features | The manifest pins torchvision ImageNet1K V1 origin, original weight SHA and exported trunk SHA. [Torchvision code](https://github.com/pytorch/vision/blob/main/LICENSE) is BSD-3-Clause; that alone does not resolve rights in ImageNet source images or every pretrained asset. See [ImageNet access terms](https://www.image-net.org/download.php). |
| EfficientAD teacher | Unofficial nelson1425/EfficientAD checkpoint, pinned revision/hash in the [fidelity audit](efficientad_fidelity_audit.md). Repository Apache-2.0 notice is [retained](third_party/efficientad/LICENSE); separate weight licensing/model card and complete distillation history remain unconfirmed. No teacher weights are published. |
| SuperSimpleNet reference | The upstream MIT notice is [retained](third_party/SuperSimpleNet_LICENSE.txt). The local method is explicitly SuperSimpleNet Lite; the notice does not license unrelated original repository material. |
| Public UI screenshots | Original procedural canvas fixture plus genuine backend response. They contain no MVTec pixels; [capture provenance](../reports/frontend/validation.json) records hashes and scope. This does not create a separate screenshot license grant. |
| Runtime/build dependencies | Their own notices remain applicable. PyTorch/torchvision use BSD-style licenses; FastAPI, React and Vite use MIT; nginx and Node include their own notices and bundled third-party terms. CUDA/NVIDIA components and Docker Desktop have separate terms. A dependency's permissive code license does not license this whole repository. |

Primary dependency notices: [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE),
[FastAPI](https://github.com/fastapi/fastapi/blob/master/LICENSE),
[React](https://github.com/facebook/react/blob/main/LICENSE),
[Vite](https://github.com/vitejs/vite/blob/main/LICENSE),
[nginx](https://nginx.org/LICENSE),
[Docker Desktop terms](https://www.docker.com/legal/docker-subscription-service-agreement/).
This is a selected dependency inventory, not a complete transitive legal bill of materials.
Torchvision's [versioned pretrained-model disclaimer](https://raw.githubusercontent.com/pytorch/vision/v0.26.0/README.md)
explicitly distinguishes model/dataset permissions from the library's code license.

## Owner decision recommended

For original source that the owner has the right to license, a conventional MIT
license is a reasonable portfolio/reuse choice after confirming ownership and
compatibility with retained third-party material. That is a recommendation for
an explicit owner decision, not authorization to add it now. Keep upstream notices,
their scope and model/data exclusions separate; decide documentation/figure rights
explicitly rather than implying a blanket grant. No recommendation clears
commercial rights in MVTec-derived models or the unofficial teacher checkpoint.

Until that decision, describe this as **publicly available source**, not as a
fully licensed open-source release. GitHub's
[no-license explanation](https://choosealicense.com/no-permission/) distinguishes
public visibility from general reuse permission. A source license cannot resolve
permissions the owner does not hold. No new SPDX license identifier, root LICENSE,
release tag or redistribution permission is introduced by this review.
