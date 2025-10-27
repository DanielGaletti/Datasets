# 🧠 Image Segmentation and Computer Vision Datasets

This repository gathers a **comprehensive collection of datasets used in Computer Vision and Image Segmentation**.  
It covers various domains such as **semantic segmentation**, **instance segmentation**, **medical imaging**, **urban scenes**, and **interactive segmentation**.

The goal is to provide a consolidated reference containing essential information — number of images, mask availability, resolution, dataset type, number of classes, description, and download links — to help researchers and developers choose suitable datasets for **Deep Learning**, **Active Learning**, **Object Detection**, and **Scene Understanding** tasks.

---

## 📊 Dataset Overview

<details>
<summary><strong>All List (Click here)</strong></summary>

| **Dataset Name** | **# Imagens** | **Máscaras** | **Tamanho** | **Resolução** | **Tipo** | **# Classes** | **Descrição** | **Ano** | **Link** | **Público?** |
|:-----------------|:--------------|:-------------|:-------------|:---------------------|:--------------------------------|:--------------|:----------------|:----------|:----------|:------------:|
| VOC 2012 | 17,000 | ✅ Sim | 4 GB | 500×375 | Object Segmentation | 20 | Splits de treino/validação/teste com anotações por pixel e rótulos de objetos. | 2012 | [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset) | ✅ |
| CityScapes | 25,000 | ✅ Sim | 25 GB | 2048×1024 | Urban Segmentation | 30 | 50 cidades com anotações em nível de pixel para 30 classes. | 2016 | [Official Site](https://www.cityscapes-dataset.com/) | ✅ |
| COCO | 330,000 | ✅ Sim | 50 GB | Variável | Object Segmentation | 80 | Cenas complexas com múltiplas máscaras de objetos. | 2014 | [COCO](https://cocodataset.org/#home) | ✅ |
| LVIS | 164,000 | ✅ Sim | 25 GB | Variável | Instance Segmentation | 1,203 | Benchmark para segmentação de instância de cauda longa (long-tail). | 2019 | [LVIS](https://www.lvisdataset.org/) | ✅ |
| ADE20K | 27,000 | ✅ Sim | 3 GB | Variável | Scene Parsing | 150 | Benchmark completo para segmentação de cenas. | 2016 | [MIT CSAIL](https://ade20k.csail.mit.edu/) | ✅ |
| GTA V Synthetic | 25,000 | ✅ Sim | 180 GB | 1914×1052 | Synthetic Semantic Segmentation | 19 | Cenas urbanas sintéticas do GTA V com anotações de pixel perfeitas. | 2016 | [VISINF](https://download.visinf.tu-darmstadt.de/data/from_games/) | ✅ |
| BraTS | 3,000 (3D) | ✅ Sim | 200 GB | 240×240×155 | 3D Medical Segmentation | 3 | Dataset de tumor cerebral com rótulos para edema, necrose e tumor ativo. | 2012 | [CBICA](https://www.med.upenn.edu/cbica/brats2018/data.html) | ❌ |
| LiTS | 130 CT (3D) | ✅ Sim | 80 GB | 512×512×Z | 3D Medical Segmentation | 2 | Dataset para segmentação 3D de fígado e lesões. | 2017 | [CodaLab](https://competitions.codalab.org/competitions/17094) | ❌ |
| Kvasir-SEG | 1,000 | ✅ Sim | 2 GB | 576×720 | Medical Segmentation | 1 | Dataset de pólipos colorretais com máscaras binárias. | 2020 | [Simula](https://datasets.simula.no/kvasir-seg/) | ✅ |
| Nuclei | 30,000 patches | ✅ Sim | 100 MB | 50×50 | Biomedical Segmentation | 1 | Dataset de núcleos celulares com máscaras binárias. | 2018 | [Kaggle](https://www.kaggle.com/datasets/espsiyam/nuclei-image-segmentation) | ✅ |
| CVC-ClinicDB | 612 | ✅ Sim | 50 MB | 384×288 | Medical Segmentation | 1 | Frames de colonoscopia para detecção de pólipos. | 2015 | [Kaggle](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) | ✅ |
| REFUGE2 | 1,200 | ✅ Sim | 3.8 GB | Variável | Medical Segmentation | 2 | Segmentação de disco e escavação óptica para triagem de glaucoma. | 2020 | [Challenge](https://refuge.grand-challenge.org/) | ✅ |
| ISIC | 1,203,225 | ✅ Sim | Variável | Variável | Medical (Dermatology) | 2–7 | Dataset massivo para segmentação de lesões de pele. | 2016 | [ISIC Archive](https://www.isic-archive.com/) | ✅ |
| BrainMRI | 3,929 | ✅ Sim | 350 MB | 256×256 | Medical Segmentation | 1 | Dataset para segmentação de tumores cerebrais. | 2020 | [Kaggle](https://www.kaggle.com/code/mateuszbuda/brain-segmentation-pytorch) | ✅ |
| LiverCT | 131 CT (3D) | ✅ Sim | 80 GB | 512×512×Z | 3D Medical Segmentation | 2 | Scans de TC para segmentação de lesões hepáticas. | 2017 | [CodaLab](https://competitions.codalab.org/competitions/17094) | ✅ |
| RESC | 110 scans | ✅ Sim | 500 MB | Variável | Medical Segmentation | 3 | Dataset para segmentação de edema retinal. | 2018 | [GitHub](https://github.com/ShawnBIT/AI-Challenger-Retinal-Edema-Segmentation) | ✅ |
| TN3K | 3,500 | ✅ Sim | 200 MB | 400×400 | Medical Segmentation | 1 | Dataset de ultrassom para segmentação de nódulos da tireoide. | 2022 | [Kaggle](https://www.kaggle.com/datasets/tjahan/tn3k-thyroid-nodule-region-segmentation-dataset) | ✅ |
| DDTI | 5,000 | ✅ Sim | 1.5 GB | Variável | Medical Segmentation | 1 | Raio-x panorâmicos dentários para segmentação de dentes. | 2022 | [Kaggle](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) | ✅ |
| TG3K | 3,100 | ✅ Sim | 250 MB | 400×400 | Medical Segmentation | 1 | Dataset de ultrassom para segmentação da glândula tireoide. | 2022 | [OpenMedLab](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TG3K) | ✅ |
| BUSI | 780 | ✅ Sim | 250 MB | 500×500 | Medical Segmentation | 3 | Dataset de ultrassom de mama para segmentação. | 2019 | [Dataset Page](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | ✅ |
| CHAOS | 80 scans (3D) | ✅ Sim | 20 GB | 512×512×Z | 3D Medical Segmentation | 4 | Scans de TC e RM para segmentação de fígado, rins e baço. | 2019 | [CHAOS](https://chaos.grand-challenge.org/) | ✅ |
| ROCO | 81,000 | ❌ Não | 8 GB | Variável | Medical Captioning | – | Imagens de radiologia com legendas textuais. | 2018 | [GitHub](https://github.com/razorx89/roco-dataset) | ✅ |
| MedPix | 59,000 | ❌ Não | Variável | Variável | Medical Image Database | – | Arquivo de imagens clínicas e de diagnóstico. | 1999 | [MedPix](https://medpix.nlm.nih.gov/home) | ✅ |
| NLPR | 1,000 pares | ✅ Sim | 998 MB | 640×480 | Salient Object Detection | 1 | Capturado com Microsoft Kinect, com cenas internas e externas. | – | [HyperAI](https://hyper.ai/en/datasets/17525) | ✅ |
| PaviaU | 1 imagem | ❌ Não | 100 MB | 610×340×103 | Spectral Classification | 9 | Imagem hiperespectral capturada sobre Pavia, Itália. | – | [Kaggle](https://www.kaggle.com/datasets/syamkakarla/pavia-university-hsi) | ✅ |
| BSDS500 | 500 | ✅ Sim | 100 MB | Variável | Contour Detection | – | Benchmark de detecção de contorno e segmentação com anotações humanas. | – | [Kaggle](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500) | ✅ |
| NYUV2 | 1,449 | ✅ Sim | 5.5 GB | 640×480 | Indoor Scene Segmentation | 40 | Dataset RGB-D capturado com Microsoft Kinect. | 2012 | [NYU](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | ✅ |
| SUNRGBD | 10,335 | ✅ Sim | 60 GB | Variável | 2D/3D Segmentation | 37 | Cenas internas 3D densamente anotadas. | 2015 | [Princeton](https://rgbd.cs.princeton.edu/) | ✅ |
| CamVid | 701 frames | ✅ Sim | 570 MB | 960×720 | Video Semantic Segmentation | 12 | Primeiro dataset de vídeo com anotações de pixel para cenas urbanas. | 2008 | [CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | ✅ |
| 300W-LP | 122,450 | ❌ Não | 4 GB | Variável | Landmark Detection | 68 | Versão aumentada do 300W com imagens faciais rotacionadas. | 2016 | [TensorFlow](https://www.tensorflow.org/datasets/catalog/the300w_lp?hl=pt-br) | ✅ |
| Visual Genome | 108,000 | ❌ Não | 12 GB | Variável | Image Captioning | – | Relações de objetos e anotações em linguagem natural. | 2016 | [VG](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | ✅ |
| ISPRS Vaihingen | 33 | ✅ Sim | 2 GB | ~2500×2000 | Aerial Image Segmentation | 6 | Imagens aéreas UHD com rótulos semânticos. | 2012 | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) | ✅ |
| NJU2K | 1,985 | ✅ Sim | 1.5 GB | Variável | Salient Object Detection | 1 | Pares de imagens RGB para detecção de objetos salientes. | 2014 | [HyperAI](https://hyper.ai/en/datasets/18303) | ✅ |
| STERE | 1,000 | ✅ Sim | 100 MB | 1024×768 | Object Detection | 1 | Pares de imagens estéreo para detecção de objetos. | 2015 | [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | ✅ |
| GrabCut | 50 | ✅ Sim | 5 MB | Variável | Interactive Segmentation | 1 | Pequeno dataset para experimentos de segmentação interativa. | 2004 | [GitHub](https://github.com/irllabs/grabcut) | ✅ |
| Awesome Medical Datasets | - | ✅ Sim | - | - | Medical Image Segmentation | - | Uma coleção de múltiplos datasets médicos abertos. | - | [OpenMedLab](https://github.com/openmedlab/Awesome-Medical-Dataset) | ✅ |
| USPS | 9,298 | ❌ Não | 10 MB | 16×16 | Classification | 10 | Dataset de dígitos manuscritos de códigos postais. | 1990 | [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html) | ✅ |
| MNIST | 70,000 | ❌ Não | 15 MB | 28×28 | Classification | 10 | Clássico dataset de dígitos manuscritos. | 1998 | [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) | ✅ |
| BioID | 1,521 | ❌ Não | 150 MB | 384×288 | Face Detection | 1 | Dataset de localização facial em escala de cinza. | 1999 | [BioID](https://www.bioid.com/face-database/) | ✅ |

</details>
---

## 🧩 Notes

- ✅ **Public** datasets are freely available for research and educational use.  
- ❌ **Non-public** datasets may require registration, challenge participation, or access requests.  
- Some datasets (e.g., LiTS, BraTS) are **3D volumetric** and require preprocessing pipelines before use.

---

## 💡 How to Use

You can:
1. Explore datasets to benchmark segmentation models (e.g., U-Net, DeepLab, Mask R-CNN).
2. Use them in Active Learning or Continual Learning pipelines.
3. Combine multiple datasets to improve model generalization.

---

## 📚 Citation

If you use this list or parts of it, please cite this repository:

```bibtex
@misc{segmentation_datasets_collection,
  author = {Galetti, Daniel Martins},
  title = {Image Segmentation and Computer Vision Datasets Collection},
  year = {2025},
  url = {https://github.com/<your-username>/<your-repo>},
  note = {Comprehensive list of datasets for segmentation, detection, and scene understanding.}
}
