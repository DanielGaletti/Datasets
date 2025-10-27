# 🧠 Image Segmentation and Computer Vision Datasets

This repository gathers a **comprehensive collection of datasets used in Computer Vision and Image Segmentation**.  
It covers various domains such as **semantic segmentation**, **instance segmentation**, **medical imaging**, **urban scenes**, and **interactive segmentation**.

The goal is to provide a consolidated reference containing essential information — number of images, mask availability, resolution, dataset type, number of classes, description, and download links — to help researchers and developers choose suitable datasets for **Deep Learning**, **Active Learning**, **Object Detection**, and **Scene Understanding** tasks.

---

## 📊 Dataset Overview

The full list contains over 40 datasets. Click below to expand the table.

<details>
<summary><strong>View the Full Dataset List (Click to expand)</strong></summary>

| **Dataset Name** | **# Images** | **Masks** | **Size** | **Resolution** | **Kind of Dataset** | **# Classes** | **Description** | **Year** | **Link** | **Public?** |
|:-----------------|:--------------|:----------|:------------|:---------------------|:--------------------------------|:--------------|:----------------|:----------|:----------|:-----------:|
| VOC 2012 | 17,000 | ✅ Yes | 4 GB | 500×375 | Object Segmentation | 20 | Includes training/validation/test splits with per-pixel annotations and object labels. | 2012 | [Kaggle](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset) | ✅ |
| CityScapes | 25,000 | ✅ Yes | 25 GB | 2048×1024 | Urban Segmentation | 30 | 50 different cities with pixel-level annotations for 30 classes. | 2016 | [Official Site](https://www.cityscapes-dataset.com/) | ✅ |
| COCO | 330,000 | ✅ Yes | 50 GB | Variable | Object Segmentation | 80 | Complex scenes with multiple object masks. | 2014 | [COCO](https://cocodataset.org/#home) | ✅ |
| LVIS | 164,000 | ✅ Yes | 25 GB | Variable | Instance Segmentation | 1,203 | Long-tail instance segmentation benchmark. | 2019 | [LVIS](https://www.lvisdataset.org/) | ✅ |
| ADE20K | 27,000 | ✅ Yes | 3 GB | Variable | Scene Parsing | 150 | Complete scene segmentation benchmark. | 2016 | [MIT CSAIL](https://ade20k.csail.mit.edu/) | ✅ |
| GTA V Synthetic | 25,000 | ✅ Yes | 180 GB | 1914×1052 | Synthetic Semantic Segmentation | 19 | Synthetic urban scenes from GTA V with perfect pixel annotations. | 2016 | [VISINF](https://download.visinf.tu-darmstadt.de/data/from_games/) | ✅ |
| BraTS | 3,000 (3D) | ✅ Yes | 200 GB | 240×240×155 | 3D Medical Segmentation | 3 | Brain tumor dataset with edema, necrosis, and active tumor labels. | 2012 | [CBICA](https://www.med.upenn.edu/cbica/brats2018/data.html) | ❌ |
| LiTS | 130 CT (3D) | ✅ Yes | 80 GB | 512×512×Z | 3D Medical Segmentation | 2 | 3D liver and lesion segmentation dataset. | 2017 | [CodaLab](https://competitions.codalab.org/competitions/17094) | ❌ |
| Kvasir-SEG | 1,000 | ✅ Yes | 2 GB | 576×720 | Medical Segmentation | 1 | Colorectal polyp dataset with binary masks. | 2020 | [Simula](https://datasets.simula.no/kvasir-seg/) | ✅ |
| Nuclei | 30,000 patches | ✅ Yes | 100 MB | 50×50 | Biomedical Segmentation | 1 | Cell nuclei dataset with binary masks. | 2018 | [Kaggle](https://www.kaggle.com/datasets/espsiyam/nuclei-image-segmentation) | ✅ |
| CVC-ClinicDB | 612 | ✅ Yes | 50 MB | 384×288 | Medical Segmentation | 1 | Colonoscopy frames for polyp detection. | 2015 | [Kaggle](https://www.kaggle.com/datasets/balraj98/cvcclinicdb) | ✅ |
| REFUGE2 | 1,200 | ✅ Yes | 3.8 GB | Variable | Medical Segmentation | 2 | Retinal disc and cup segmentation for glaucoma screening. | 2020 | [Challenge](https://refuge.grand-challenge.org/) | ✅ |
| ISIC | 1,203,225 | ✅ Yes | Variable | Variable | Medical (Dermatology) | 2–7 | Massive dataset for skin lesion segmentation. | 2016 | [ISIC Archive](https://www.isic-archive.com/) | ✅ |
| BrainMRI | 3,929 | ✅ Yes | 350 MB | 256×256 | Medical Segmentation | 1 | Brain tumor segmentation dataset. | 2020 | [Kaggle](https://www.kaggle.com/code/mateuszbuda/brain-segmentation-pytorch) | ✅ |
| LiverCT | 131 CT (3D) | ✅ Yes | 80 GB | 512×512×Z | 3D Medical Segmentation | 2 | CT scans for liver injury segmentation. | 2017 | [CodaLab](https://competitions.codalab.org/competitions/17094) | ✅ |
| RESC | 110 scans | ✅ Yes | 500 MB | Variable | Medical Segmentation | 3 | Retinal edema segmentation dataset. | 2018 | [GitHub](https://github.com/ShawnBIT/AI-Challenger-Retinal-Edema-Segmentation) | ✅ |
| TN3K | 3,500 | ✅ Yes | 200 MB | 400×400 | Medical Segmentation | 1 | Thyroid nodule ultrasound segmentation dataset. | 2022 | [Kaggle](https://www.kaggle.com/datasets/tjahan/tn3k-thyroid-nodule-region-segmentation-dataset) | ✅ |
| DDTI | 5,000 | ✅ Yes | 1.5 GB | Variable | Medical Segmentation | 1 | Panoramic dental x-rays for teeth segmentation. | 2022 | [Kaggle](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images) | ✅ |
| TG3K | 3,100 | ✅ Yes | 250 MB | 400×400 | Medical Segmentation | 1 | Ultrasound thyroid gland segmentation dataset. | 2022 | [OpenMedLab](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TG3K) | ✅ |
| BUSI | 780 | ✅ Yes | 250 MB | 500×500 | Medical Segmentation | 3 | Breast ultrasound segmentation dataset. | 2019 | [Dataset Page](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | ✅ |
| CHAOS | 80 scans (3D) | ✅ Yes | 20 GB | 512×512×Z | 3D Medical Segmentation | 4 | MRI and CT scans for liver, kidneys, and spleen segmentation. | 2019 | [CHAOS](https://chaos.grand-challenge.org/) | ✅ |
| ROCO | 81,000 | ❌ No | 8 GB | Variable | Medical Captioning | – | Radiology images paired with textual captions. | 2018 | [GitHub](https://github.com/razorx89/roco-dataset) | ✅ |
| MedPix | 59,000 | ❌ No | Variable | Variable | Medical Image Database | – | Clinical and diagnostic image archive. | 1999 | [MedPix](https://medpix.nlm.nih.gov/home) | ✅ |
| NLPR | 1,000 pairs | ✅ Yes | 998 MB | 640×480 | Salient Object Detection | 1 | Captured by Microsoft Kinect with indoor and outdoor scenes. | – | [HyperAI](https://hyper.ai/en/datasets/17525) | ✅ |
| PaviaU | 1 image | ❌ No | 100 MB | 610×340×103 | Spectral Classification | 9 | Hyperspectral image captured over Pavia, Italy. | – | [Kaggle](https://www.kaggle.com/datasets/syamkakarla/pavia-university-hsi) | ✅ |
| BSDS500 | 500 | ✅ Yes | 100 MB | Variable | Contour Detection | – | Human-annotated segmentation and contour detection benchmark. | – | [Kaggle](https://www.kaggle.com/datasets/balraj98/berkeley-segmentation-dataset-500-bsds500) | ✅ |
| NYUV2 | 1,449 | ✅ Yes | 5.5 GB | 640×480 | Indoor Scene Segmentation | 40 | RGB-D dataset captured using Microsoft Kinect. | 2012 | [NYU](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | ✅ |
| SUNRGBD | 10,335 | ✅ Yes | 60 GB | Variable | 2D/3D Segmentation | 37 | Densely annotated 3D indoor scenes. | 2015 | [Princeton](https://rgbd.cs.princeton.edu/) | ✅ |
| CamVid | 701 frames | ✅ Yes | 570 MB | 960×720 | Video Semantic Segmentation | 12 | First video dataset with pixel-level annotations for urban scenes. | 2008 | [CamVid](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) | ✅ |
| 300W-LP | 122,450 | ❌ No | 4 GB | Variable | Landmark Detection | 68 | Augmented version of 300W with rotated facial images. | 2016 | [TensorFlow](https://www.tensorflow.org/datasets/catalog/the300w_lp?hl=en) | ✅ |
| Visual Genome | 108,000 | ❌ No | 12 GB | Variable | Image Captioning | – | Object relationships and natural language annotations. | 2016 | [VG](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | ✅ |
| ISPRS Vaihingen | 33 | ✅ Yes | 2 GB | ~2500×2000 | Aerial Image Segmentation | 6 | UHD aerial imagery with semantic labels. | 2012 | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) | ✅ |
| NJU2K | 1,985 | ✅ Yes | 1.5 GB | Variable | Salient Object Detection | 1 | RGB image pairs for salient object detection. | 2014 | [HyperAI](https://hyper.ai/en/datasets/18303) | ✅ |
| STERE | 1,000 | ✅ Yes | 100 MB | 1024×768 | Object Detection | 1 | Stereo image pairs for object detection. | 2015 | [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | ✅ |
| GrabCut | 50 | ✅ Yes | 5 MB | Variable | Interactive Segmentation | 1 | Small dataset for interactive segmentation experiments. | 2004 | [GitHub](https://github.com/irllabs/grabcut) | ✅ |
| Awesome Medical Datasets | - | ✅ Yes | - | - | Medical Image Segmentation | - | A collection of multiple open medical datasets. | - | [OpenMedLab](https://github.com/openmedlab/Awesome-Medical-Dataset) | ✅ |
| USPS | 9,298 | ❌ No | 10 MB | 16×16 | Classification | 10 | Handwritten digit dataset from postal codes. | 1990 | [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html) | ✅ |
| MNIST | 70,000 | ❌ No | 15 MB | 28×28 | Classification | 10 | Classic handwritten digit dataset. | 1998 | [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) | ✅ |
| BioID | 1,521 | ❌ No | 150 MB | 384×288 | Face Detection | 1 | Grayscale face localization dataset. | 1999 | [BioID](https://www.bioid.com/face-database/) | ✅ |

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
