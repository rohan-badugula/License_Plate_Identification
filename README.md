# 🚗 Automatic License Plate Recognition Using YOLOv8 and EfficientNet B7

📄 **Published Paper:** [IEEE Xplore - ICCCIS 2023](https://ieeexplore.ieee.org/document/10425775)

> A two-stage ALPR system leveraging YOLOv8 for robust license plate detection and EfficientNet B7 for accurate character recognition.

---

## 🧠 Overview

This repository contains the implementation of an **Automatic License Plate Recognition System (ALPRS)** designed to perform real-time vehicle identification with high accuracy. The system utilizes:

- 🔍 **YOLOv8** – for detecting license plates from images.
- 🔤 **EfficientNet B7** – for recognizing characters on the plates.

This project addresses common ALPR challenges such as lighting variations and complex backgrounds using state-of-the-art deep learning techniques.

---

## 👨‍💻 Authors

- **Rohan Reddy B**
- **Gunti Swathi**

🎤 *Presented at the 4th International Conference on Computing Communication and Intelligent Systems (ICCCIS-2023), 3–4 November 2023.*

---

## 🚦 Problem Statement

Traditional manual vehicle identification is slow and prone to errors. Existing ALPR systems often fail in uncontrolled environments. This project aims to:
- Improve detection accuracy under real-world conditions.
- Achieve high-speed recognition suitable for traffic and surveillance systems.

---

## 💡 Proposed Solution

We propose a **two-stage deep learning-based ALPR system**:

### 1. License Plate Detection
Utilizes **YOLOv8**, a state-of-the-art object detection model with:
- Anchor-free architecture
- Lightweight backbone
- Advanced loss function
- Spatial pyramid pooling

### 2. Character Recognition
Uses **EfficientNet B7**, a high-performing CNN pre-trained on ImageNet, known for:
- High accuracy
- Lower computational cost
- Robust generalization

---

## 🛠️ Methodology

![System Architecture](images/img3.jpg)

### 🔲 Stage 1: License Plate Detection with YOLOv8
- Trained for two classes: license plate and background
- Steps:
  - Grid-based localization
  - Non-Maximum Suppression (NMS)
  - Post-processing refinements

### 🔡 Stage 2: Character Recognition with EfficientNet B7
- Tested multiple CNN models — EfficientNet B7 achieved the best accuracy.
- Final accuracy: **98.22%**

---

## 📊 Datasets Used

![Dataset](images/img2.jpg)

| Dataset Type | Description |
|--------------|-------------|
| **Artificial Mercosur** | 3,840 license plate images from 5 South American countries |
| **Open-source OCR** | 37,623 alphanumeric characters (0–9, A–Z) |
| **Custom Test Set** | 500 real-world annotated vehicle images |

---

## 📈 Results & Analysis

### 🔍 License Plate Detection

YOLOv8 outperformed traditional detectors in both precision and recall.

![Detection](images/img6.jpg)

### 🔡 Character Recognition

EfficientNet B7 achieved:
- **98.22% character-level accuracy**
- Superior generalization on unseen license plate styles

![Character Bar Chart](images/img4.jpg)
![Character Bar Chart](images/img5.jpg)

---

## 🧪 Real-World Testing

### Sample Inference on Camera-Captured Image
![Sample Result](images/img7.jpg)

### Evaluation on Custom Test Dataset
![Overall Result](images/img8.jpg)

---

## 🧾 Conclusion

This ALPR system demonstrates significant improvements in license plate recognition by combining:
- YOLOv8’s modern detection techniques
- EfficientNet B7’s precision in OCR tasks

The system is robust, lightweight, and suitable for deployment in real-time surveillance applications.

---

## 🔭 Future Work

- Increase robustness under challenging weather and lighting conditions
- Extend dataset diversity to include global license formats
- Real-time inference optimization for deployment on edge devices

---

## 🙏 Acknowledgements

We extend our gratitude to the organizers of **ICCCIS 2023** for providing an opportunity to present our research and to the open-source community for datasets and tools that made this project possible.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvement or wish to collaborate, feel free to:

- Create a pull request
- Open an issue
- Reach out directly

---

## 📬 Contact

For questions, feedback, or collaborations:

**Rohan Reddy B**  
📧 [rohanbadugula9@gmail.com](mailto:rohanbadugula9@gmail.com)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

⭐ *If you find this repository helpful, please consider starring it!*
