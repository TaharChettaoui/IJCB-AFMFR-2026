# IJCB-AFMFR-2026

## Overview

The aim of this competition is to systematically **benchmark, evaluate and compare adaptation strategies for foundation models for the downstream task of FR** within a privacy-friendly framework. By providing a standardized evaluation protocol and metrics, the competition will highlight the strengths and limitations of different approaches, including their ability to generalize across diverse datasets. 

The competition will feature two tracks:
- In the **first track**, participants will be provided with the full training dataset.
- In the **second track**, only a small subset of the data may be used to adapt the models.

The results are expected to guide future research and encourage the development of effective, data-efficient adaptation methods for foundation models in FR. The final competition paper will be submitted to **IEEE/IAPR IJCB 2026** and the top-performing teams will be invited as co-authors.

---

## Submission Guidelines

- **Model Format:**  
  - Submissions must be provided as a **ZIP file containing two trained models**, one for each track.
  - Teams may upload their training data as a ZIP file to a **cloud provider of their choice**, provided that it is accessible in **Germany** without requiring an account registration.  


- **Model Creation Instructions:**  
  - Instructions and example code for exporting a **CLIP ViT-B/16 model** to **ONNX** are provided in `export_clip_to_onnx.py`.  
  - The script also includes an **evaluation step** to verify that the exported ONNX model produces the same outputs as the original PyTorch model, ensuring that the model conversion is correct. For testing purposes, the provided code should achieve approximately **93.50% accuracy on the LFW dataset** when the model is exported and evaluated correctly. You can download the LFW `.bin` evaluation data using the following link: [HERE](https://owncloud.fraunhofer.de/index.php/s/AQ9s1XqCKyfVnAZ)
 
- **BEFORE Submitting:**  
  All participants must ensure that their submitted code runs in the specified execution environment described below. In addition, we provide a Python script `test.py` that can be used to upload and test your model on the LFW .bin dataset. **Before submitting**, test your model using the provided script without modifying or adding imports to the code. The only allowed changes are adapting the paths so they point to your model and the evaluation data (LFW).

- **Deadline:**  
  All submissions must be received by **10.05.2026 (Anywhere on Earth, AOE)**.

---

## Execution Environment
- The models must run on **Ubuntu 24.04** and **Python 3.9**.
- The provided code was tested using cudatoolkit 11.8 and cudnn 8.9
- You can install the required dependencies to create an ONNX model using the requirements.txt file: `pip install -r requirements.txt`
- No external setup, installation, or internet access is allowed at runtime.  

---

## Evaluation

- **Evaluation Data:** The dataset used for evaluation will **not be released** to participants.  
- **Reproducibility Checks:** Top-performing models will be retrained and reevaluated by the competition organizers.

---

## FAQ

**Q:** How can I register for the competition?  
**A:** `Registration instructions are available on the competition website https://sites.google.com/view/ijcb-afmfr-2026`    

---

## Contact

For questions, please contact:  
- **Email:** Tahar Chettaoui `tahar.chettaoui@igd.fraunhofer.de`
  
---

