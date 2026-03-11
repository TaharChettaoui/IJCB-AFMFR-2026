# IJCB-AFMFR-2026

## Overview

This competition aims to systematically **benchmark, evaluate and compare adaptation strategies for foundation models for the downstream task of FR** within a privacy-friendly framework. By providing a standardized evaluation protocol and metrics, the competition will highlight the strengths and limitations of different approaches, including their ability to generalize across diverse datasets. 

The competition will feature two tracks:
- In the **first track**, participants will be provided with the full training dataset.
- In the **second track**, only a small subset of the data may be used to adapt the models.

The results are expected to guide future research and encourage the development of effective, data-efficient adaptation methods for foundation models in FR. The final competition paper will be submitted to **IEEE/IAPR IJCB 2026** and the top-performing teams will be invited as co-authors.

---

## CLIP Foundation Model
In this competition, participants are restricted to using the **Contrastive Language–Image Pretraining (CLIP)** foundation model. **CLIP** is a multimodal model developed by OpenAI that learns joint representations of images and text. It is trained to associate images with their corresponding textual descriptions, allowing it to understand visual concepts through natural language. **CLIP** consists of two main components: an image encoder, which converts images into feature embeddings, and a text encoder, which converts textual descriptions into embeddings in the same feature space. This shared embedding space allows the model to measure the similarity between images and text.

Participants are restricted to use the pretrained  **ViT-B/16** variant of **CLIP**. Participants are free to use both encoders or only the image encoder, depending on their approach. Code to import and test the model is provided in `export_clip_to_onnx.py` (in the provided example, we exclusively use the image encoder). The official source code to load CLIP is provided [HERE](https://github.com/openai/CLIP) and the pretrained ViT-B/16 is provided [HERE](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt). Please strictly follow their terms of use and licenses. 

---

## Submission Guidelines

- **Model Format:**  
  - Submissions must be provided as a **ZIP file containing one or two pretrained models**, based on each selected track. Participants can freely choose to participate in Track 1, Track 2, or both tracks.
  - Teams may upload their adapted models as a ZIP file to a **cloud provider of their choice**, provided that it is accessible in **Germany** without requiring an account registration.  

- **Model Creation Instructions:**    
  - Instructions and example code for exporting a **CLIP ViT-B/16 model** to **ONNX** are provided in `export_clip_to_onnx.py`.  
  - The script also includes an **evaluation step** to verify that the exported ONNX model produces the same outputs as the original PyTorch model, ensuring that the model conversion is correct. For testing purposes, the provided code should achieve approximately **93.50% accuracy on the LFW dataset** when the model is exported and evaluated correctly. You can download the LFW `.bin` evaluation data using the following link: [HERE](https://owncloud.fraunhofer.de/index.php/s/AQ9s1XqCKyfVnAZ)
 
- **BEFORE Submitting:**  
  - We provide a Python script `test.py` that can be used to upload and test your model on the LFW `.bin` dataset. This script is intended solely to verify that the ONNX model you provided is working correctly.
  - **Before submitting**, test your model using the provided script without adding imports to the code. If any modifications to the `test.py` code are made (e.g., adapting the ONNX wrapper), please include the modified code in the submission ZIP file.

- **Rules and Restrictions:**
  - The final submitted model must have the exact same architecture as the CLIP ViT-B/16 architecture.
  - Participants may use the image encoder alone or both the image and text encoders of the CLIP ViT-B/16 foundation model.
  - The use of external models other than CLIP ViT-B/16 (image and text encoders) is not allowed. This also prohibits the use of, for example, pretrained face recognition models for knowledge distillation.
  - No external setup, installation, or internet access is allowed at runtime.
  
---

## Execution Environment  
- You can install the required dependencies to create an ONNX model using the requirements.txt file: `pip install -r requirements.txt`
- The provided code was tested using **cudatoolkit 11.8** and **cudnn 8.9**.

---

## FAQ

**Q:** How can I register for the competition?  
**A:** `Registration instructions are available on the competition website https://sites.google.com/view/ijcb-afmfr-2026`    

---

## Contact

For questions, please contact:  
- **Email:** Tahar Chettaoui `tahar.chettaoui@igd.fraunhofer.de`
  
---

