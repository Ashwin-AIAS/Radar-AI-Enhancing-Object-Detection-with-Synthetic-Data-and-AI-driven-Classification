# 🚀 AI-Based RADAR Object Classification Using GAN-Generated Synthetic Data 📡  

## 📌 Overview  
This project explores how **AI can improve RADAR-based perception** for autonomous systems by leveraging **GAN-generated synthetic data** to enhance classification accuracy.  

Traditional RADAR perception struggles due to **low resolution** and **limited labeled data**. We use **Generative Adversarial Networks (GANs)** to generate **synthetic RADAR data**, which augments real-world datasets, leading to **better model performance**.  

---

## 🔍 **Motivation**  
🔹 **RADAR is critical for autonomous vehicles** due to its ability to detect objects in all weather conditions.  
🔹 **However, RADAR has low spatial resolution**, making object classification difficult.  
🔹 **Collecting and labeling real RADAR data is expensive** and time-consuming.  
🔹 **Can we generate synthetic RADAR data** to augment real data and improve model accuracy?  

This project answers that question by using **GANs to synthesize high-quality RADAR data**, improving **deep learning-based object classification**.

---

## 🛠 **How It Works**  

### **Step 1: Data Collection & Processing**  
✅ Collected RADAR data consisting of:  
- **X, Y coordinates**  
- **Doppler velocity** (motion speed)  
- **Reflectivity** (RADAR cross-section)  

✅ Applied **K-Means Clustering** to pre-label objects as:  
- 🚗 **Cars**  
- 🚶 **Pedestrians**  
- 🏠 **Static Objects**  

---

### **Step 2: Building a CNN-Based Classification Model**  
✅ Designed a **Convolutional Neural Network (CNN)** to classify objects from RADAR sensor data.  
✅ Trained the CNN using **real sensor data**.  
✅ Achieved baseline performance but needed **more training data** to improve generalization.  

---

### **Step 3: Synthetic Data Generation Using GANs 🌀**  
✅ Trained a **Generative Adversarial Network (GAN)** to generate synthetic RADAR samples.  
✅ The **Generator** created new RADAR-like data from random noise.  
✅ The **Discriminator** learned to differentiate real from fake RADAR data.  
✅ Over multiple training iterations, the **GAN started producing realistic RADAR features**.  

---

### **Step 4: Combining Real & Synthetic Data for Training**  
✅ Merged **real and GAN-generated synthetic data** into a **final dataset**.  
✅ Retrained the CNN model using this expanded dataset.  
✅ **Result:** Higher accuracy & better generalization to new RADAR inputs.  

---

### **Step 5: AI Agent for Real-Time Object Detection**  
✅ Implemented an **AI-driven agent** that:  
- Accepts **live RADAR sensor feeds**  
- Runs the trained CNN model in real-time  
- Classifies detected objects as **car, pedestrian, or static object**  
- Outputs results for decision-making in autonomous systems  

---

## 🎯 **Final Outcomes**
✅ **GAN-generated synthetic RADAR data significantly improved model accuracy.**  
✅ **The AI system successfully classified RADAR-detected objects in real time.**  
✅ **Demonstrated how synthetic data can solve real-world sensor perception challenges.**  

---

## 🔬 **Why Use GANs for Sensor Data?**
💡 **Solves Data Scarcity:** Real RADAR data is hard to collect and label—GANs generate additional training data.  
💡 **Improves Model Generalization:** GAN-augmented datasets help models perform better on unseen test data.  
💡 **Enables Rare Event Simulation:** GANs can create **edge cases** (low-reflectivity pedestrians, occlusions) to improve model robustness.  

---

## 🔧 **What Can Be Improved?**
🔹 **Sensor Fusion:** Combine RADAR with **LiDAR & Camera data** for richer perception.  
🔹 **StyleGAN for Higher-Quality Synthesis:** Explore **more advanced GAN architectures** for better realism.  
🔹 **Edge Deployment:** Optimize AI model for real-time **embedded system inference (e.g., NVIDIA Jetson)**.  
🔹 **Self-Supervised Learning (SSL):** Reduce dependency on manually labeled RADAR data.  

---

## 🚀 **Installation & Usage**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/radar-object-classification
cd radar-object-classification
