# mobi-deep
This is a lightweight terrain classifier for automatic Mars rover navigation

This project takes the rover as the research object, aiming at the terrain classification problem in the automatic navigation of the rover, combined with deep learning, adopts the latest Mars terrain AI4MARS semantic data set for training, and develops a lightweight terrain classifier. Based on Deeplabv3+ segmentation model, the classifier replaces the backbone network with the lightweight mobilenetv3 network, and replaces the feature extraction part of the traditional ASPP module with the RFB module based on the visual field, which improves the accuracy while maintaining the recognition speed. And has the advantages of lightweight and high recognition rate. The project's lightweight terrain classifier optimizes the rover's autonomous navigation performance and improves traversability over non-geometric rugged terrain

![image](https://github.com/chensheng1213/Mobile-DeepRFB/assets/134072174/6295aa3c-0a5f-4dfb-9c7b-bb2660e39eca)
![Uploading image.png…]()
