# Video-based Face Recognition

Nowadays, face recognition is one of the most actively studied problems in computer vision and biometrics. Es- pecially, video-based face recognition offers a plethora of potential applications in the real-world including visual surveillance, access control, and video content analysis. In this work, a face recognition system secured with a liveness detection model was implemented. The results suggest that the proposed network can accurately detect faces from a video stream, identify whether it is fake or not, and recog- nize the corresponding person.

# Dataset

The dataset contains 3,425 videos of 1,595 different peo- ple. An average of 2.15 videos are available for each sub- ject.The shortest clip duration is 48 frames, the longest clip is 6,070 frames, and the average length of a video clip is 181.3 frames [7]. 

# Methodology

As already mentioned in the previous section, face recognition consists of several tasks, which are compiled into a pipeline. The first and most important task of the pipeline is detecting faces in the given frame. Then, these faces go through a liveness detection model, which discards all faces that are categorized as “fake” (spoofed). In the third step, from remaining faces feature vector is created. At last, these feature vectors are compared with already known persons’ face features to recognize the name of the person.

### Face detection

Face detection is a crucial part of the model. Therefore the most accurate face detector must be implemented. We have chosen CNN based face detection as its performance, in terms of accuracy, is better than other types of detection. However, such performance comes with a cost of longer time to process each frame. Thus, a GPU is required for this task.

### Liveness detection

After detecting faces, we need to make sure that the detected face is real, meaning it is an im- age of the face directly taken from the webcam or security cameras. As face recognition has become one of the most used unlock features on the phones, some security measures must be implemented. That is, say, if someone recorded the face of a certain person and showed this recording to unlock this person’s phone (spoofing).

To detect if the face is real or fake, we again employed CNN model, which consists of two CNN layers. While live- ness detection model is basic CNN, it works fast due to this simplicity. To train this model, we have recorded 3-8 sec- onds videos of ourselves placing them in the “real” folder, then re-recorded these videos but from the screen of the monitor. Obtained videos were placed in a “fake” folder. Later, all faces from these videos are extracted in places in respective folders for training of the model.

### Face Recognition

For face recognition model, we are planning to use a pre-trained model that is a ResNet model.

# References

[1] Chakraborty, S., Das, D. (2014). An overview of face liveness detection. arXiv preprint arXiv:1405.2227.
[2] G. Hu, Y. Yang, D. Yi, J. Kittler, W. Christmas, S. Z. Li, and T. Hospedales (2015), “When face recognition meets with deep learning: an evaluation of convolu- tional neural networks for face recognition,” in ICCV workshops, pp. 142–150.
[3] He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep resid- ual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
[4] K. Simonyan and A. Zisserman (2015). Very deep convolutional networks for large-scale image recog- nition. In ICLR.
[5] S.Yang,P.Luo,C.C.Loy,andX.Tang(2015),“From facial parts responses to face detection: A deep learn- ing approach,” in IEEE International Conference on Computer Vision, pp. 3676-3684.
[6] Wang, M., Deng, W. (2018). Deep face recognition: A survey. arXiv preprint arXiv:1804.06655.
[7] Wolf, L., Hassner, T., Maoz, I. (2011). Face recog- nition in unconstrained videos with matched back- ground similarity. In CVPR 2011 (pp.529-534). IEEE.
[8] X. P. Burgos-Artizzu, P. Perona, and P. Dollar (2013), “Robust face landmark estimation under occlusion,” in IEEE International Conference on Computer Vision, pp. 1513-1520.
[9] X. Zhu, and D. Ramanan (2012), “Face detection, pose estimation, and landmark localization in the wild,” in IEEE Conference on Computer Vision and Pattern Recognition, pp. 2879-2886.
[10] Zhang, K., Zhang, Z., Li, Z., Qiao, Y. (2016). Joint face detection and alignment using multitask cascaded convolutional networks. IEEE Signal Processing Let- ters, 23(10), 1499-1503.
[11] Yilmaz, O. Javed and M. Shah, Object Tracking: A Survey, ACM Computing Surveys, vol. 38, no. 4, June (2006).
[12] S. Li and Z. Zhang, Float Boost Learning and Sta- tistical Face Detection, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 9, pp. 1112–1123, November (2004).
[13] Z. Kalal, J. Matas and K. Mikolajczyk, Pn Learning: Bootstrapping Binary Classifiers by Structural Con- straints, In Proceedings of IEEE Conference on Com- puter Vision and Pattern Recognition, pp. 49–56, April (2010).
[14] S.Oron,A.Bar-Hillel,D.LeviandS.Avidan,Locally Order Less Tracking, In Proceedings of IEEE Confer- ence on Computer Vision and Pattern Recognition, pp. 1940–1947, January (2012).
[15] P. Viola and M. Jones, Rapid Object Detection using a Boosted Cascade of Simple Features, In Proceed- ings of IEEE Conference on Computer Vision and Pat- tern Recognition, vol. 1, no. 2, pp. 511–518, August (2005).
[16] S. Sankaranarayanan, A. Alavi, C. D. Castillo, and R. Chellappa, “Triplet probabilistic embedding for face verification and clustering,” CoRR, vol. abs/1604.05417, 2016.
[17] E. G. Ortiz, A. Wright, and M. Shah, “Face recog- nition in movie trailers via mean sequence sparse representation-based classification,” in CVPR, 2013, pp. 3531–3538.
[18] R. G. Cinbis, J. J. Verbeek, and C. Schmid, “Unsu- pervised metric learning for face identification in tv video,” ICCV, pp. 1559–1566, 2011.
[19] J. Sivic, M. Everingham, and A. Zisserman, “Person spotting: Video shot retrieval for face sets,” in CIVR, 2005.
[20] O. Arandjelovic and A. Zisserman, “Automatic face recognition for film character retrieval in feature- length films,” in CVPR, 2005, pp. 860–867. 19


