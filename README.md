# Vision-based-Crowd-Analysis
An analysis on the research paper : Over-crowdedness Alert! Forecasting the Future Crowd Distribution

# reference links : https://arxiv.org/abs/2006.05127

### Authors : Yuzhen Niu Weifeng Shi Wenxi Liu Shengfeng He Jia Pan Antoni B. Chan

### Abstract
In recent years, vision-based crowd analysis has been studied extensively due to its practical applications in real world. In this paper, we formulate a novel crowd analysis
problem, in which we aim to predict the crowd distribution in the near future given sequential frames of a crowd video without any identity annotations. Studying this research
problem will benefit applications concerned with forecasting crowd dynamics. To solve this problem, we propose a global-residual two-stream recurrent network, which leverages the consecutive crowd video frames as inputs and their corresponding density maps as auxiliary information to predict the future crowd distribution. Moreover, to strengthen
the capability of our network, we synthesize scene-specific crowd density maps using simulated data for pretraining. Finally, we demonstrate that our framework is able to predict the crowd distribution for different crowd scenarios and we delve into applications including predicting future crowd count, forecasting high-density region, etc.

### Introduction
In recent years, vision-based crowd analysis has been extensively researched, due to its wide applications in crowd management, traffic control, urban planning, and
surveillance. The recent researches mainly focus on crowd counting [8, 14, 50, 52], multi-target tracking [34, 37], motion pattern analysis [47, 54], holistic crowd evaluation [55], crowd attribute learning [40, 49], and pedestrian path prediction [1, 12] in images or videos.
In real-world scenarios, in order to manage crowd behavior, it is critical to forecast the dynamics of crowd motion to prevent the dangers brought by over-crowded people, such
as crowd crush that may cause people falls or fatalities. Existing research either investigate the previous or current status of the crowd [8, 14, 52], or predict the individual trajectories within a less crowded scene [1, 12]. These methods can hardly be applied in situations to issue an alert for the potential danger of large-scale crowd in advance. On
the other hand, little attention has been paid to predict the dynamics for large crowds holistically in the short-term or long-term futures.

Figure 1. We formulate a novel problem to forecast the crowd
distribution from sparsely sampled previous crowd video frames
without knowing the individual identities. To solve this problem, we propose a prediction model that is able to learn the crowd
dynamics to predict the crowd density in the near future. As illustrated, observing the crowd gathering behavior within the red box,
our model manages to forecast the high-density area (indicated by the yellow region) ahead of time.Hence, in this paper, we formulate a novel yet challenging crowd distribution prediction problem. Given several
sequential frames of a crowd video without any exact position or identity information of the individuals, our goal is to estimate the crowd distribution in the near future (see Fig. 1). To benefit long-term prediction, the provided frames of the crowd video are sampled sequentially yet sparsely over an equal interval (up to 6 seconds), and we aim to predict the crowd distribution of the very next frame in the same interval. Specifically, the reason of sampling input frames over a large interval is that it allows to observe more variations of crowd dynamics and inject contextual information for a longer-term prediction. Compared with tracking and path prediction tasks, the challenges of our problem is the identities or the positions of pedestrians are not provided in the input. Although it mitigates the laborious annotation efforts in real-world application scenarios, the difficulty of prediction is also increase. Besides, instead of predicting trajectories as outputs, we forecast the crowd distribution in the form of future crowd density map, which is informative for analyzing crowd dynamics, monitoring the high-density regions, and even detecting the abnormal crowd behavior. Furthermore, enabling the crowd density prediction without revealing the identities can well preserve the privacy of individuals in certain applications.

To solve the posed challenge, we propose a globalresidual two-stream network to forecast the crowd density
given the input sequential frames of the crowd video. In
the first stream, given the input frames, we adopt a multiscale recurrent network which extracts spatial context feature and leverage a series of convolutional LSTM layers,
or a ConvLSTM block, to correlate the spatial and temporal features. To enhance the prediction, in the second
stream, we set up a recurrent auto-encoder to predict the
future crowd density from the corresponding density maps
of the given frames. Since the sequential density maps can
provide more abstract representation of crowd status and
dynamics, it enables the prediction of crowd dynamics to be
more accurate. Moreover, to further strengthen the capability of the second stream, we simulate diverse crowd behaviors, and thus generate a large amount of synthetic crowd
density maps for pretraining. The computed features will be
jointly passed through an attention-based module to forecast the future crowd density. Finally, to incorporate the
recent motion prior into prediction, we introduce an additional branch that combines the warped density map guided
by flow map with our fused feature so as to improve the
quality of the predicted density map.
In experiments, we adopt the public video-based crowd
counting datasets, UCSD [8] and Mall [10], to evaluate
the crowd density prediction. However, existing crowd
video sequences are often too short to observe the complex dynamics of crowd, or the captured crowd scenes under
limited camera views lead to little variation of crowd density. Therefore, we manually annotate the crowd from an
over 30-min video [56] captured with a large camera view
in the Grand Central Station, New York. For evaluating
the predicted density map, we propose a metric that hierarchically measures the difference of local crowd density
between predictions and ground-truths. Besides, we perform comprehensive experiments to compare our approach
with the optical flow-based methods and video frame prediction approaches.
To sum up, the contributions of our paper are fourfold:
• We formulate a novel problem for predicting crowd
density in the near future, given the past sequential yet
sparsely sampled crowd video frames.
• We propose a global-residual two-stream network
architecture that learns from the crowd videos and the
corresponding density maps separately to forecast the
future crowd density.
• We incorporate different motion priors into the density prediction by simulating diverse synthetic density
maps. It largely enriches the feature representations
and robustness of the network.
• For evaluation, we manually label a long duration and
large scale crowd video and we propose a spatialaware metric for measuring the quality of the predicted
crowd density. Moreover, we delve into several related
crowd analysis applications.
2. Related Works
In this paper, we propose a novel research problem,
future crowd distribution prediction. In this section, we will
survey related aspects of our work, including crowd counting, path prediction, and video frame prediction.
Crowd counting has been studied for years in computer vision [17], whose purpose is to count the number
of people and to estimate how crowd is spatially arranged
in images. Detection- or tracking-based methods [6, 35, 45]
can solve the counting problem, but their performance are
often limited by low-resolution and severe occlusion. In
recent years, regression-based methods have been investigated for counting [9, 14]. Specifically, they directly map
the image features to the number of people, without explicit
object detection, or map local features to crowd blob count
based on segmentation [8]. Besides, the concept of density
map, where the integral (sum) over any sub-region equals
the number of objects in that region, was first proposed
in [19]. The density values are estimated from low-level
features, thus sharing the advantages of general regressionbased methods, while also maintaining location information [3, 19]. With the progress of deep learning techniques,
convolutional neural network (CNN)-based methods have
demonstrated excellent performance on the task of counting dense crowds [7,20,25,30,39,41,50,52]. Most of these
methods first estimate the density map via deep neural networks and then calculate the counts. Unlike prior works
on crowd counting, our work aims at predicting the crowd
spatial distribution in the near future, given the multiple previously observed crowd images.
Trajectory prediction is another related research topic
that learns to forecast the human behavior under complex
social interactions in the term of trajectory [1, 5, 29, 48]. In
these methods, the focus is rested on human-human interaction, which has been investigated for decades in social science, graphics, vision, and robotics [2, 13, 18, 22, 23, 26, 34,
42]. The interaction has been exhaustively addressed by traditional methods based on hand-crafted features [2, 34, 46].
Social awareness in multi-person scenes has been recently
revisited with data-driven techniques based on deep neural
networks [1,12,31,38,48,51,53]. All these methods require
the identities of the persons with their previous positions,
and their studies are mostly evaluated on low-density or
medium-density of crowd motions. Compared with them,
our approach is able to work on large crowd scenes with a
varying density without knowing the identities of individuals in the crowd.
Video frame prediction recently achieves significant
