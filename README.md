# Vision-based-Crowd-Analysis
An analysis on the research paper : Over-crowdedness Alert! Forecasting the Future Crowd Distribution

# reference links : https://arxiv.org/abs/2006.05127

### Authors : Yuzhen Niu Weifeng Shi Wenxi Liu Shengfeng He Jia Pan Antoni B. Chan

### Abstract
In recent years, vision-based crowd analysis has been studied extensively due to its practical applications in real world. In this paper, we formulate a novel crowd analysis
problem, in which we aim to predict the crowd distribution in the near future given sequential frames of a crowd video without any identity annotations. Studying this research
problem will benefit applications concerned with forecasting crowd dynamics. To solve this problem, we propose a global-residual two-stream recurrent network, which leverages the consecutive crowd video frames as inputs and their corresponding density maps as auxiliary information to predict the future crowd distribution. Moreover, to strengthen
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
our model manages to forecast the high-density area (indicated by
the yellow region) ahead of time.
Hence, in this paper, we formulate a novel yet challenging crowd distribution prediction problem. Given several
sequential frames of a crowd video without any exact position or identity information of the individuals, our goal is
to estimate the crowd distribution in the near future (see
Fig. 1). To benefit long-term prediction, the provided
frames of the crowd video are sampled sequentially yet
sparsely over an equal interval (up to 6 seconds), and we
aim to predict the crowd distribution of the very next frame
in the same interval. Specifically, the reason of sampling
input frames over a large interval is that it allows to observe
more variations of crowd dynamics and inject contextual
information for a longer-term prediction. Compared with
tracking and path prediction tasks, the challenges of our
problem is the identities or the positions of pedestrians are
not provided in the input. Although it mitigates the laborious annotation efforts in real-world application scenarios,
the difficulty of prediction is also increase. Besides, instead
of predicting trajectories as outputs, we forecast the crowd
distribution in the form of future crowd density map, which
is informative for analyzing crowd dynamics, monitoring
the high-density regions, and even detecting the abnormal
crowd behavior. Furthermore, enabling the crowd density
prediction without revealing the identities can well preserve
the privacy of individuals in certain applications.

