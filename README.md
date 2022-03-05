# Deep Learning Forecasts Sleep Apnea from Single-lead ECG

In this project, we implemented convolutional neural network (CNN), long short-term memory (LSTM) and artificial neural network (ANN) for forecasting sleep
apnea events before occurrence. 

Here, we used PhysioNet ECG Database which is available in: https://physionet.org/content/apnea-ecg/1.0.0/ 


Firstly, we extracted R-R Intervals and R-peak amplitude and then fed to ANN and deep learning methods.

During this project, we used different slow time for sleep apnea forecasting in 1-5 min in the future.

CNN-5, LSTM-5: Slow time = 5min

CNN-10, LSTM-10: Slow time = 10 min

CNN-20, LSTM-20: Slow time = 20 min

We used some special pre-processing for this project: 1- labeling data based on our target, 2- Based on slow time we removed begining intervals for example when we used 5-min for slow time, we should remove ECG intervals 1-5 and so on. 3- Based on time of forecasting we removed ending intervals, for example when we predicted 5-min ahead, we should remove 5-min of ending ECG intervals and so on.

# Papers:

If this work is helpful in your research, please consider starring ⭐ us and citing our paper:


2- Bahrami, Mahsa, and Mohamad Forouzanfar. "Detection of sleep apnea from single-lead ECG: Comparison of deep learning algorithms." In 2021 IEEE International Symposium on Medical Measurements and Applications (MeMeA), pp. 1-5. IEEE, 2021.

This paper can be downloded at: https://ieeexplore.ieee.org/abstract/document/9478745

3- Bahrami, Mahsa, and Mohamad Forouzanfar. "Sleep Apnea Detection from Single-Lead ECG: A Comprehensive Analysis of Machine Learning and Deep Learning Algorithms." IEEE Transactions on Instrumentation and Measurement (2022).

This paper can be downloded at: https://ieeexplore.ieee.org/abstract/document/9714370

For more information about codes of paper 2 and paper 3 please refer to: https://github.com/mahsaabahrami/Sleep-Apnea



# Requirements:

1-numpy

2-keras

3-tensorflow

4-scipy


# References:

1- keras for deep learning: https://keras.io/  

