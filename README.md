# DQN_TF
Deep Q Network implemented using Tensorflow

(1) This repo implements basic Deep Q Network using Tensorflow (2.8.0) and applies to the "Pong" Atari game as an example.

(2) This repo is inspired by Maxim Lapan's "Deep Reinformance Learning Hands-on (2nd Edition)" book along with its github repo implemened by Pytorch https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition. The infrastructure on fetching data from Atari is customized from Lapan's code.

(3) This repo is tested on both Nvidia (ubuntu 20.04 + i7 9700 + 32Gb RAM + RTX 2060 + cuda 11.2) and Apple silicon platforms (M1 max 10c CPU + 32c GPU + 64Gb RAM). Some performance insights for M1 max using RTX 2060 as a reference can be gained. 




Performance Comparison (M1 max vs RTX 2060) 

|  | frames per second |
| ------------- | ------------- |
| RTX 2060  |   51.5 f/s |
| M1 Max |  38.0 f/s|

Specifically, the tensorboad training plots can be found below
![tensorboard_plots](https://user-images.githubusercontent.com/6441064/187032536-7b22d528-5c3c-4428-8c7e-8b2693877af8.png)





