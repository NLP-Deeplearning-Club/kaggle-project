## Now: LB 0.69

## Project structure:
```
.
+-- data
|   +-- test            # extracted
|   |   +-- audio       # all test
|   +-- test.7z         # downloaded
|   +-- train           # extracted
|   |   +-- audio       # folder with all train command/file.wav
|   |   +-- LICENSE
|   |   +-- README.md
|   |   +-- testing_list.txt
|   |   +-- validation_list.txt
|   +-- train.7z        # downloaded
+-- readme.md
+-- utils.py            # functions and models
+-- train.py            # main script to train, test and creat submission.csv file
+-- model               # created by train.py, folder for model, checkpoints, logs
```

## Terminal outputs on my computer(GeForce 1060)
```
rk@rk:~/Amy/mycode/TensorFlow Speech Recognition Challenge$ python train.py 
num_classes: 12
(name:id): {'on': 6, 'right': 5, 'off': 7, 'no': 1, 'unknown': 11, 'stop': 8, 'up': 2, 'down': 3, 'go': 9, 'yes': 0, 'silence': 10, 'left': 4}
There are 57929 train and 6798 val samples
trainset example: (9, '6a014b29', './data/train/audio/go/6a014b29_nohash_0.wav')
validset example: (9, 'cc6bae0d', './data/train/audio/go/cc6bae0d_nohash_0.wav')
Start training............
WARNING:tensorflow:uid (from tensorflow.contrib.learn.python.learn.estimators.run_config) is experimental and may change or be removed at any time, and without warning.
WARNING:tensorflow:uid (from tensorflow.contrib.learn.python.learn.estimators.run_config) is experimental and may change or be removed at any time, and without warning.
WARNING:tensorflow:continuous_train_and_eval (from tensorflow.contrib.learn.python.learn.experiment) is experimental and may change or be removed at any time, and without warning.
WARNING:tensorflow:From /home/rk/Amy/mycode/TensorFlow Speech Recognition Challenge/utils.py:152: get_global_step (from tensorflow.contrib.framework.python.ops.variables) is deprecated and will be removed in a future version.
Instructions for updating:
Please switch to tf.train.get_global_step
2017-11-26 02:38:01.626167: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2017-11-26 02:38:01.798197: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-11-26 02:38:01.798744: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
name: GeForce GTX 1060 major: 6 minor: 1 memoryClockRate(GHz): 1.6705
pciBusID: 0000:01:00.0
totalMemory: 5.93GiB freeMemory: 5.54GiB
2017-11-26 02:38:01.798760: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
/usr/lib/python2.7/dist-packages/scipy/io/wavfile.py:221: WavFileWarning: Chunk (non-data) not understood, skipping it.
  WavFileWarning)
2017-11-26 02:39:57.007957: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:40:07.048630: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:42:01.040555: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:42:10.663087: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:44:06.827291: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:44:17.104133: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:46:13.430423: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:46:22.904146: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:48:20.201925: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:48:29.714400: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:50:23.091609: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:50:32.667119: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:52:30.497799: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:52:39.915129: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:54:35.833894: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:54:45.297204: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:56:40.615346: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:56:50.413589: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
2017-11-26 02:58:45.181858: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
Start predicting............
Writing to submission.csv............
0it [00:00, ?it/s]2017-11-26 02:58:54.928663: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1060, pci bus id: 0000:01:00.0, compute capability: 6.1)
158560it [01:43, 1535.94it/s]
```
## Potential Improvements
1. Change the super parameters
2. Change the neural net structure
3. Add data augmentation
4. Change data form(now is just waveform of sound, could try to convert it to frequency domain)
5. Try other ML methods
6. Ensemble Learning
7. Read papers

## For Test on Rasperry Pi 3
### Inputs and outputs:
decoded_sample_data:0, taking a [16000, 1] float tensor as input, representing the audio PCM-encoded data.

decoded_sample_data:1, taking a scalar [] int32 tensor as input, representing the sample rate, which must be the value 16000.

labels_softmax:0, a [12] float tensor representing the probabilities of each class label as an output, from zero to one.

### Requirements:
1. Be runnable as frozen TensorFlow GraphDef files with no additional dependencies beyond TensorFlow 1.4.
2. Run in < 200ms (better ~ 175ms).
3. Size < 5,000,000 bytes.
4. License-compatible with TensorFlow (Apache), and be submittable through Google’s CLA to the TensorFlow project.

### Benchmark Test:
```sh
curl -O https://storage.googleapis.com/download.tensorflow.org/models/speech_commands_v0.01.zip 
unzip speech_commands_v0.01.zip

curl -O https://storage.googleapis.com/download.tensorflow.org/deps/pi/2017_10_07/benchmark_model 
chmod +x benchmark_model 
./benchmark_model --graph=conv_actions_frozen.pb --input_layer="decoded_sample_data:0,decoded_sample_data:1" --input_layer_shape="16000,1:" --input_layer_type="float,int32" --input_layer_values=":16000" --output_layer="labels_softmax:0" --show_run_order=false --show_time=false --show_memory=false --show_summary=true --show_flops=true 
```




