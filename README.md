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
### Example result running on Rasperry Pi 3:
```
guest-6oou09@ubuntu-mate:~$ ./benchmark_model --graph=conv_actions_frozen.pb --input_layer="decoded_sample_data:0,decoded_sample_data:1" --input_layer_shape="16000,1:" --input_layer_type="float,int32" --input_layer_values=":16000" --output_layer="labels_softmax:0" --show_run_order=false --show_time=false --show_memory=false --show_summary=true --show_flops=true 
2017-11-27 08:56:25.024559: I tensorflow/tools/benchmark/benchmark_model.cc:426] Graph: [conv_actions_frozen.pb]
2017-11-27 08:56:25.024789: I tensorflow/tools/benchmark/benchmark_model.cc:427] Input layers: [decoded_sample_data:0,decoded_sample_data:1]
2017-11-27 08:56:25.024861: I tensorflow/tools/benchmark/benchmark_model.cc:428] Input shapes: [16000,1:]
2017-11-27 08:56:25.024919: I tensorflow/tools/benchmark/benchmark_model.cc:429] Input types: [float,int32]
2017-11-27 08:56:25.024954: I tensorflow/tools/benchmark/benchmark_model.cc:430] Output layers: [labels_softmax:0]
2017-11-27 08:56:25.025006: I tensorflow/tools/benchmark/benchmark_model.cc:431] Num runs: [1000]
2017-11-27 08:56:25.025037: I tensorflow/tools/benchmark/benchmark_model.cc:432] Inter-inference delay (seconds): [-1.0]
2017-11-27 08:56:25.025067: I tensorflow/tools/benchmark/benchmark_model.cc:433] Inter-benchmark delay (seconds): [-1.0]
2017-11-27 08:56:25.025097: I tensorflow/tools/benchmark/benchmark_model.cc:435] Num threads: [-1]
2017-11-27 08:56:25.025126: I tensorflow/tools/benchmark/benchmark_model.cc:436] Benchmark name: []
2017-11-27 08:56:25.025155: I tensorflow/tools/benchmark/benchmark_model.cc:437] Output prefix: []
2017-11-27 08:56:25.025186: I tensorflow/tools/benchmark/benchmark_model.cc:438] Show sizes: [0]
2017-11-27 08:56:25.025215: I tensorflow/tools/benchmark/benchmark_model.cc:439] Warmup runs: [2]
2017-11-27 08:56:25.025244: I tensorflow/tools/benchmark/benchmark_model.cc:54] Loading TensorFlow.
2017-11-27 08:56:25.025278: I tensorflow/tools/benchmark/benchmark_model.cc:61] Got config, 0 devices
2017-11-27 08:56:25.158104: I tensorflow/tools/benchmark/benchmark_model.cc:291] Running benchmark for max 2 iterations, max -1 seconds without detailed stat logging, with -1s sleep between inferences
2017-11-27 08:56:25.497476: I tensorflow/tools/benchmark/benchmark_model.cc:324] count=2 first=237900 curr=100842 min=100842 max=237900 avg=169371 std=68529

2017-11-27 08:56:25.497686: I tensorflow/tools/benchmark/benchmark_model.cc:291] Running benchmark for max 1000 iterations, max 10 seconds without detailed stat logging, with -1s sleep between inferences
2017-11-27 08:56:35.518009: I tensorflow/tools/benchmark/benchmark_model.cc:324] count=65 first=157668 curr=150971 min=84108 max=186161 avg=154048 std=15025

2017-11-27 08:56:35.518210: I tensorflow/tools/benchmark/benchmark_model.cc:291] Running benchmark for max 1000 iterations, max 10 seconds with detailed stat logging, with -1s sleep between inferences
2017-11-27 08:56:45.691915: I tensorflow/tools/benchmark/benchmark_model.cc:324] count=61 first=143913 curr=150588 min=89136 max=256518 avg=165940 std=28180

2017-11-27 08:56:45.692227: I tensorflow/tools/benchmark/benchmark_model.cc:538] Average inference timings in us: Warmup: 169371, no stats: 154047, with stats: 165939
2017-11-27 08:56:45.692371: I tensorflow/core/util/stat_summarizer.cc:358] Number of nodes executed: 26
2017-11-27 08:56:45.693501: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Summary by node type ==============================
2017-11-27 08:56:45.693624: I tensorflow/core/util/stat_summarizer.cc:468] 	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
2017-11-27 08:56:45.694009: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	        2	   137.228	    83.089%	    83.089%	  1269.760	        2
2017-11-27 08:56:45.694352: I tensorflow/core/util/stat_summarizer.cc:468] 	        AudioSpectrogram	        1	     7.013	     4.246%	    87.335%	   101.772	        1
2017-11-27 08:56:45.694694: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Mfcc	        1	     6.744	     4.083%	    91.419%	    15.840	        1
2017-11-27 08:56:45.695063: I tensorflow/core/util/stat_summarizer.cc:468] 	                  MatMul	        1	     4.269	     2.585%	    94.003%	     0.048	        1
2017-11-27 08:56:45.695425: I tensorflow/core/util/stat_summarizer.cc:468] 	                 MaxPool	        1	     3.933	     2.381%	    96.385%	   256.000	        1
2017-11-27 08:56:45.695774: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	        3	     3.047	     1.845%	    98.230%	     0.000	        3
2017-11-27 08:56:45.696111: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Relu	        2	     1.980	     1.199%	    99.428%	     0.000	        2
2017-11-27 08:56:45.696470: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	        8	     0.671	     0.406%	    99.835%	     0.000	        8
2017-11-27 08:56:45.696730: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Reshape	        2	     0.081	     0.049%	    99.884%	     0.000	        2
2017-11-27 08:56:45.696920: I tensorflow/core/util/stat_summarizer.cc:468] 	                    _Arg	        2	     0.067	     0.041%	    99.924%	     0.000	        2
2017-11-27 08:56:45.697297: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Softmax	        1	     0.062	     0.038%	    99.962%	     0.000	        1
2017-11-27 08:56:45.697480: I tensorflow/core/util/stat_summarizer.cc:468] 	                    NoOp	        1	     0.032	     0.019%	    99.981%	     0.000	        1
2017-11-27 08:56:45.697590: I tensorflow/core/util/stat_summarizer.cc:468] 	                 _Retval	        1	     0.031	     0.019%	   100.000%	     0.000	        1
2017-11-27 08:56:45.697858: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-11-27 08:56:45.697918: I tensorflow/core/util/stat_summarizer.cc:468] Timings (microseconds): count=61 first=143089 curr=149091 min=88276 max=254865 avg=165168 std=27816
2017-11-27 08:56:45.697981: I tensorflow/core/util/stat_summarizer.cc:468] Memory (bytes): count=61 curr=1643420(all same)
2017-11-27 08:56:45.698041: I tensorflow/core/util/stat_summarizer.cc:468] 26 nodes observed
2017-11-27 08:56:45.698095: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-11-27 08:56:45.974773: I tensorflow/tools/benchmark/benchmark_model.cc:573] FLOPs estimate: 410.32M
2017-11-27 08:56:45.974932: I tensorflow/tools/benchmark/benchmark_model.cc:575] FLOPs/second: 2.66B
```



