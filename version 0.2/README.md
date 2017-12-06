## LB 0.78

This is a basic speech recognition example. For more information, see the
tutorial at https://www.tensorflow.org/versions/master/tutorials/audio_recognition.

## Run
For training:
```
python train.py
```

For export the graph to a compact format:
```
python freeze.py --start_checkpoint=/media/enroutelab/sdd/mycodes/kaggle-project/logs02/conv.ckpt-18000 --output_file=/media/enroutelab/sdd/mycodes/kaggle-project/logs02/my_frozen_graph.pb
```

For run on the test set and create submition csv:
```
python mysubmission.py  --graph=/media/enroutelab/sdd/mycodes/kaggle-project/logs02/my_frozen_graph.pb --labels=/media/enroutelab/sdd/mycodes/kaggle-project/logs02/conv_labels.txt
```
## Results
```
INFO:tensorflow:Step #17992: rate 0.000100, accuracy 82.0%, cross entropy 0.564173
INFO:tensorflow:Step #17993: rate 0.000100, accuracy 86.0%, cross entropy 0.402874
INFO:tensorflow:Step #17994: rate 0.000100, accuracy 89.0%, cross entropy 0.369776
INFO:tensorflow:Step #17995: rate 0.000100, accuracy 85.0%, cross entropy 0.395649
INFO:tensorflow:Step #17996: rate 0.000100, accuracy 86.0%, cross entropy 0.469917
INFO:tensorflow:Step #17997: rate 0.000100, accuracy 80.0%, cross entropy 0.612350
INFO:tensorflow:Step #17998: rate 0.000100, accuracy 85.0%, cross entropy 0.441275
INFO:tensorflow:Step #17999: rate 0.000100, accuracy 82.0%, cross entropy 0.482268
INFO:tensorflow:Step #18000: rate 0.000100, accuracy 81.0%, cross entropy 0.508963
INFO:tensorflow:Confusion Matrix:
 [[258   0   0   0   0   0   0   0   0   0   0   0]
 [  2 196   3   2   7   7   7  12   8   4   1   9]
 [  3   5 243   3   0   2   4   0   0   0   0   1]
 [  0   6   2 227   4   3   1   2   0   0   2  23]
 [  3   6   0   0 241   0   0   0   0   4   5   1]
 [  0   8   3  16   0 224   0   0   0   0   4   9]
 [  1   4   9   2   1   0 226   4   0   0   0   0]
 [  1   7   0   0   1   0   5 241   0   1   0   0]
 [  4   4   0   0   4   1   0   0 242   1   0   1]
 [  0   4   1   0  18   0   2   0   1 228   1   1]
 [  2   5   0   1   9   0   4   0   0   2 222   1]
 [  7   9   0  15   3   4   1   4   1   2   0 214]]
INFO:tensorflow:Step 18000: Validation accuracy = 89.3% (N=3093)
INFO:tensorflow:Saving to "/media/enroutelab/sdd/mycodes/TensorflowSpeechRecognitionChallenge/logs/conv.ckpt-18000"
INFO:tensorflow:set_size=3081
INFO:tensorflow:Confusion Matrix:
 [[257   0   0   0   0   0   0   0   0   0   0   0]
 [  0 198   5   1   3   5   3  16   9   3   4  10]
 [  1   5 230   5   2   0  10   2   0   0   1   0]
 [  1   8   0 205   2   7   3   1   0   0   3  22]
 [  0   2   0   0 255   0   3   0   1   2   7   2]
 [  2   7   0  13   2 210   1   0   1   0   2  15]
 [  0   4  14   0   4   0 241   3   0   0   1   0]
 [  1   8   0   0   3   0   1 242   1   2   0   1]
 [  0   4   0   0   3   2   1   1 233   2   0   0]
 [  0   2   0   0  20   1   1   3   7 225   3   0]
 [  0   2   1   0   7   1   2   0   0   2 234   0]
 [  0  14   0  27   4   4   5   2   0   0   1 194]]
INFO:tensorflow:Final test accuracy = 88.4% (N=3081)
```
## Result running on Rasperrypi3
```
rk@rk:~/Desktop/benchmark_tensorflow$ ./benchmark_model --graph=my_frozen_graph.pb --input_layer="decoded_sample_data:0,decoded_sample_data:1" --input_layer_shape="16000,1:" --input_layer_type="float,int32" --input_layer_values=":16000" --output_layer="labels_softmax:0" --show_run_order=true --show_time=true --show_memory=true --show_summary=true --show_flops=true
2017-12-06 17:16:13.798424: I tensorflow/tools/benchmark/benchmark_model.cc:426] Graph: [my_frozen_graph.pb]
2017-12-06 17:16:13.798684: I tensorflow/tools/benchmark/benchmark_model.cc:427] Input layers: [decoded_sample_data:0,decoded_sample_data:1]
2017-12-06 17:16:13.798750: I tensorflow/tools/benchmark/benchmark_model.cc:428] Input shapes: [16000,1:]
2017-12-06 17:16:13.798784: I tensorflow/tools/benchmark/benchmark_model.cc:429] Input types: [float,int32]
2017-12-06 17:16:13.798814: I tensorflow/tools/benchmark/benchmark_model.cc:430] Output layers: [labels_softmax:0]
2017-12-06 17:16:13.798865: I tensorflow/tools/benchmark/benchmark_model.cc:431] Num runs: [1000]
2017-12-06 17:16:13.798898: I tensorflow/tools/benchmark/benchmark_model.cc:432] Inter-inference delay (seconds): [-1.0]
2017-12-06 17:16:13.798927: I tensorflow/tools/benchmark/benchmark_model.cc:433] Inter-benchmark delay (seconds): [-1.0]
2017-12-06 17:16:13.798957: I tensorflow/tools/benchmark/benchmark_model.cc:435] Num threads: [-1]
2017-12-06 17:16:13.798985: I tensorflow/tools/benchmark/benchmark_model.cc:436] Benchmark name: []
2017-12-06 17:16:13.799013: I tensorflow/tools/benchmark/benchmark_model.cc:437] Output prefix: []
2017-12-06 17:16:13.799042: I tensorflow/tools/benchmark/benchmark_model.cc:438] Show sizes: [0]
2017-12-06 17:16:13.799071: I tensorflow/tools/benchmark/benchmark_model.cc:439] Warmup runs: [2]
2017-12-06 17:16:13.799098: I tensorflow/tools/benchmark/benchmark_model.cc:54] Loading TensorFlow.
2017-12-06 17:16:13.799133: I tensorflow/tools/benchmark/benchmark_model.cc:61] Got config, 0 devices
2017-12-06 17:16:13.926709: I tensorflow/tools/benchmark/benchmark_model.cc:291] Running benchmark for max 2 iterations, max -1 seconds without detailed stat logging, with -1s sleep between inferences
2017-12-06 17:16:14.701491: I tensorflow/tools/benchmark/benchmark_model.cc:324] count=2 first=449059 curr=324822 min=324822 max=449059 avg=386940 std=62118

2017-12-06 17:16:14.702007: I tensorflow/tools/benchmark/benchmark_model.cc:291] Running benchmark for max 1000 iterations, max 10 seconds without detailed stat logging, with -1s sleep between inferences
2017-12-06 17:16:24.731586: I tensorflow/tools/benchmark/benchmark_model.cc:324] count=31 first=397353 curr=312297 min=195528 max=400049 avg=323375 std=47537

2017-12-06 17:16:24.731926: I tensorflow/tools/benchmark/benchmark_model.cc:291] Running benchmark for max 1000 iterations, max 10 seconds with detailed stat logging, with -1s sleep between inferences
2017-12-06 17:16:35.078006: I tensorflow/tools/benchmark/benchmark_model.cc:324] count=31 first=340414 curr=316272 min=225089 max=395537 avg=332730 std=32797

2017-12-06 17:16:35.079017: I tensorflow/tools/benchmark/benchmark_model.cc:538] Average inference timings in us: Warmup: 386940, no stats: 323374, with stats: 332730
2017-12-06 17:16:35.085375: I tensorflow/core/util/stat_summarizer.cc:358] Number of nodes executed: 28
2017-12-06 17:16:35.087927: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Run Order ==============================
2017-12-06 17:16:35.096968: I tensorflow/core/util/stat_summarizer.cc:468] 	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
2017-12-06 17:16:35.097167: I tensorflow/core/util/stat_summarizer.cc:468] 	                    NoOp	    0.000	    0.053	    0.067	  0.020%	  0.020%	     0.000	        1	_SOURCE
2017-12-06 17:16:35.097272: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.253	    0.049	    0.058	  0.018%	  0.038%	     0.000	        1	Reshape/shape
2017-12-06 17:16:35.097372: I tensorflow/core/util/stat_summarizer.cc:468] 	                    _Arg	    1.238	    0.034	    0.364	  0.110%	  0.148%	     0.000	        1	_arg_decoded_sample_data_1_1
2017-12-06 17:16:35.097462: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.329	    0.026	    0.026	  0.008%	  0.156%	     0.000	        1	Reshape_1/shape
2017-12-06 17:16:35.097557: I tensorflow/core/util/stat_summarizer.cc:468] 	                    _Arg	    0.652	    0.027	    0.041	  0.012%	  0.168%	     0.000	        1	_arg_decoded_sample_data_0_0
2017-12-06 17:16:35.097641: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.371	    0.025	    0.036	  0.011%	  0.179%	     0.000	        1	Reshape_2/shape
2017-12-06 17:16:35.097732: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.420	    0.026	    0.034	  0.010%	  0.189%	     0.000	        1	Variable_5/read/_0__cf__0
2017-12-06 17:16:35.097818: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.465	    0.025	    0.025	  0.007%	  0.197%	     0.000	        1	Variable_4/read/_1__cf__1
2017-12-06 17:16:35.097908: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.500	    0.023	    0.025	  0.007%	  0.204%	     0.000	        1	Variable_3/read/_2__cf__2
2017-12-06 17:16:35.097996: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.536	    0.024	    0.025	  0.008%	  0.212%	     0.000	        1	Variable_2/read/_3__cf__3
2017-12-06 17:16:35.098082: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    3.014	    0.021	    0.025	  0.008%	  0.220%	     0.000	        1	Variable_1/read/_4__cf__4
2017-12-06 17:16:35.098167: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    3.050	    0.023	    0.027	  0.008%	  0.228%	     0.000	        1	Variable/read/_5__cf__5
2017-12-06 17:16:35.098264: I tensorflow/core/util/stat_summarizer.cc:468] 	        AudioSpectrogram	    0.718	   10.060	    8.973	  2.717%	  2.945%	   100.744	        1	AudioSpectrogram
2017-12-06 17:16:35.098352: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Mfcc	    9.768	    7.135	    8.528	  2.582%	  5.527%	    15.680	        1	Mfcc
2017-12-06 17:16:35.098441: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Reshape	   18.406	    0.052	    0.070	  0.021%	  5.548%	     0.000	        1	Reshape
2017-12-06 17:16:35.098538: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Reshape	   18.496	    0.031	    0.035	  0.011%	  5.559%	     0.000	        1	Reshape_1
2017-12-06 17:16:35.098620: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	   18.550	   78.366	   65.429	 19.814%	 25.373%	  1003.520	        1	Conv2D
2017-12-06 17:16:35.098700: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	   84.092	    6.586	    5.779	  1.750%	 27.123%	     0.000	        1	add
2017-12-06 17:16:35.098779: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Relu	   89.925	    3.557	    2.545	  0.771%	 27.894%	     0.000	        1	Relu
2017-12-06 17:16:35.098868: I tensorflow/core/util/stat_summarizer.cc:468] 	                 MaxPool	   92.525	    3.023	    4.703	  1.424%	 29.318%	   250.880	        1	MaxPool
2017-12-06 17:16:35.098964: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	   97.337	  219.400	  223.502	 67.683%	 97.001%	   250.880	        1	Conv2D_1
2017-12-06 17:16:35.099059: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	  320.940	    1.903	    1.744	  0.528%	 97.529%	     0.000	        1	add_1
2017-12-06 17:16:35.099154: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Relu	  322.732	    0.452	    0.932	  0.282%	 97.811%	     0.000	        1	Relu_1
2017-12-06 17:16:35.099254: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Reshape	  323.694	    0.048	    0.391	  0.118%	 97.929%	     0.000	        1	Reshape_2
2017-12-06 17:16:35.099353: I tensorflow/core/util/stat_summarizer.cc:468] 	                  MatMul	  324.106	    6.669	    6.634	  2.009%	 99.938%	     0.048	        1	MatMul
2017-12-06 17:16:35.099462: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	  330.796	    0.100	    0.092	  0.028%	 99.966%	     0.000	        1	add_2
2017-12-06 17:16:35.099562: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Softmax	  330.907	    0.080	    0.073	  0.022%	 99.988%	     0.000	        1	labels_softmax
2017-12-06 17:16:35.099675: I tensorflow/core/util/stat_summarizer.cc:468] 	                 _Retval	  330.997	    0.047	    0.039	  0.012%	100.000%	     0.000	        1	_retval_labels_softmax_0_0
2017-12-06 17:16:35.099778: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-12-06 17:16:35.099836: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Top by Computation Time ==============================
2017-12-06 17:16:35.099900: I tensorflow/core/util/stat_summarizer.cc:468] 	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
2017-12-06 17:16:35.100004: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	   97.337	  219.400	  223.502	 67.683%	 67.683%	   250.880	        1	Conv2D_1
2017-12-06 17:16:35.100116: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	   18.550	   78.366	   65.429	 19.814%	 87.497%	  1003.520	        1	Conv2D
2017-12-06 17:16:35.100215: I tensorflow/core/util/stat_summarizer.cc:468] 	        AudioSpectrogram	    0.718	   10.060	    8.973	  2.717%	 90.214%	   100.744	        1	AudioSpectrogram
2017-12-06 17:16:35.100325: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Mfcc	    9.768	    7.135	    8.528	  2.582%	 92.796%	    15.680	        1	Mfcc
2017-12-06 17:16:35.100418: I tensorflow/core/util/stat_summarizer.cc:468] 	                  MatMul	  324.106	    6.669	    6.634	  2.009%	 94.805%	     0.048	        1	MatMul
2017-12-06 17:16:35.100511: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	   84.092	    6.586	    5.779	  1.750%	 96.555%	     0.000	        1	add
2017-12-06 17:16:35.100606: I tensorflow/core/util/stat_summarizer.cc:468] 	                 MaxPool	   92.525	    3.023	    4.703	  1.424%	 97.979%	   250.880	        1	MaxPool
2017-12-06 17:16:35.100709: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Relu	   89.925	    3.557	    2.545	  0.771%	 98.750%	     0.000	        1	Relu
2017-12-06 17:16:35.100816: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	  320.940	    1.903	    1.744	  0.528%	 99.278%	     0.000	        1	add_1
2017-12-06 17:16:35.100914: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Relu	  322.732	    0.452	    0.932	  0.282%	 99.560%	     0.000	        1	Relu_1
2017-12-06 17:16:35.101020: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-12-06 17:16:35.101076: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Top by Memory Use ==============================
2017-12-06 17:16:35.101137: I tensorflow/core/util/stat_summarizer.cc:468] 	             [node type]	  [start]	  [first]	 [avg ms]	     [%]	  [cdf%]	  [mem KB]	[times called]	[Name]
2017-12-06 17:16:35.101244: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	   18.550	   78.366	   65.429	 19.814%	 19.814%	  1003.520	        1	Conv2D
2017-12-06 17:16:35.101350: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	   97.337	  219.400	  223.502	 67.683%	 87.497%	   250.880	        1	Conv2D_1
2017-12-06 17:16:35.101441: I tensorflow/core/util/stat_summarizer.cc:468] 	                 MaxPool	   92.525	    3.023	    4.703	  1.424%	 88.921%	   250.880	        1	MaxPool
2017-12-06 17:16:35.101540: I tensorflow/core/util/stat_summarizer.cc:468] 	        AudioSpectrogram	    0.718	   10.060	    8.973	  2.717%	 91.638%	   100.744	        1	AudioSpectrogram
2017-12-06 17:16:35.101651: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Mfcc	    9.768	    7.135	    8.528	  2.582%	 94.220%	    15.680	        1	Mfcc
2017-12-06 17:16:35.101755: I tensorflow/core/util/stat_summarizer.cc:468] 	                  MatMul	  324.106	    6.669	    6.634	  2.009%	 96.229%	     0.048	        1	MatMul
2017-12-06 17:16:35.101863: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    3.050	    0.023	    0.027	  0.008%	 96.238%	     0.000	        1	Variable/read/_5__cf__5
2017-12-06 17:16:35.101974: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	    2.371	    0.025	    0.036	  0.011%	 96.248%	     0.000	        1	Reshape_2/shape
2017-12-06 17:16:35.102073: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	   84.092	    6.586	    5.779	  1.750%	 97.998%	     0.000	        1	add
2017-12-06 17:16:35.102180: I tensorflow/core/util/stat_summarizer.cc:468] 	                 _Retval	  330.997	    0.047	    0.039	  0.012%	 98.010%	     0.000	        1	_retval_labels_softmax_0_0
2017-12-06 17:16:35.102282: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-12-06 17:16:35.102337: I tensorflow/core/util/stat_summarizer.cc:468] ============================== Summary by node type ==============================
2017-12-06 17:16:35.102416: I tensorflow/core/util/stat_summarizer.cc:468] 	             [Node type]	  [count]	  [avg ms]	    [avg %]	    [cdf %]	  [mem KB]	[times called]
2017-12-06 17:16:35.102510: I tensorflow/core/util/stat_summarizer.cc:468] 	                  Conv2D	        2	   288.931	    87.500%	    87.500%	  1254.400	        2
2017-12-06 17:16:35.102607: I tensorflow/core/util/stat_summarizer.cc:468] 	        AudioSpectrogram	        1	     8.972	     2.717%	    90.217%	   100.744	        1
2017-12-06 17:16:35.102703: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Mfcc	        1	     8.527	     2.582%	    92.800%	    15.680	        1
2017-12-06 17:16:35.102798: I tensorflow/core/util/stat_summarizer.cc:468] 	                     Add	        3	     7.612	     2.305%	    95.105%	     0.000	        3
2017-12-06 17:16:35.102888: I tensorflow/core/util/stat_summarizer.cc:468] 	                  MatMul	        1	     6.633	     2.009%	    97.114%	     0.048	        1
2017-12-06 17:16:35.102982: I tensorflow/core/util/stat_summarizer.cc:468] 	                 MaxPool	        1	     4.702	     1.424%	    98.538%	   250.880	        1
2017-12-06 17:16:35.103071: I tensorflow/core/util/stat_summarizer.cc:468] 	                    Relu	        2	     3.476	     1.053%	    99.590%	     0.000	        2
2017-12-06 17:16:35.103158: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Reshape	        3	     0.494	     0.150%	    99.740%	     0.000	        3
2017-12-06 17:16:35.103249: I tensorflow/core/util/stat_summarizer.cc:468] 	                    _Arg	        2	     0.403	     0.122%	    99.862%	     0.000	        2
2017-12-06 17:16:35.103342: I tensorflow/core/util/stat_summarizer.cc:468] 	                   Const	        9	     0.278	     0.084%	    99.946%	     0.000	        9
2017-12-06 17:16:35.103431: I tensorflow/core/util/stat_summarizer.cc:468] 	                 Softmax	        1	     0.073	     0.022%	    99.968%	     0.000	        1
2017-12-06 17:16:35.103523: I tensorflow/core/util/stat_summarizer.cc:468] 	                    NoOp	        1	     0.066	     0.020%	    99.988%	     0.000	        1
2017-12-06 17:16:35.103607: I tensorflow/core/util/stat_summarizer.cc:468] 	                 _Retval	        1	     0.039	     0.012%	   100.000%	     0.000	        1
2017-12-06 17:16:35.103683: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-12-06 17:16:35.103738: I tensorflow/core/util/stat_summarizer.cc:468] Timings (microseconds): count=31 first=337865 curr=314042 min=222568 max=396605 avg=330219 std=33490
2017-12-06 17:16:35.103800: I tensorflow/core/util/stat_summarizer.cc:468] Memory (bytes): count=31 curr=1621752(all same)
2017-12-06 17:16:35.103860: I tensorflow/core/util/stat_summarizer.cc:468] 28 nodes observed
2017-12-06 17:16:35.103915: I tensorflow/core/util/stat_summarizer.cc:468] 
2017-12-06 17:16:35.698804: I tensorflow/tools/benchmark/benchmark_model.cc:573] FLOPs estimate: 402.91M
2017-12-06 17:16:35.698999: I tensorflow/tools/benchmark/benchmark_model.cc:575] FLOPs/second: 1.25B
```
