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
