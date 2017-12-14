"""示范用的测试训练过程,注意,要使用预处理过程函数来初始化装饰器,这样才能在命令行中显示.

这个过程的执行细节是是:

+ 预处理训练数据使用生成器生成
+ 使用默认Adam做为优化器
+ 使用categorical_crossentropy作为损失函数
+ 训练10个epoch,每个epoch训练的batchsize为140
+ 验证数据使用生成器生成
+ 验证集每个batch生成60个数据
"""
from pathlib import Path
from keras.optimizers import Adam
from speech_recognizer.libsr.preprocessing import (
    log_spec_perprocess
)
from speech_recognizer.libsr.data_gen import TrainData
from speech_recognizer.libsr.models import build_basecnn_model
from speech_recognizer.libsr.train import train_generator
from .utils import regist, get_current_function_name, tb_callback


@regist(log_spec_perprocess)
def cnn_spec_gen_process(
        model_kwargs=dict(input_shape=(99, 81, 1),
                          cnn_layer1={"filters": 8, "kernel_size": 2,
                                      "activation": "relu"},
                          pool_layer1={"pool_size": (2, 2)},
                          dropout_layer1={"rate": 0.2},
                          cnn_layer2={"filters": 16, "kernel_size": 3,
                                      "activation": "relu"},
                          pool_layer2={"pool_size": (2, 2)},
                          dropout_layer2={"rate": 0.2},
                          cnn_layer3={"filters": 32, "kernel_size": 3,
                                      "activation": "relu"},
                          pool_layer3={"pool_size": (2, 2)},
                          dropout_layer3={"rate": 0.2},
                          mlp_layer1={"units": 128,
                                      "activation": "relu"}),
        optimizer='adam', loss='categorical_crossentropy',  # 训练用的参数
        metrics=['mae', 'accuracy'], train_batch_size=140,
        validation_batch_size=60, epochs=10):
    
    p = Path(__file__).absolute()
    _dir = p.parent.parent
    func_name = get_current_function_name()
    index_path = _dir.joinpath(
        "serialized_models/" + func_name + "_index.json")

    path = _dir.joinpath(
        "serialized_models/" + func_name + "_model.h5")

    data = TrainData(log_spec_perprocess, index_path=index_path)
    #print(data.test_data[0].shape)
    #print(data.test_data[1].shape)

    train_gen = data.train_gen(train_batch_size)
    lenght = next(train_gen)
    validation_gen = data.validation_gen(validation_batch_size)
    steps = next(validation_gen)
    trained_model = train_generator(build_basecnn_model(**model_kwargs),
                                    train_gen,
                                    steps_per_epoch=lenght,
                                    epochs=epochs,
                                    optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics,
                                    validation_data=validation_gen,
                                    validation_steps=steps,
                                    callbacks=[tb_callback(func_name)]
                                    )
    trained_model.save(str(path))
    print("model save done!")
