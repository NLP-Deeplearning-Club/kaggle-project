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
from speech_recognizer.libsr.data_gen import (
    log_spec_train_gen,
    log_spec_validation_gen
)
from speech_recognizer.libsr.models import build_basecnn_model
from speech_recognizer.libsr.train import train_generator
from .utils import regist, get_current_function_name, tb_callback


@regist(log_spec_perprocess)
def cnn_spec_gen_process(batch_size=140, epochs=10):
    p = Path(__file__).absolute()
    _dir = p.parent.parent
    func_name = get_current_function_name()
    index_path = _dir.joinpath(
        "serialized_models/" + func_name + "_index.json")
    path = _dir.joinpath(
        "serialized_models/" + func_name + "_model.h5")

    train_gen = log_spec_train_gen(batch_size, index_path)
    lenght = next(train_gen)
    validation_gen = log_spec_validation_gen(60, index_path)
    steps = next(validation_gen)
    trained_model = train_generator(build_basecnn_model(),
                                    train_gen,
                                    steps_per_epoch=lenght,
                                    epochs=epochs,
                                    optimizer=Adam(),
                                    loss='categorical_crossentropy',
                                    metrics=['mae', 'accuracy'],
                                    validation_data=validation_gen,
                                    validation_steps=steps,
                                    callbacks=[tb_callback(func_name)]
                                    )
    trained_model.save(str(path))
    print("model save done!")
