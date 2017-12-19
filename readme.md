# 骨架分支1.0.0

这个分支是一个用来做扩展的骨架,可以使用命令行执行训练任务,以避免由于notebook打印信息引起的卡死问题.同时也作为执行机构,可以用来指定模型进行预测.

这个分支提供了一个keras的官方例子作为跑通的例子.并将baseline重构为了符合本骨架的形式

## 特性

+ 将预处理,数据增强,特征提取,模型构建,训练流程都模块化了,可以相对简单的组合(1.0.0)
+ 命令行执行训练,预测,预测测试集等操作(1.0.0)

## 结构:

+ `speech_recognizer.libsr`

    这个模块是核心部分

    + `speech_recognizer.libsr.train`

        定义了使用数据训练和使用生成器作为参数训练的训练方法

    + `speech_recognizer.libsr.preprocessing`

        定义了数据预处理的基本组件和组合好的预处理过程,要求预处理只能处理sample_rate,wav

    + `speech_recognizer.libsr.data_augmentation`

        定义了数据增强的基本组件和组合好的处理过程.要求返回的是sample_rate,wav

    + `speech_recognizer.libsr.feature_extract`

        定义了数据特征提取的步骤和组合好的处理过程.要求返回的是最终用于输入模型的数据特征

    + `speech_recognizer.libsr.models`

        用于定义模型的蓝图

+ `speech_recognizer.process`

    用于定义训练过程,也就是如何组合`speech_recognizer.libsr`中定义的方法,其中的`utils.py`中定义了几个实用的对象:

    + `REGIST_PROCESS/REGIST_PERPROCESS/REGIST_FEATURE_EXTRACT`用于保存被注册的训练过程,预处理过程和特征提取过程
    + `tb_cb`预定义好的tensorboard回调函数对象
    + `regist`将过程,特征提取过程绑定在`REGIST_PROCESS/REGIST_PERPROCESS/REGIST_FEATURE_EXTRACT`中的装饰器

    使用该装饰器需要指定预处理过程和特征提取过程作为初始化参数,之后就可以定义训练过程了,如果过程有返回值,那么返回值需要是keras模型,装饰器也会自动将这个模型保存到默认位置

+ `speech_recognizer.command`

    用于定义命令行命令使用什么操作

+ `speech_recognizer.main`

    用于定义命令行命令

+ `speech_recognizer.serialized_models`

    默认的用于保存已经训练好的模型数据

+ `speech_recognizer.conf`

    用于配置这个项目的文件.其中需要定义的有:

    + `TRAIN_DATASET_PATH` 训练数据集
    + `TEST_DATASET_PATH` 测试数据集
    + `MODEL_PATH` 模型序列化后存放位置
    + `LOG_PATH` tensorboard回调函数保存模型训练时log的位置
    + `WANTED_WORDS` 指定的需要的单词
    + `POSSIBLE_LABELS` 指定的合法的标签

+ `speech_recognizer.utils`

    用于定义一些通用的方法,其中包括:
    + `lab_to_vector`将标签转为onehot向量
    + `vector_to_lab`将向量转化为标签

## 使用方式

1. 在`speech_recognizer.libsr.preprocessing.blueprints`中新建一个对应的预处理流程,预处理流程要求输入为一个path,使用的工具可以在`speech_recognizer.libsr.preprocessing`中找了做组合,如果没有想要的可以在对应位置写好方便别人使用.

在`speech_recognizer.libsr.preprocessing.blueprints.utils`中定义了四个通用方法`get_train_data`,`get_test_data`,`train_gen`,`test_gen`,他们是用来生成数据的工具,可以在`speech_recognizer.libsr.preprocessing.blueprints,__init__.py`中使用偏函数工具将其与定义的预处理流程组合以方便使用.

2. 在`speech_recognizer.libsr.models`中新建一个模型结构模块.推荐使用层次结构,比较直观,可以在其`__init__.py`中注册自己定义的结构,如果这样做的话注意不要命名冲突

3. 在`speech_recognizer.process`中新建一个训练流程模块,主要就是使用什么输入,使用什么模型结构这样的东西,注意要使用装饰器`regist(预处理流程函数对象)`装饰下过程,而且要将这个过程注册到`__init__.py`中,这样命令行可以找到这个过程.而且
在最后要使用`.save`将训练好的模型以及使用`json`将标签的对应onehot保存起来,约定保存在`speech_recognizer.serialized_models`下的各自过程名的文件下,模型的命名规则为`过程名+_model.h5`,标签的命名规则为`过程名+_index.json`

4. 在根目录下使用命令行`python main.py train xxxx` 训练流程.

5. 在根目录下使用命令行`python main.py predict xxxx path` 预测某段音频的标签

6. 在根目录下使用命令行`python main.py predict_submit xxxx` 预测测试集数据







