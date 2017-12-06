# 骨架分支

这个分支是一个用来做扩展的骨架,可以使用命令行执行训练任务,以避免由于notebook打印信息引起的卡死问题.同时也作为执行机构,可以用来指定模型进行预测.

这个分支提供了一个keras的官方例子作为跑通的例子.并将baseline重构为了符合本骨架的形式

## 结构:

+ `speech_recognizer.libsr`

    这个模块是核心部分,在其`__init__.py`中定义了使用model对象和指明过程名字的字符串反序列化后的对象进行预测等操作的方法.

    + `speech_recognizer.libsr.train`

        定义了使用数据训练和使用生成器作为参数训练的训练方法

    + `speech_recognizer.libsr.preprocessing`

        定义了数据预处理的基本组件和组合好的预处理过程,组合好的预处理过程在其子模块`blueprints`中定义

    + `speech_recognizer.libsr.models`

        用于定义模型的蓝图

+ `speech_recognizer.process`

    用于定义训练过程,也就是如何组合`speech_recognizer.libsr`中定义的方法

+ `speech_recognizer.command`

    用于定义命令行命令使用什么操作

+ `speech_recognizer.main`

    用于定义命令行命令

+ `speech_recognizer.serialized_models`

    用于保存已经训练好的模型数据和标签onehot对应数据

## 使用方式

1. 在`speech_recognizer.libsr.preprocessing.blueprints`中新建一个对应的预处理流程,使用的工具可以在`speech_recognizer.libsr.preprocessing`中找了做组合,如果没有想要的可以在对应位置写好方便别人使用.推荐参考`log_specgram_gen.py`的写法使用生成器按batch喂数据到显存以减轻机器负担.需要注意的是需要定义一个以`过程名+perprocess`为名的方法用于提取音频中的信息.这个是为了预测时好用,这个函数需要使用装饰器`regist`装饰,并注册到`__init__.py`中

2. 在`speech_recognizer.libsr.models`中新建一个模型结构模块.推荐使用层次结构,比较直观,可以在其`__init__.py`中注册自己定义的结构,如果这样做的话注意不要命名冲突

3. 在`speech_recognizer.process`中新建一个训练流程模块,主要就是使用什么输入,使用什么模型结构这样的东西,注意要使用装饰器`regist`装饰下过程,而且要将这个过程注册到`__init__.py`中,这样命令行可以找到这个过程.而且
在最后要使用`.save`将训练好的模型以及使用`json`将标签的对应onehot保存起来,约定保存在`speech_recognizer.serialized_models`下的各自过程名的文件下,模型的命名规则为`过程名+_model.h5`,标签的命名规则为`过程名+_index.json`

4. 在根目录下使用命令行`python main.py train xxxx` 训练流程.

5. 在根目录下使用命令行`python main.py predict xxxx prexxxx path` 预测某段音频的标签

6. 在根目录下使用命令行`python main.py predict_submit xxxx prexxxx` 预测测试集数据

### 挑选超参

未实现后续可加




