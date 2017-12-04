# 骨架分支

这个分支是一个用来做扩展的骨架,可以使用命令行执行训练任务,以避免由于notebook打印信息引起的卡死问题.同时也作为执行机构,可以用来指定模型进行预测.

这个分支提供了一个keras的官方例子作为跑通的例子.并将baseline重构为了符合本骨架的形式

## 结构:

+ `speech_recognizer.libsr`

    这个模块是核心部分,在其`__init__.py`中定义了使用model对象和指明过程名字的字符串反序列化后的对象进行预测等操作的方法.

    + `speech_recognizer.libsr.train`

        定义了使用数据训练和使用生成器作为参数训练的训练方法

    + `speech_recognizer.libsr.preprocessing`

        定义了数据预处理的基本组件和组合好的预处理过程,组合好的预处理过程在其`__init__.py`中定义

    + `speech_recognizer.libsr.models`

        用于定义模型的蓝图

+ `speech_recognizer.process`

    用于定义训练过程,也就是如何组合`speech_recognizer.libsr`中定义的方法

+ `speech_recognizer.command`

    用于定义命令行命令使用什么操作

+ `speech_recognizer.main`

    用于定义命令行命令

+ `speech_recognizer.serialized_models`

    用于保存已经训练好的模型

## 训练


### 挑选超参


### 预测
