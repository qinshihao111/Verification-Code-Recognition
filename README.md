# Verification-Code-Recognition
基于卷积神经网络的验证码识别
基于python3.5，tensorflow1.11.0 <br>
read_dataset.py文件随机生成验证码图片，并将图片与标签作为卷积网络的输入，不需要保存在硬盘中，因此可以产生无数张图片 <br>
model.py定义了卷积网络的模型，基本模型为vgg16+bn层，后面接一个全连接层，输出4位验证码的预测结果 <br>
训练模型，直接运行train.py <br>
