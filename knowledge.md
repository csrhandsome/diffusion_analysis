# view()需要连续的内存，并且不会复制数据 有点像reshape
import torch
x = torch.randn(4, 4)
y = x.view(16)  # 4x4 -> 16
y = x.view(-1)  # 自动计算维度
y = x.view(2, 8) # 4x4 -> 2x8
# 关键特性：
# 1. 要求张量是连续的(contiguous)
# 2. 共享原始数据的内存，修改会影响原数据
# 3. 如果张量不连续会报错
# 检查是否连续
print(x.is_contiguous())  # True or False
# 不连续的情况
x = torch.randn(4, 4)
y = x.transpose(0, 1)  # 转置后内存不连续
# y.view(16)  # 这里会报错



# flatten()更灵活，可以指定维度范围进行展平
x = torch.randn(4, 3, 2)
# 展平所有维度
y = x.flatten()  # 等同于x.flatten(0, -1)
# 只展平指定维度
y = x.flatten(1, 2)  # 只展平后两个维度：4 x 3 x 2 -> 4 x 6
# 关键特性：
# 1. 可以处理不连续的张量
# 2. 可以指定展平的维度范围
# 3. 会返回一个新的连续张量



# 1. 基本对比
def basic_comparison():
    # 创建一个示例张量
    x = torch.randn(2, 3, 4)  # [batch, channel, length]
    print("原始张量形状:", x.shape)  # torch.Size([2, 3, 4])

    # moveaxis: 移动维度位置
    x1 = torch.moveaxis(x, 1, -1)  
    print("moveaxis后:", x1.shape)  # torch.Size([2, 4, 3])

    # view: 重塑维度
    x2 = x.view(2, -1)  
    print("view后:", x2.shape)  # torch.Size([2, 12])

    # flatten: 展平指定维度
    x3 = x.flatten(1)  
    print("flatten后:", x3.shape)  # torch.Size([2, 12])


2024/10/31
在将任务从二维扩展到三维空间时，**action（动作）信息通常需要涵盖更多的维度和控制要素**。具体到你描述的使用夹爪抓取杯子的任务，action 信息可以包括以下几个方面：

1. **夹爪的运动控制**：
   - **位置变化**：在三维空间中，夹爪需要在三个轴（X, Y, Z）上移动，以定位到杯子的位置。
   - **姿态变化**：除了位置，夹爪的朝向（例如俯仰角、偏航角和滚转角）也是重要的，以确保能够正确夹持杯子。
   - **运动速度或加速度**（如果任务需要动态控制）：这可以帮助实现平滑的抓取动作，避免因过快移动而导致杯子滑落或摔落。

2. **夹爪的操作控制**：
   - **开合状态**：控制夹爪的开合程度，以适应不同大小的杯子。这不仅仅是简单的开/关，还可以包括开合的幅度，以实现精细化的抓取。
   - **抓取力度**（如果需要）：有些任务可能需要根据杯子的材质或重量调整抓取力度，以确保抓取稳固但不至于损坏物体。

3. **其他可能的动作参数**（视具体任务需求而定）：
   - **附加的控制参数**：例如，是否需要在抓取过程中施加额外的力或扭矩，以应对杯子的平衡或倾斜。
   - **时间参数**：某些动作可能需要在特定的时间窗口内完成，比如快速抓取或缓慢调整位置。

**综合起来，action 信息应包括**：
- **空间移动指令**：定位夹爪在三维空间中的目标位置和姿态。
- **夹爪操作指令**：控制夹爪的开合状态及其变化。
- **可能的力度或其他控制参数**：根据任务需求，调整抓取的力度或施加的力。

**举例**：
假设你使用的是笛卡尔空间控制，你的action 可以表示为一个六维向量，其中前三个分量表示夹爪在X, Y, Z轴上的位置变化，后三个分量表示夹爪的姿态变化（例如俯仰、偏航、滚转）。此外，可以加上一个额外的参数来表示夹爪的开合状态（如一个连续值从开到关）。

```plaintext
Action = [ΔX, ΔY, ΔZ, ΔRoll, ΔPitch, ΔYaw, Gripper_Closure]
```

这样的表示方式使得机器人能够同时控制夹爪的位置、姿态和抓取动作，从而更灵活地完成三维空间中的抓取任务。

---

**总结**：在三维空间中的夹取任务中，action 信息不仅包括夹爪的开合状态，还应涵盖夹爪在三维空间中的移动和姿态调整。因此，action 通常是位置和姿态指令与夹爪操作指令的组合。

**希望这些信息对你有所帮助！如果有更多具体问题，欢迎继续讨论。**

---

**参考资料**：
- *Robotics: Modelling, Planning and Control* by Bruno Siciliano et al.
- *Reinforcement Learning for Robotics* - various academic papers and tutorials.

---


# 2024.11.1
# 多模态融合？
# 视觉方案是不是要和文本方案分开？
# 加入多模态机制？
# 文本方案中的state和action如何定义？ -> state是iphone的位姿 action是夹爪的开合


# 2024.11.4(组会日)
# 真正的action应该包括了位姿数据加上夹爪的开合数据[posedata,angledata]  那么原有的state应该为?
# 是不是可以加上用扩散模型估计深度？


# 2024.11.5(英语pre日)
# 位姿在2D平面上由以下5个参数描述：位置 (2维)：x: 水平坐标 y: 垂直坐标 姿态 (3维)：θ(旋转角度) sx: x方向的缩放比例 sy: y方向的缩放比例
# however 位姿在3D空间中由以下6个参数描述：位置 (3维):x: x轴坐标 y: y轴坐标 z: z轴坐标 姿态 (3维) - 欧拉角表示 roll: 绕x轴旋转 pitch: 绕y轴旋转 yaw: 绕z轴旋转

# 2024.11.6()
action变成[posedata,angledata,force]
state变成[]
 
# 2024.11.7(万老师中期pre)
很酷的实现类的方法  model类（实现模型的init,forward）-->model policy(实现模型的 conditional_sample,predict_action,compute_loss) --> train_ _workspace(实现模型的train和eval过程)

# 2024.11.8(无事)
你的疑虑非常合理，理解输入和输出数据之间的关系对于设计有效的模型至关重要。让我们深入探讨一下在你的场景中，输入和输出包含重复（或类似）类型的数据是否合理，以及为什么这是一个常见且有效的做法。

1. 输入和输出的数据类型相同，但代表的含义不同

首先，虽然输入和输出可能包含相同类型的数据（如位姿、力、开合角度等），但它们通常代表不同的时间点或不同的角色。在你的案例中：

输入：当前时刻的机器人状态，包括当前的位姿、开合角度、力传感器数据，以及当前的环境感知（例如视频数据）。
输出：下一时刻机器人的目标状态，可能是要达到的位姿、预期的开合角度，以及可能的力反馈。
2. 时间序列预测中的常见做法

在时间序列预测或控制任务中，输入和输出包含相同类型的数据是非常常见的。例如，在机器人控制中，使用当前的状态信息来预测下一时刻的动作或状态：

当前状态（输入）：用于感知和理解当前的环境和自身状态。
未来状态（输出）：模型需要预测的下一步动作或状态，以实现目标任务。
这种设置下，虽然数据类型相同，但在时间维度上是不同的，输入和输出实际上并不重复。

3. 避免数据泄露（Data Leakage）

需要注意的是，模型设计时必须确保输入中不包含关于输出的未来信息，否则会导致数据泄露，模型可能在训练中“作弊”，无法正确地泛化到新数据。然而，在你的情况下，输入只包含当前时刻的信息，而输出是未来需要预测的状态，因而不存在数据泄露的问题。

4. 多模态数据融合的优势

将多种传感器数据（如视觉、位姿、力等）作为输入，可以让模型更全面地理解当前的状态和环境。这对于复杂的任务，如三维空间中的机器人操作，尤其重要。即使输入和输出包含相似的数据类型，模型仍需要学习从当前状态到未来目标状态的映射关系。

5. 具体到你的应用场景

在你的机器人项目中：

输入：
视频数据：提供环境的视觉信息，有助于理解周围环境和物体的位置、形状等信息。
当前位姿数据：机器人当前的空间位置和朝向。
机械手的开合角度：当前机械手的状态，是抓取操作的重要信息。
力传感器数据：当前承受的力，可以反映接触情况或环境阻力等。
输出：
下一步的运动位姿：机器人需要移动到的新位置和朝向。
下一个开合角度：机械手需要调整的开合程度，可能是为了抓取或释放物体。
预期的力反馈：根据任务需要，可能需要控制接触力，以防止损坏物体或保证稳定抓取。
在这个框架下，模型的任务是从当前的多模态感知输入，预测下一步需要执行的动作指令，以完成特定的任务。

6. 类似的实际应用案例

许多机器人学习和控制的研究都采用了类似的输入输出设计。例如：

模仿学习（Imitation Learning）：机器人学习从专家示范中模仿动作，输入是当前的感知和状态，输出是需要执行的动作。
强化学习（Reinforcement Learning）：机器人通过试错学习策略，输入是当前状态，输出是动作策略。
序列预测模型：在自然语言处理或时间序列预测中，模型会使用先前的词或数据点作为输入，预测下一个词或数据点。
7. 总结

重叠的数据类型并不代表重复的数据：关键是数据所处的时间点和所代表的意义不同。
模型需要学习的是从当前状态到未来状态的映射：即使输入和输出包含相同类型的数据，也是为了实现这一目的。
合理性：在遵循时间因果关系和避免数据泄露的前提下，这种设计是合理且有效的。
8. 建议

如果仍然担心数据的重复，可以考虑以下措施：

明确区分时间点：确保输入只包含当前时刻或过去的信息，输出是未来的预测。
输出动作指令而非直接的状态：例如，输出机器人的控制命令（如速度、加速度、关节角度的变化量等），而不是直接输出下一时刻的状态。
使用差分形式：模型输出预测的变化量，而非绝对值，这样可以强调模型预测的是从当前状态到下一状态的差异。



# 2024/11/21
多模态的数据对齐 因为频率的差距不大，因此准备强行对齐 simple_align 后续可能有更好的对齐方式


# 2024/11/22
原来的episodes_ends的数据的维度确实小很多  
类目！！！！ 终于把dataloader跑通了

# 2024/11/23
类目，train,test也跑通了
通过调整输入的dim，可以很好的跑通数据
现在目前来看结果好像不太理想，不知道是不是数据太少了，还有就是可能输入的数据还有很多没有用上。

# 2024/11/24 (分手了。。。。。。。。)
输入：当前时刻的机器人状态，包括当前的位姿、开合角度、力传感器数据，以及当前的环境感知（例如视频数据）。
输出：下一时刻机器人的目标状态，可能是要达到的位姿、预期的开合角度，以及可能的力反馈。

朝着上述目标努力，目前还需要
1. 根据输出的数据来跑通仿真（linux环境下）
2. 添加视频的数据（linux环境下）
3. 添加力的数据（要数据）
4. 添加声音的数据(数据处理)
5. 调整模型结构还有transformer（尝试多种模型）
6. 将RGB的视频数据和深度数据的对齐
# 2024/11/25
帮本科同学写了个读json的函数，可以直接读取json文件，返回字典
完成第4个工作(未完成)

# 2024/11/26
配置linux的环境 pytorch cuda mujoco 全部配置成功 牛逼！！！

# 2024/11/27 (究极休息日)
尝试第六个 
2可能可以了，但是Angle的数据师兄没有采上，所以数据还是对不齐
最终发现timestamp的对齐函数有问题,simple_align

# 2024/12/2
尝试利用开源的xml文件来进行模型的仿真
render（渲染）在程序中通常指的是将数据或场景转换为可视化图像的过程

# 2024/12/4(推胸50kg做组)
今天尝试简单的跑通仿真
在threedim_state_and_vision_test_model()函数中，需要一个评估的方法来暂停扩散的输出，这个方法暂时没写，先通过仿真来看

# 2024/12/6
尝试理解xml,正确的运用模型

# 2024/12/17
用双流网络统一rgb和深度数据
u-net是指
对称的编码器-解码器结构
逐步降采样和上采样
跳跃连接
特征维度先扩张后收缩

Sequential会自动按顺序执行,ModuleList需要手动执行


# 2024/12/21
Generalizable Humanoid Manipulation with  Improved 3D Diffusion Policies 说延伸预测的维度？

# 2024/12/24
可以建立一个dict，不带什么数据的版本对应什么维度对应什么训练，建立这样的关系，方便每次直接进行调用


“DiffusionPolicy在simulation和realworld中部署了相同的任务，证明了任务可以从虚拟迁移到现实。umi运用成熟的diffusionpolicy，通过简单的人类演示，成功的完成了很多现实当中的任务。3D-ViTac通过在机械夹具上面进行创新，在夹具上可视化了点云的数据，并且外置了LiDAR相机，生成了整体在运动中的点云数据。而在RDT-1B当中，首次引入了双臂机械臂和diffusion来协同在现实进行工作“
# 2024/12/27
假设在t=0时刻拍到一张图片
经过0.1秒的观测延迟
经过0.05秒的推理延迟
机器人需要0.15秒的执行延迟
那么实际上t=0时刻的图片对应的动作,最早要在t=0.3秒才能执行(大家都有延迟)
因此需要
预测未来序列：
模型不仅预测当前应该做什么
而是预测一个未来的动作序列
丢弃过时动作：
如果预测序列是 [a1, a2, a3, a4, a5]
而等到实际可以执行时，a1, a2 已经过时
那么直接从 a3 开始执行\

# 2024/12/30
要记得改成pybullet

visualencoder：
3D-vitac：
resnet 但权重改为r3m
diffusion policy & umi：
clip预训练的vit
resnet 但权重改为r3m
resnet
rdt:
clip,
siglip,
t5
# 2025/1/9
用动捕捕捉一组夹爪位姿的数据，拟合出一个函数曲线
Accelerate 自动管理设备（如 GPU、CPU、TPU），用户无需手动指定设备（如 .to(device)）自动处理数据并行（Data Parallelism）和模型并行（Model Parallelism）
自动处理数据并行（Data Parallelism）和模型并行（Model Parallelism）

每次训练或实验被视为一个 Run，Wandb 会记录该 Run 的所有相关信息（如超参数、指标、日志）提供直观的可视化界面

# 2025/1/10
hiddensize是指在模型当中的中间的size

# 2025/1/11
我悟了，为什么在模型内，数据的最后一个格式是统一的hiddensize或者outputsize了：
首先数据进入模型之前就要经过MLP的线性层进行维度的变化，变为统一的格式，在模型中更快的运算。
最后在输出的时候再次使用MLP进行decoder,又可以变回之前的格式。

# 2025/1/12（重要）
数据中的state和action不需要是重复相同的。
二维的推T中，state为通过仿真得出T形块在环境中的真实位置和方向，可以直接从物理引擎获取精确值。
在现实环境当中，需要通过动作捕捉装置来确定，所以这个维度需要根据动捕来待定。
在现实环境中，state还可以通过两个摄像机来确定。 
action可以是机械末端的操作指令，可以通过指令来完成操作。

非常怪异，action的维度和state的序列是一样的。
action这个动作序列可以直接就是位姿，知道了末端执行器的位姿就可以使用机械臂按照这个来了

# 2025/1/13(重要)
进行图像增强，包括颜色抖动和图像损坏，并在输入的本体感知中添加高斯噪声，信噪比（SNR）为40dB。我们还使用GPT-4-Turbo来增强和扩展语言指令

# 2025/1/14
在本文的机器人任务当中，在处理序列时：
q = 机器人需要操作的动作序列
k = ["给我拿一个红色小球", "帮我取一下蓝色方块", "拿住红色三角体"]
v = [红色小球视觉特征, 蓝色方块视觉特征,红色三角体视觉特征]

# 2025/1/17
CFG 通过引入一个“无条件”的生成路径（即不依赖条件的生成），并与条件生成路径结合，动态调整条件信号的影响强度。(但没用)
直接遍历字典是遍历的key,真无语，遍历字典要用item()


# 2025/2/14
https://poe.com/s/gDuATmJeHn1NrMu89Sym unet和transformer的模型特性,以及后续可以用作分析为什么transformer性能更加优异(可变长以及多注意力机制：1.在 RDT 中，x 的序列长度从 T 变为 T+2（拼接 t 和 freq），自注意力机制自动适应这一变化，无需调整模型结构。而unet可能需要重新去调整模型的dimension。2.unet只能将非action的都作为cond而不能直接拼接，然而transformer可以将和actiono有关的state,freq,t全部合成x然后通过强大的注意力机制去预测)https://poe.com/s/iEGy4k2o552C4gv9NjCE

# 2025/2/16
https://poe.com/s/L7xwn6Dk3o08AKNZ5lvo
维度变换
# 2025/2/17
https://poe.com/s/eoIB94OdQ4kcbdD4oOIK 
模型输入是  动作序列和观测特征直接拼接后的结果。input_dim = action_dim + obs_feature_dim，因为每个时间步的动作都与其对应的观测特征拼接，作为模型的输入。
模型需要同时观测到当前噪声动作和对应的环境状态（观测特征），才能更精准地预测噪声。例如，在机器人控制中，机械臂的当前关节角度（动作）需要与摄像头捕捉的环境状态（观测）联合分析，才能推断出下一步的调整方向。
也就是说，必须要input_dim = action_dim + obs_feature_dim，有state和action的输入，模型才能更好的预测

https://poe.com/s/k6Qx8BQtEPHDAdUBRs89
根据视频是全局还是局部，模型的input_dim需要变化。

https://poe.com/s/urvGClKfOks30OAHe0kf
根据是否全局的讨论:
带任务指令的需要全局，反之用局部

# 2025/2/18
https://poe.com/s/P5GxuUnGqtPFz4HDwhkx 可能的动作空间

# 2025/2/19
https://poe.com/s/hjy6vlcK1ZxcQBBEj5dK 在unet中将action_mask作为global_cond及其理由
https://poe.com/s/dpuDkT6biEdi0ITBUjz6 Diffusion_Policy原文当中使用全局变量和不是全局变量的做法

# 2025/2/24
准备重新修改数据，去掉timestamp，直接变成一个二维数组，每一个timestamp就是一个时间戳，没有单位了

# 2025/2/25
{
    "state": (1, STATE_DIM),           # 当前状态
    "actions": (CHUNK_SIZE, STATE_DIM), # 未来动作序列
    "cam_high": (IMG_HISORY_SIZE, H, W, 3) # 历史图像序列
}
### 1. `qpos` 的维度
qpos = qpos / np.array([[1, 1, 1, 1, 1, 1, 4.7908, 1, 1, 1, 1, 1, 1, 4.7888]])
这里的 `np.array()` 只包含了与 `qpos` 中的某些特定维度相对应的值，而不是所有 128 个维度。这是因为在这个特定的上下文中，只有部分维度需要进行归一化。
### 2. 数据集的时间顺序
- **过去的数据集**: 你提到的数据集确实是过去的数据集。在一个 episode 开始时，系统会记录当前的状态（`qpos`），然后在接下来的时间步中记录相应的动作（`target_qpos`）。因此，`target_qpos` 中的动作是基于 `qpos` 的状态而生成的。
- **时间步的关系**: 在强化学习中，通常会将当前状态与未来的动作进行关联，以便模型能够学习如何根据当前状态做出决策。虽然 `target_qpos` 是未来的动作，但它们是基于过去的状态数据生成的。
### 总结

- `qpos` 是当前状态，维度为 (1, STATE\_DIM)。
- `target_qpos` 是未来的动作，维度为 (CHUNK_SIZE, STATE_DIM) 。
- 数据集记录的是过去的状态和动作，模型通过学习这些数据来预测未来的动作。
  

# 2025/2/27
maniskill是一个仿真库
500k step可以训练出一个小模型


# 2025/3/1
在 Table 4 中，虽然没有直接提到“14维”，但通过表格内容可以明确推断出双臂机器人的动作空间维度：
每个手臂有7个关节位置（Joint Positions）和7个关节速度（Joint Velocities）：
右臂关节位置：[0, 10) → 0-9（10个位置，但实际使用7个）
右臂关节速度：[15, 25) → 15-24（10个位置，但实际使用7个）
左臂关节位置：[50, 60) → 50-59（10个位置，但实际使用7个）
左臂关节速度：[65, 75) → 65-74（10个位置，但实际使用7个）
左右手臂的关节位置和速度加起来是14维：
右臂：7个关节位置 + 7个关节速度 = 14维
左臂：7个关节位置 + 7个关节速度 = 14维

# 2025/3/2
这 14 个维度代表的是双臂机器人的关节位置，具体来说：
左臂 7 个维度:
前 6 个维度 (索引 0-5): 左臂的 6 个关节位置
第 7 个维度 (索引 6): 左臂夹爪的开合程度 (4.7908 是归一化系数)
右臂 7 个维度:
接下来的 6 个维度 (索引 7-12): 右臂的 6 个关节位置
最后 1 个维度 (索引 13): 右臂夹爪的开合程度 (4.7888 是归一化系数)
qpos是机械臂实际的，action是夹爪产生的应该让机械臂执行的

# 2025/3/3
我们已经将模型的推理封装到一个名为 RoboticDiffusionTransformerModel 的类中（参见此文件）。你可以调用该类的 step() 方法进行推理。然而，你可能需要根据你的特定机器人重新实现某些部分。你至少需要修改 _format_joint_to_state()（第164行）和 _unformat_action_to_joint()（第196行），以便在机器人的原始动作和 RDT 接受的统一动作向量之间进行转换。你可能还需要指定机器人的控制频率（第49行）。

重要提示：当你将图像输入到 step() 时，记住图像的顺序必须是 [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1}, ext_{t}, right_wrist_{t}, left_wrist_{t}]。

我们在此文件中提供了一个用于在 Mobile ALOHA 上部署的示例硬件代码，以及相应的运行脚本（inference.sh），详细内容如下：

# 2025/3/4
argparse.ArgumentParser() 创建了一个参数解析器,通过 parser.add_argument() 方法添加各种参数定义
可以通过--解析命令行参数

用生产者消费者模式来处理数据
1. 启动producer并等待缓冲区填充完成
conda activate rdt-data
python -m data_analysis.producer --fill_up --n_workers 4
-m会将其作为 __main__ 模块来执行
2. 启动训练（在另一个终端）
conda activate [your-training-env]
source pretrain.sh  # 或者直接运行训练脚本
本地HDF5文件 -> vla_dataset(读取到内存) -> producer(内存缓冲区) -> consumer_dataset(从内存读取) -> hdf5_dataset(从内存中读取)

# 2025/3/12(某人的生日)
EMA=α⋅θ EMA+(1−α)⋅θ model
​θ model是当前模型的参数。
θ EMA是 EMA 模型的参数。
α 是平滑系数（通常接近 1，例如 0.999）。
EMA 模型的作用是 ​减少模型参数的波动，从而可能提高模型的泛化能力和鲁棒性。

mpi4y 
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
conda install -c conda-forge mpi4py

cutlass
要gcc还有cmake预编译然后记录工具路径

# 2025/3/13
本地HDF5文件 -> vla_dataset(读取到内存) -> producer(内存缓冲区) -> hdf5_dataset(从内存读取)

# 2025/3/16
例如，如果有3个形状为 [3, 4] 的张量：
torch.cat 结果可能是 [9, 4] 或 [3, 12]（取决于维度）
torch.stack 结果是 [3, 3, 4]（在第0维创建新维度）
未完成: 在load_video_data函数当中将[num_frames, channels, height, width] 的视频帧张量，需要将其转换为多[img_history_size, H, W, 3]

# 2025/3/17
运行程序之前跑一下来设置加速细节accelerate config
accelerate configuration saved at /home/three/.cache/huggingface/accelerate/default_config.yaml

echo 'export NCCL_SOCKET_IFNAME=wlp4s0' >> ~/.bashrc
source ~/.bashrc

# 2025/3/18
"zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true, 
        "reduce_bucket_size": 1e8  # 从5e8调整到了1e8
        "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },# 在里面添加了这个，将optimizer负载放到cpu 当天又删掉了
    }
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH 在bash里面添加了这个


健身房哥告诉我的关于优化器还有一些东西的知识
https://yuanbao.tencent.com/bot/app/share/chat/E9Tn2VzzQ61t

通过在大量通用数据集上预训练，然后在特定任务数据集上微调，可以实现更好的迁移学习效果
这种预训练+微调的方式是很常见的范式：

- 预训练：使用大量diverse的数据学习通用特征
- 微调：使用特定任务的数据调整模型，使其更适合目标场景

# 2025/3/24
需要一个dataset的stat,是真正的一个dataset,例如一个任务就为一个dataset


n_obs_steps主要用在生成以下几种数据上面：
观察历史序列：
它决定了模型在做决策时会考虑过去多少个时间步的观察数据
例如，当n_obs_steps=2时，模型会使用当前和前一个时间步的观察数据
全局条件向量：
在扩散模型中，n_obs_steps决定了用作全局条件的观察序列长度
计算方式通常是：global_cond = obs[:,:self.n_obs_steps,:].reshape(obs.shape[0], -1)
全局条件向量的维度为：global_cond_dim = obs_feature_dim * n_obs_steps
掩码生成：
在训练过程中，n_obs_steps决定了哪些观察数据被保留为条件
掩码生成器会使用n_obs_steps来确定条件掩码的形状
环境观察缓冲区：
在MultiStepWrapper类中，它决定了环境观察缓冲区的大小
这个缓冲区存储了最近的n_obs_steps个观察，用于提供给策略
特征提取：
在图像策略中，n_obs_steps决定了从图像序列中提取特征的帧数
例如：rgb_features_map[key] = net(nobs[key][:,:self.n_obs_steps])
简而言之，n_obs_steps主要用于生成模型决策所需的历史观察数据，它决定了模型能"看到"多远的过去，从而影响模型对时间依赖性的感知能力。


在RDTRunner中确实没有直接使用n_obs_steps这个参数，但是它有类似的设计理念。让我解释一下两个模型之间的差异和相似之处：



## UNETRunner 与 RDTRunner 的设计差异

### UNETRunner的设计
UNETRunner中的`n_obs_steps`表示模型考虑的历史观测步数，这些历史观测会被拼接成一个长向量作为全局条件：
```python
global_cond_dim = (state_dim + img_dim + lan_dim) * n_obs_steps
```

这种设计允许模型同时考虑多个历史时间步的观测来预测未来动作。

### RDTRunner的相似设计
RDTRunner中虽然没有直接使用`n_obs_steps`参数，但它通过其他方式实现了对历史观测的处理：

1. **图像历史**：通过`img_history_size`参数实现
```python
img_cond_len = (config["common"]["img_history_size"] 
                * config["common"]["num_cameras"] 
                * vision_encoder.num_patches)
```

2. **状态历史处理**：
```python
# 在计算loss函数时
states = batch["states"].to(dtype=weight_dtype) # (B, T, D_a)
# We only use the last state as input
states = states[:, -1:, :]
```
这里虽然只使用了最后一个状态作为输入，但实际上`states`变量是包含了历史状态的张量(B, T, D_a)，RDT选择只使用最新的状态。

3. **语言历史**：通过`max_lang_cond_len`参数控制语言历史的长度。

## 区别与联系

- **数据处理方式**：UNETRunner将所有历史观测拼接到一起作为全局条件；RDTRunner则是通过Transformer架构的注意力机制来处理时序信息。

- **架构差异**：UNETRunner基于CNN架构，而RDTRunner基于Transformer架构，后者天然适合处理序列数据，因此不需要显式地将历史观测拼接成向量。

- **功能等价性**：尽管实现方式不同，两者的目标都是让模型能够利用历史信息来预测未来动作，只是实现机制不同。

总结来说，RDTRunner没有直接使用`n_obs_steps`，而是通过`img_history_size`、状态的时间维度以及Transformer的序列处理能力来实现对历史观测的利用。这种设计更适合Transformer的架构特点，能够更好地捕捉时序依赖关系。

# 2025/3/26
在config.yaml中设置了lang_token_dim: 4096，但这可能是基于不同的文本编码器模型（比如"google/t5-v1_1-xxl"等大型模型）
实际使用的编码器是"google/t5-v1_1-base"，其输出维度是768

# 2025/3/27
unet_runner.py里面
将每个3D条件张量 (B, seq_len, hidden_size) 通过平均池化变为2D (B, hidden_size)
这样每个条件都变成固定大小，不再依赖序列长度，避免了维度爆炸

# 2025/3/31
之后的任务：
1.读取数据的部分的pytorch分离出去
2.实现内存读写(现在内存使用率太低了)

当使用 `python -m data_analysis.producer` 运行时：

1. Python 将 `data_analysis.producer` 作为模块导入并执行
2. 导入模块时，Python 会执行模块中的所有顶层代码
3. 代码中的 `if __name__ == '__main__':` 条件判断会生效

在 `producer.py` 中，`if __name__ == '__main__':` 部分包含了解析命令行参数并启动多进程的逻辑。当通过模块方式执行时：

```python
if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # ...设置各种参数...
    
    # 运行生产者
    args = parser.parse_args()
    # ...启动多进程...
```

这个条件语句的作用是：当模块被直接执行时运行此代码块，但当模块被导入到其他文件时不执行。

所以使用 `-m` 运行模块时，`__name__` 被设置为 `'__main__'`，从而触发主函数的执行。这是 Python 模块系统的标准行为。


预训练的数据流程：
producer.py从多个原始数据集读取数据
将这些数据转换为统一格式并存储在内存缓冲区中
训练时，VLAConsumerDataset从这个内存缓冲区读取数据
微调的数据流程：
数据已经预先处理并存储为HDF5格式
训练时，VLAConsumerDataset通过HDF5VLADataset直接从HDF5文件读取数据
这种模式通过设置参数--load_from_hdf5启用

1. **VLADataset与Producer（预训练/所有数据集）**：
   
   `VLADataset`（定义在`data/vla_dataset.py`中）和`producer.py`是为**预训练**阶段设计的，它们负责处理**多个数据集**的数据。观察`vla_dataset.py`的代码:

   ```python
   # VLADataset从多个数据集中读取
   self.name2dataset = {}
   for dataset_name in self.dataset_names:
       if dataset_name in DATASET_NAMES_NOOPENX:
           dataset = globals()[dataset_name].load_dataset(seed)
       else:
           dataset_path = dataset_to_path(dataset_name, self.openx_dir)
           dataset = tfds.builder_from_directory(builder_dir=dataset_path)
   ```

   从`configs/pretrain_datasets.json`可以看出，预训练使用了大约45个不同的数据集。

2. **HDF5VLADataset（微调/只有agilex）**：

   `HDF5VLADataset`（定义在`data/hdf5_vla_dataset.py`中）是为**微调**阶段设计的，它只处理**agilex数据集**。这可以从代码中看出：

   ```python
   # HDF5VLADataset只读取agilex数据集
   HDF5_DIR = "data/datasets/agilex/rdt_data/"
   self.DATASET_NAME = "agilex"
   ```

   从`configs/finetune_datasets.json`可以看出，微调阶段只使用"agilex"一个数据集。

### 数据流程的不同

1. **预训练的数据流程**：
   - `producer.py`从多个原始数据集读取数据
   - 将这些数据转换为统一格式并存储在内存缓冲区中
   - 训练时，`VLAConsumerDataset`从这个内存缓冲区读取数据

2. **微调的数据流程**：
   - 数据已经预先处理并存储为HDF5格式
   - 训练时，`VLAConsumerDataset`通过`HDF5VLADataset`直接从HDF5文件读取数据
   - 这种模式通过设置参数`--load_from_hdf5`启用

### 为什么采用这种设计？

1. **预训练需要大规模、多样化的数据**：
   - 预训练阶段需要从多个不同的数据集中学习通用表示
   - 使用producer-consumer模式可以高效处理大量数据
   - 内存缓冲区作为一个"数据池"，可以混合不同数据集的样本

2. **微调需要针对性的数据**：
   - 微调阶段只关注目标机器人（agilex）的特定任务
   - HDF5格式提供更高效的数据存取
   - 简化的数据流程更适合针对性训练

3. **效率考虑**：
   - 预训练的producer-consumer模式允许数据预处理与模型训练并行进行
   - 微调阶段使用HDF5直接访问方式更简单，减少了复杂性

### 代码证据

通过以下代码可以看出这种区别：

1. **训练时的条件分支**：
   ```python
   # train/dataset.py中
   if self.use_hdf5:  # 微调模式
       res = self.hdf5_dataset.get_item()
       content = res['meta']
       states = res['state']
       # ...
   else:  # 预训练模式
       (content, _, states, _, actions, _, 
       state_elem_mask, *image_metas, 
       state_std, state_mean, state_norm) = self._safe_load(index)
   ```

2. **微调时的命令行参数**：
   ```bash
   # finetune.sh中
   --dataset_type="finetune" \
   --load_from_hdf5 \
   ```

总结来说，这是一个精心设计的两阶段训练系统：
- 预训练阶段使用生产者-消费者模式处理大量多样化数据集
- 微调阶段使用简化的HDF5直接访问模式专注于目标机器人

这种设计允许模型先从大量数据中学习通用能力，然后再在特定任务上进行精细调整，这正是当代大型模型训练的常用范式。

我想换cnn的话就不能微调了

# 2025/4/1(愚人节)
今天：
实现硬盘互斥读写
改进一个模型,原来的作为baseline

# 2025/4/2
rename 's/\.h5$/.hdf5/' *.h5 将所有的h5改成了hdf5