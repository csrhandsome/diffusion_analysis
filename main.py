from diffusion.model.cnn_based import *
from data_analysis.dataset.demo_push_t_dataset import *
from data_analysis.dataset.demo_push_t_image_dataset import *
from data_analysis.load_data.load_matrix_data import *
from data_analysis.create_random_data import *
# from data.global_data import *
from runner.twodim_state_train import *
from runner.twodim_vision_train import *
from runner.threedim_cnn_train import *
from diffusion.model.visionencoder import *
# 神经网络模型通常需要的输入形状是(batch_size, seq_len, dim)
# 程序为算法+数据

def state():
    # threedim_state_and_vision_train_model()
    threedim_state_and_vision_test_model()

def vision():
    twodim_vision_test_model()


if __name__ == "__main__":
    state()
    #vision()


