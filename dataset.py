import torch
import numpy as np
import os
import scipy.io
from PIL import Image
from torchvision.transforms import *
from torch.utils.data import DataLoader

def standardize_label(label, orim):  #保证了原图像的坐标的准确
    label_std = []
    for idx, _ in enumerate(label):
        labelX = label[idx][0] / orim.size[0]  #x的值除于原始图像的宽
        labelY = label[idx][1] / orim.size[1]  #y的值除于原始图像的高
        label_std.append([labelX, labelY])
    label_std = np.array(label_std)
    # print(label_std)
    return label_std


# guassian generation
def getGaussianMap(joint = (16, 16), heat_size = 128, sigma = 2):
    # by default, the function returns a gaussian map with range [0, 1] of typr float32
    heatmap = np.zeros((heat_size, heat_size),dtype=np.float32)
    tmp_size = sigma * 3
    ul = [int(joint[0] - tmp_size), int(joint[1] - tmp_size)]
    br = [int(joint[0] + tmp_size + 1), int(joint[1] + tmp_size + 1)]
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))
    g.shape
    # usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heat_size) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heat_size) - ul[1]
    # image range
    img_x = max(0, ul[0]), min(br[0], heat_size)
    img_y = max(0, ul[1]), min(br[1], heat_size)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    """
    heatmap *= 255
    heatmap = heatmap.astype(np.uint8)
    cv2.imshow("debug", heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return heatmap


class PoseImageDataset(torch.utils.data.Dataset):
    def __init__(self, transforms, imagespath='', labelsfilepath=''):

        imgs_list = sorted(os.listdir(os.path.join(imagespath)))  # 获得文件夹内的图片的名称列表
        self.filenames = imgs_list

        #将注释文件加载到矩阵中
        self.annotationmat = scipy.io.loadmat(labelsfilepath)
        # print(self.annotationmat) #加载.mat文件的数据

        joints = self.annotationmat['joints']
        # print(joints) # 只加载'joints'键的数据
        # print(joints.shape) # (3, 14, 2000)

        joints = np.swapaxes(joints, 2, 0)
        """
        print(joints)
        将0轴和2轴转换，使[ 29.74645941 143.34544031   0.        ]就为第一张图片的x轴和y轴以及二进制的值
        [[[ 29.74645941 143.34544031   0.        ]
          [ 30.5501068  117.22690013   0.        ]
          [ 28.94281202  84.67918082   0.        ]
          ...
        """

        labels = []
        images = []
        heatmap_set = np.zeros((2000, 128, 128, 14), dtype=np.float32)

        origin_image_size = []

        for file_idx, file_name in enumerate(imgs_list):
            fn = imgs_list[file_idx]
            orim = Image.open(os.path.join(imagespath,fn))
            origin_image_size.append(orim.size)
            # print(orim)   # Image.open根据拼接的路径获取图像信息

            # print(self.transforms)
            image1 = transforms(orim)  #将图像信息归一化
            # print(image1.shape)

            label = joints[file_idx]
            # print(label)

            #standardizing标准化
            label = standardize_label(label, orim)
            # print(label)
            for j in range(14):
                _joint = (label[j, 0:2] // 2).astype(np.uint16)
                # print(_joint)
                heatmap_set[file_idx, :, :, j] = getGaussianMap(joint=_joint, heat_size=128, sigma=4)
    
            label = torch.from_numpy(label)  # torch.from_numpy()方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            label1 = label.type(torch.FloatTensor)

            images.append(image1)
            labels.append(label1)

        # print(heatmap_set.size)
        self.images = images
        self.labels = labels
        self.orim_size = origin_image_size

    def __getitem__(self, idx):
        
        return self.images[idx], self.heatmap_set[idx], self.orim_size[idx]
        # return self.images[idx], self.labels[idx], self.orim_size[idx]

    def __len__(self):
        return len(self.filenames)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 优先使用显卡
print(f"using {device}")

image_path = './dataset/lsp/images'
labels_file_path = './dataset/lsp/joints.mat'
image_size = 128
# image_size = 256 # blazepose
# image_size = 196 # deeppose
batch_size = 32
transforms = Compose([
    Resize((image_size,image_size)),
    ToTensor(),  #张量化
    #ToTensor()能够把灰度范围从0-255变换到0-1之间，
    # 而transform.Normalize()则把0-1变换到(-1,1).具体地说，对每个通道而言
    Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


dataset = PoseImageDataset(transforms, image_path, labels_file_path)
# print(dataset.images.shape)  #(2000, 14, 3)

#数据集划分
total = len(dataset)
print(total)