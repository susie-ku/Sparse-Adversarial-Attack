from main_universal import main
import pickle
from PIL import Image
import numpy as np
from tqdm import tqdm

#parsing input parameters
from flags import parse_handle
parser = parse_handle()
args = parser.parse_args()

img_path = args.img_path

with open(img_path, 'rb') as fo:
        dictio = pickle.load(fo, encoding='bytes')

labels = np.array(dictio[b'labels'])

img_num = int(args.img_num)

input_images = []
for i in range(img_num):
        img = np.reshape(dictio[b'data'][i], (3,32,32))
        img = np.transpose(img, (1,2,0))
        input_image = Image.fromarray(img)
        input_images.append(img)
input_images = np.asarray(input_images)
num_success_list, success_init_list, counter_acc_list = main(input_images, labels)

asr = np.sum(np.multiply(num_success_list, success_init_list)) / int(len([i for i in success_init_list if i == 1]))
acc_init = np.mean(success_init_list)
acc_noise = np.mean(np.multiply(success_init_list, counter_acc_list))
print(asr)
print(acc_init)
print(acc_noise)
print(f'Attack Success Rate: {asr:.4f}, Accuracy for old model: {acc_init:.4f}, Accuracy for new model: {acc_noise:.4f}')