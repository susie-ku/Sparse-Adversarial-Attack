from main import main
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

img_num = int(args.img_num)

labels = np.array(dictio[b'labels'])

num_success_array = np.array([])
success_init_array = np.array([])
counter_acc_array = np.array([])
for i in tqdm(range(img_num)):
        img = np.reshape(dictio[b'data'][i], (3,32,32))
        img = np.transpose(img, (1,2,0))
        input_image = Image.fromarray(img)
        num_success, success_init, counter_acc = main(input_image, labels[i])
        num_success_array = np.append(num_success_array, num_success)
        success_init_array = np.append(success_init_array, success_init)
        counter_acc_array = np.append(counter_acc_array, counter_acc)

asr = np.mean(np.multiply(num_success_array, success_init_array))
acc_init = np.mean(success_init_array)
acc_noise = np.mean(np.multiply(success_init_array, counter_acc_array))
print(f'Attack Success Rate: {asr:.4f}, Accuracy for old model: {acc_init:.4f}, Accuracy for new model: {acc_noise:.4f}')

        