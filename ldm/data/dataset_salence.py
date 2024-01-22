import os

import cv2
import numpy as np
from basicsr.utils import img2tensor


class dataset_fi_salence:
    def __init__(self, root_path_im, root_path_sal, mood_list, image_size, ratio):
        super(dataset_fi_salence, self).__init__()
        self.ratio = ratio
        self.root_path_im = root_path_im
        self.root_path_sal = root_path_sal
        self.files = []
        for mood in mood_list:
            cur_path = os.path.join(root_path_im, mood)
            img_id_list = os.listdir(cur_path)
            for i in range(int(len(img_id_list) * self.ratio[0]), int(len(img_id_list) * self.ratio[1])):
                self.files.append({'name': mood + '/' + img_id_list[i], 'sentence': 'photo of ' + str(mood)})

        self.files = np.array(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        name = file['name']

        im = cv2.imread(os.path.join(self.root_path_im, name))
        im = cv2.resize(im, (512, 512))
        im = img2tensor(im, bgr2rgb=True, float32=True) / 255.

        sal = cv2.imread(os.path.join(self.root_path_sal, name.replace('.jpg', '.png')))  # [:,:,0]
        sal = cv2.resize(sal, (512, 512))
        sal = img2tensor(sal, bgr2rgb=True, float32=True) / 255.  # [0].unsqueeze(0)#/255.

        sentence = file['sentence']
        return {'im': im, 'sal': sal, 'sentence': sentence, 'name': name}

    def __len__(self):
        return len(self.files)
