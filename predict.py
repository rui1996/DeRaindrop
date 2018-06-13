#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
#Models lib
from models import *

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--mode", type=str)
	parser.add_argument("--input_dir", type=str)
	parser.add_argument("--output_dir", type=str)
	args = parser.parse_args()
	return args

def align_to_four(img):
	print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
	#align to four
	a_row = int(img.shape[0]/4)*4
	a_col = int(img.shape[1]/4)*4
	img = img[0:a_row, 0:a_col]
	print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
	return img


def predict(image):
	image = np.array(image, dtype='float32')/255.
	image = image.transpose((2, 0, 1))
	image = image[np.newaxis, :, :, :]
	image = torch.from_numpy(image)
	image = Variable(image).cuda()

	out = model(image)[-1]

	out = out.cpu().data
	out = out.numpy()
	out = np.transpose(out, (0, 2, 3, 1))
	out = out[0, :, :, :]*255.
	
	return out


if __name__ == '__main__':
	args = get_args()

	model = Generator().cuda()
	model.load_state_dict(torch.load('./weights/gen.pkl'))

	if args.mode == 'demo':
		input_list = sorted(os.listdir(args.input_dir))
		num = len(input_list)
		for i in range(num):
			print ('processing image: %s'%(input_list[i]))
			img = cv2.imread(args.input_dir + input_list[i])
			img = align_to_four(img)
			result = predict(img)
			img_name = input_list[i].split('.')[0]
			cv2.imwrite(args.output_dir + img_name + '.jpg', result)

	else:
		print ('Mode to be update')
