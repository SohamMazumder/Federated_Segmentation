import numpy as np
import torch
import nibabel as nb

from os import listdir
from os.path import isfile, join
import torch.utils.data as data
import torch.nn.functional as F
from collections import OrderedDict
#from unet3d.utils import save_checkpoint


class IMDB(data.Dataset):
	def __init__(self, X, y, transforms=None, dataset='malc', malc_factor=16, patches=False, desired_patch_size=64):
		if dataset == 'malc':
			if not patches:
				self.X, self.belonging = IMDB.slicing_multiplier(X, malc_factor)
				self.y, _ = IMDB.slicing_multiplier(y, malc_factor)
			else:
				self.X, self.belonging = IMDB.patches_creator(X, desired_patch_size)
				self.y, _ = IMDB.patches_creator(y, desired_patch_size)
			self.X, self.y, self.belonging = IMDB.clean_data(self.X, self.y, self.belonging)
		elif dataset == 'hippo':
			self.X = X
			self.y = y
			self.belonging = []
		self.dataset = dataset
		self.transforms = transforms

	@staticmethod
	def slicing_multiplier(list_of_values, factor):
		out = np.empty((len(list_of_values) * factor), dtype=list)
		belonging = np.ones((len(list_of_values) * factor)) * -1
		depth = np.shape(list_of_values[0])[2]//factor
		for i in range(len(list_of_values)):
			for j in range(factor):
				out[(i * factor) + j] = list_of_values[i][4:-4, 4:-4, j * depth:(j + 1) * depth]
				belonging[(i * factor) + j] = i
		return out, belonging

	@staticmethod
	def patches_creator(list_of_values, desired_size):
		f = np.shape(list_of_values[0])[0] // desired_size
		ds = desired_size
		out = np.empty((len(list_of_values) * f**3), dtype=list)
		belonging = np.ones((len(list_of_values) * f**3)) * -1
		for i in range(len(list_of_values)):
			for j1 in range(f):
				for j2 in range(f):
					for j3 in range(f):
						out[i*f**3 + j1*f**2 + j2*f + j3] = list_of_values[i][j1*ds:(j1+1)*ds, j2*ds:(j2+1)*ds, j3*ds:(j3+1)*ds]
						belonging[i*f**3 + j1*f**2 + j2*f + j3] = i
		return out, belonging

	@staticmethod
	def clean_data(data, labels, belonging):
		clean_data, clean_labels, clean_belonging = [], [], []
		for i, frame in enumerate(labels):
			unique, counts = np.unique(frame, return_counts=True)
			if counts[0] / np.sum(counts) < .99:
				clean_labels.append(frame)
				clean_data.append(data[i])
				clean_belonging.append(belonging[i])
		return np.array(clean_data), np.array(clean_labels), clean_belonging

	def __getitem__(self, index):
		# potato = 31
		# eight = 8 * potato
		img = torch.from_numpy(self.X[index])
		label = torch.from_numpy(self.y[index])
		return img, label, (self.belonging[index] if self.dataset == 'malc' else [0])

	def __len__(self):
		return len(self.y)

'''
class ModelSaver():
	def __init__(self, optimizer, optimizer_sf, model, checkpoint_dir, num_bits=1.5, num_sf=1):
		self.model = model
		self.scaling_factors = optimizer_sf.param_groups[0]['params']
		self.weights_orig = optimizer.param_groups[1]['params']
		self.checkpoint_dir = checkpoint_dir
		assert num_bits in [1.5, 1], "Error! ModelSaver is only compatible with num_bits being" \
									"1 or 1.5 (binary or ternary), provided " + str(num_bits)
		self.ternary = num_bits == 1.5
		self.num_sf = num_sf

	def compress_model(self, is_best):
		ones_b = []
		if self.ternary:
			zeros_b = []
		# minus ones are those that are not in these two, see TernaryNet picture

		for i in range(self.weights_orig.__len__()):
			if self.ternary:
				ones_b.append(np.packbits((self.weights_orig[i] / self.scaling_factors[i][0] == 1).detach().cpu().numpy(), axis=1))
				zeros_b.append(np.packbits((self.weights_orig[i] == 0).detach().cpu().numpy(), axis=1))
			else:
				if self.num_sf == 2:
					ones_b.append(np.packbits((self.weights_orig[i] /
												self.scaling_factors[i][0][0] == 1).detach().cpu().numpy(), axis=1))
				else:
					ones_b.append(np.packbits((self.weights_orig[i] /
												self.scaling_factors[i] == 1).detach().cpu().numpy(), axis=1))

		new_state_dict = OrderedDict()
		i = 0
		for k in self.model.state_dict().keys():
			if 'conv' in k and 'weight' in k and 'norm' not in k and k != 'encoders.0.double_conv.conv1.weight':
				new_state_dict[k] = {
					'ones': ones_b[i],
					'sf': self.scaling_factors[i]
				}
				if self.ternary:
					new_state_dict[k]['zeros'] = zeros_b[i]
				i += 1
			else:
				new_state_dict[k] = self.model.state_dict()[k]

		save_checkpoint(new_state_dict, False, self.checkpoint_dir, file_name='compressed_last.pytorch')
		if is_best:
			save_checkpoint(new_state_dict, False, self.checkpoint_dir, file_name='compressed_best.pytorch')

	@staticmethod
	def decompress(compressed_state, num_bits=1.5, shift_down=False, num_sf=2):
		new_state_dict = OrderedDict()
		for k in compressed_state.keys():
			if 'conv' in k and 'weight' in k and 'norm' not in k and k != 'encoders.0.double_conv.conv1.weight':
				ones_b = compressed_state[k]['ones']
				if num_bits == 1.5:
					zeros_b = compressed_state[k]['zeros']
				sf = compressed_state[k]['sf']

				old_ones = np.unpackbits(ones_b, axis=1)
				if num_bits == 1.5:
					old_zeros = np.unpackbits(zeros_b, axis=1)
					old_minus_ones = np.asarray(np.logical_not(np.logical_or(old_ones, old_zeros)), dtype=int)
					new_state_dict[k] = torch.from_numpy(old_ones * 1.) * sf[0] - torch.from_numpy(old_minus_ones * 1.) * sf[1]
				else:
					if shift_down:
						old_minus_ones = (old_ones == 0)  # could be old_zeros if there was no shift_down
						if num_sf == 2:
							new_state_dict[k] = torch.from_numpy(old_ones * 1.) * sf[0] -\
												torch.from_numpy(old_minus_ones * 1.) * sf[1]
						else:
							new_state_dict[k] = torch.from_numpy(old_ones * 1.) * sf - \
												torch.from_numpy(old_minus_ones * 1.) * sf
					else:
						new_state_dict[k] = torch.from_numpy(old_ones * 1.) * sf

			else:
				new_state_dict[k] = compressed_state[k]

		return new_state_dict
'''

def pad_to(orig_input, requested_size):
	orig_input = torch.from_numpy(orig_input)
	size = orig_input.size()
	l = (requested_size - size[-3]) // 2 + size[-3] % 2  # left
	r = (requested_size - size[-3]) // 2                 # right
	t = (requested_size - size[-2]) // 2 + size[-2] % 2  # top
	b = (requested_size - size[-2]) // 2                 # bottom
	f = (requested_size - size[-1]) // 2 + size[-1] % 2  # front
	ba = (requested_size - size[-1]) // 2                # back
	return F.pad(orig_input, (f, ba, t, b, l, r)).numpy()


def load_data(file_path, pad_to_size=None):
	volume_nifty, labelmap_nifty = nb.load(file_path[0]), nb.load(file_path[1])
	volume, labelmap = volume_nifty.get_fdata(), labelmap_nifty.get_fdata()
	# if volume[0,0,0] % 1 != 0:
	volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
	if pad_to_size is not None:
		volume = pad_to(volume, pad_to_size)
		labelmap = pad_to(labelmap, pad_to_size)
	# volume, labelmap = preprocessor.rotate_orientation(volume, labelmap, orientation)
	return volume, labelmap, volume_nifty.header


def get_data(volumes_dir, labels_dir, training_amount, save_data_to_file=True, files_for_ref=None, pad_to_size=None,
			dataset='hippo'):

	if save_data_to_file:
		if dataset == 'hippo':
			volumes_files = [f for f in listdir(volumes_dir) if isfile(join(volumes_dir, f)) and f[0] != '.']
			labels_files = [f for f in listdir(labels_dir) if isfile(join(labels_dir, f)) and f[0] != '.']
			correspondance = True
			if len(labels_files) == len(volumes_files):
				for i in range(len(volumes_files)):
					if volumes_files[i] != labels_files[i]:
						correspondance = False
						break
			else:
				correspondance = False
			if not correspondance:
				import sys
				sys.exit("\nvolumes and labels do not correspond with each other!")

			idxs = np.arange(len(labels_files))
			np.random.shuffle(idxs)
			volumes_tr = np.empty((training_amount), dtype=object)
			labels_tr = np.empty((training_amount), dtype=object)
			volumes_te = np.empty((len(volumes_files) - training_amount), dtype=object)
			labels_te = np.empty((len(labels_files) - training_amount), dtype=object)

			with open(files_for_ref[0], 'w') as tr_file:
				with open(files_for_ref[1], 'w') as te_file:
					load_training_test(idxs, labels_dir, labels_files, labels_te, labels_tr, training_amount,
										volumes_dir, volumes_files, volumes_te, volumes_tr, save_data_to_file,
										te_file=te_file, tr_file=tr_file, pad_to_size=pad_to_size)
		else:
			raise Exception("with MALC save_data_to_file should be false!")

	else:
		train_files = get_file_content(files_for_ref[0])
		test_files = get_file_content(files_for_ref[1])

		volumes_tr = np.empty((len(train_files)), dtype=object)
		labels_tr = np.empty((len(train_files)), dtype=object)
		volumes_te = np.empty((len(test_files)), dtype=object)
		labels_te = np.empty((len(test_files)), dtype=object)

		if dataset == 'hippo':
			for i in range(len(train_files)):
				volumes_tr[i], labels_tr[i], _ = load_data(
					[join(volumes_dir, train_files[i]), join(labels_dir, train_files[i])], pad_to_size=pad_to_size)
			for i in range(len(test_files)):
				volumes_te[i], labels_te[i], _ = load_data(
					[join(volumes_dir, test_files[i]), join(labels_dir, test_files[i])], pad_to_size=pad_to_size)
		else:
			# labels_set = [2, 3, 4, 5, 7, 8, 10, 11, 12,
			# 				13, 14, 15, 16, 17, 18, 24, 26,
			# 				28, 41, 42, 43, 44, 46, 47, 49,
			# 				50, 51, 52, 53, 54, 58, 60, 85]
			# labels_set = [45, 211, 52, 50, 41, 39, 60,
			# 				37, 58, 56, 4, 11, 35, 48, 32,
			# 				46, 30, 62, 44, 210, 51, 49, 40,
			# 				38, 59, 36, 57, 55, 47, 31, 23, 61, 69]
			labels_set = [45, 211, 44, 210, 52, 41, 39, 60, 37, 58, 56, 4, 11,
						35, 48, 32, 62, 51, 40, 38, 59, 36, 57, 55, 47, 31, 61]

			for i in range(len(train_files)):
				volumes_tr[i], labels_tr[i], _ = load_data(
					[join(volumes_dir, train_files[i]+'/mri/orig.mgz'), join(labels_dir, train_files[i]+'_glm.mgz')])
				labels_tr[i][(labels_tr[i] >= 100) & (labels_tr[i] % 2 == 0)] = 210
				labels_tr[i][(labels_tr[i] >= 100) & (labels_tr[i] % 2 == 1)] = 211

				new_labels = np.zeros_like(labels_tr[i])
				for j in range(len(labels_set)):
					mask = np.zeros_like(labels_tr[i])
					mask[labels_tr[i] == labels_set[j]] = 1
					new_labels = new_labels + (j+1)*mask
				labels_tr[i] = new_labels

			for i in range(len(test_files)):
				volumes_te[i], labels_te[i], _ = load_data(
					[join(volumes_dir, test_files[i]+'/mri/orig.mgz'), join(labels_dir, test_files[i]+'_glm.mgz')])
				labels_te[i][(labels_te[i] >= 100) & (labels_te[i] % 2 == 0)] = 210
				labels_te[i][(labels_te[i] >= 100) & (labels_te[i] % 2 == 1)] = 211

				new_labels = np.zeros_like(labels_te[i])
				for j in range(len(labels_set)):
					mask = np.zeros_like(labels_te[i])
					mask[labels_te[i] == labels_set[j]] = 1
					new_labels = new_labels + (j + 1) * mask
				labels_te[i] = new_labels

	return volumes_tr, labels_tr, volumes_te, labels_te


def load_training_test(idxs, labels_dir, labels_files, labels_te, labels_tr, training_amount,
						volumes_dir, volumes_files, volumes_te, volumes_tr, save_data_to_file,
						te_file=None, tr_file=None, pad_to_size=None):

	for i in range(len(volumes_files)):
		if i < training_amount:
			volumes_tr[i], labels_tr[i], _ = load_data(
				[join(volumes_dir, volumes_files[idxs[i]]), join(labels_dir, labels_files[idxs[i]])],
				pad_to_size=pad_to_size)
			if save_data_to_file:
				tr_file.write(volumes_files[idxs[i]] + "\n")
		else:
			volumes_te[i - training_amount], labels_te[i - training_amount], _ = load_data(
				[join(volumes_dir, volumes_files[idxs[i]]), join(labels_dir, labels_files[idxs[i]])],
				pad_to_size=pad_to_size)
			if save_data_to_file:
				te_file.write(volumes_files[idxs[i]] + "\n")


def get_file_content(file_name):
	with open(file_name) as f:
		content = f.readlines()
	return [x.strip() for x in content]
