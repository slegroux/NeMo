# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from the following repository (https://github.com/ronghanghu/gqa_single_hop_baseline)

import json
import os.path as osp
from glob import glob

import h5py
import numpy as np

__all__ = [
    "SpatialFeatureLoader",
    "ObjectsFeatureLoader",
    "SceneGraphFeatureLoader",
    "VocabDict",
]


# Feature loader
class SpatialFeatureLoader:
    def __init__(self, feature_dir):
        info_file = osp.join(feature_dir, 'gqa_spatial_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)
        num_files = len(glob(osp.join(feature_dir, 'gqa_spatial_*.h5')))
        h5_paths = [osp.join(feature_dir, 'gqa_spatial_%d.h5' % n) for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

    def load_feature(self, imageId):
        info = self.all_info[imageId]
        file, idx = info['file'], info['idx']
        return self.h5_files[file]['features'][idx]


class ObjectsFeatureLoader:
    def __init__(self, feature_dir):
        info_file = osp.join(feature_dir, 'gqa_objects_info.json')
        with open(info_file) as f:
            self.all_info = json.load(f)
        num_files = len(glob(osp.join(feature_dir, 'gqa_objects_*.h5')))
        h5_paths = [osp.join(feature_dir, 'gqa_objects_%d.h5' % n) for n in range(num_files)]
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]

    def load_feature(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        feature = self.h5_files[file]['features'][idx]
        valid = get_valid(len(feature), num)
        return feature, valid

    def load_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        bbox = self.h5_files[file]['bboxes'][idx]
        valid = get_valid(len(bbox), num)
        return bbox, valid

    def load_feature_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        h5_file = self.h5_files[file]
        feature, bbox = h5_file['features'][idx], h5_file['bboxes'][idx]
        valid = get_valid(len(bbox), num)
        return feature, bbox, valid

    def load_normalized_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        bbox = self.h5_files[file]['bboxes'][idx]
        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)
        return normalized_bbox, valid

    def load_feature_normalized_bbox(self, imageId):
        info = self.all_info[imageId]
        file, idx, num = info['file'], info['idx'], info['objectsNum']
        h5_file = self.h5_files[file]
        feature, bbox = h5_file['features'][idx], h5_file['bboxes'][idx]
        w, h = info['width'], info['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)
        return feature, normalized_bbox, valid


class SceneGraphFeatureLoader:
    def __init__(self, scene_graph_file, vocab_name_file, vocab_attr_file, vocab_relation_file, max_num):
        print('Loading scene graph from %s' % scene_graph_file)
        with open(scene_graph_file) as f:
            self.SGs = json.load(f)
        self.name_dict = VocabDict(vocab_name_file)
        self.attr_dict = VocabDict(vocab_attr_file)
        self.rel_dict = VocabDict(vocab_relation_file)
        self.num_name = self.name_dict.num_vocab
        self.num_attr = self.attr_dict.num_vocab
        self.num_rel = self.rel_dict.num_vocab
        self.max_num = max_num

    def load_feature_normalized_bbox(self, imageId):
        sg = self.SGs[imageId]
        num = len(sg['objects'])
        # if num > self.max_num:
        #     print('truncating %d objects to %d' % (num, self.max_num))

        # object names and attributes
        feature = np.zeros((self.max_num, self.num_name + self.num_attr), np.float32)
        # relations between objects
        rel = np.zeros((self.max_num, self.num_name, self.num_rel), np.float32)

        names = feature[:, : self.num_name]
        attrs = feature[:, self.num_name :]
        bbox = np.zeros((self.max_num, 4), np.float32)

        # for populating relations
        obj_id_2_name = {}
        for idx, objId in enumerate(sorted(sg['objects'])):
            obj = sg['objects'][objId]
            obj_id_2_name[objId] = obj['name']

        objIds = sorted(sg['objects'])[: self.max_num]
        for idx, objId in enumerate(objIds):
            obj = sg['objects'][objId]
            bbox[idx] = obj['x'], obj['y'], obj['w'], obj['h']
            names[idx, self.name_dict.word2idx(obj['name'])] = 1.0
            # attributes of the objects
            for a in obj['attributes']:
                attrs[idx, self.attr_dict.word2idx(a)] = 1.0
            # relation between objects
            for relation in obj['relations']:
                obj_name = obj_id_2_name[relation['object']]
                obj_idx = self.name_dict.word2idx(obj_name)
                rel_idx = self.rel_dict.word2idx(relation['name'])
                rel[idx, obj_idx, rel_idx] = 1.0
        # xywh -> xyxy
        bbox[:, 2] += bbox[:, 0] - 1
        bbox[:, 3] += bbox[:, 1] - 1

        # normalize the bbox coordinates
        w, h = sg['width'], sg['height']
        normalized_bbox = bbox / [w, h, w, h]
        valid = get_valid(len(bbox), num)

        return feature, rel, normalized_bbox, valid


def get_valid(total_num, valid_num):
    valid = np.zeros(total_num, np.bool)
    valid[:valid_num] = True
    return valid


# Vocab class for constructing object, attributes vocabulary
def load_str_list(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    return lines


class VocabDict:
    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}
        self.num_vocab = len(self.word_list)
        self.UNK_idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_idx is not None:
            return self.UNK_idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does' ' not contain <unk>)' % w)
