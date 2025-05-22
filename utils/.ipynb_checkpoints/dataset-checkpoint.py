import numbers
import os
import queue as Queue
import threading
import random
import mxnet as mx
from augmentation import aug_rand_4_16
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_aug = transforms.Compose(aug_rand_4_16)
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank, root2=None, auth_ids=1000):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_aug = transforms.Compose(aug_rand_4_16)
        self.root_dir = root_dir
        self.root_dir2 = root2
        self.local_rank = local_rank
        self.imgidx, self.labels, self.num_ids, self.is_synth = self.scan(root_dir, root2, auth_ids=auth_ids)

    def scan(self, root_syn, root_auth, auth_ids):
        imgidex = []
        labels = []
        is_synth = []
        lb = -1
        list_dir = os.listdir(root_syn)
        list_dir.sort()
        for l in list_dir:
            images = os.listdir(os.path.join(root_syn, l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l, img))
                labels.append(lb)
                is_synth.append(True)
        list_dir2 = os.listdir(root_auth)
        list_dir2.sort()
        syn = lb
        for l in list_dir2[:auth_ids]:
            images = os.listdir(os.path.join(root_auth, l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l, img))
                labels.append(lb)
                is_synth.append(False)
        return imgidex, labels, [lb, lb - syn], is_synth

    def readImage(self, path, issyn):
        if issyn:
            return cv2.imread(os.path.join(self.root_dir, path))
        return cv2.imread(os.path.join(self.root_dir2, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.readImage(path, self.is_synth[index])
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
import numbers
import os
import queue as Queue
import threading
import random
import mxnet as mx
from utils.augmentation import aug_rand_4_16
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_aug = transforms.Compose(aug_rand_4_16)
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class FaceDatasetFolder(Dataset):
    def __init__(self, root_dir, local_rank, root2=None, auth_ids=1000):
        super(FaceDatasetFolder, self).__init__()
        self.transform = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
             ])
        self.transform_aug = transforms.Compose(aug_rand_4_16)
        self.root_dir = root_dir
        self.root_dir2 = root2
        self.local_rank = local_rank
        self.imgidx, self.labels, self.num_ids, self.is_synth = self.scan(root_dir, root2, auth_ids=auth_ids)

    def scan(self, root_syn, root_auth, auth_ids):
        imgidex = []
        labels = []
        is_synth = []
        lb = -1
        list_dir = os.listdir(root_syn)
        list_dir.sort()
        for l in list_dir:
            images = os.listdir(os.path.join(root_syn, l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l, img))
                labels.append(lb)
                is_synth.append(True)
        list_dir2 = os.listdir(root_auth)
        list_dir2.sort()
        syn = lb
        for l in list_dir2[:auth_ids]:
            images = os.listdir(os.path.join(root_auth, l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l, img))
                labels.append(lb)
                is_synth.append(False)
        return imgidex, labels, [lb, lb - syn], is_synth

    def readImage(self, path, issyn):
        if issyn:
            return cv2.imread(os.path.join(self.root_dir, path))
        return cv2.imread(os.path.join(self.root_dir2, path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.readImage(path, self.is_synth[index])
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)
