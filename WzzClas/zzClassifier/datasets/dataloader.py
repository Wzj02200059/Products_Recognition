import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Product_Dataloader_Close(object):
    def __init__(self, train_dataroot, val_dataroot, use_gpu=True, num_workers=8, batch_size=32,
                 img_size=224):
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = ImageFolder(root=train_dataroot, transform=train_transform)
        print('All Train Data:', len(trainset))

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = ImageFolder(root=val_dataroot, transform=transform)
        print('All Test Data:', len(testset))

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.num_classes = len(trainset.classes)
        self.known = len(trainset.classes)

        print('Selected Labels: ', self.known)
        print('Train: ', len(trainset), 'Test: ', len(testset))
        print('All Test: ', (len(testset)))

class Product_Dataloader_Open(object):
    def __init__(self, train_dataroot, val_dataroot, unknown_train_dataroot, unknown_val_dataroot, use_gpu=True, num_workers=8, batch_size=32,
                 img_size=224):
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        pin_memory = True if use_gpu else False

        trainset = ImageFolder(root=train_dataroot, transform=train_transform)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = ImageFolder(root=val_dataroot, transform=transform)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        unknow_trainset = ImageFolder(root=unknown_train_dataroot, transform=transform)

        self.unknow_train_loader = torch.utils.data.DataLoader(
            unknow_trainset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        unknow_valset = ImageFolder(root=unknown_val_dataroot, transform=transform)

        self.unknow_val_loader = torch.utils.data.DataLoader(
            unknow_valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.num_classes = len(trainset.classes) + len(unknow_trainset.classes) + len(unknow_valset.classes)
        self.known = len(trainset.classes)
        self.unknown = len(unknow_trainset.classes) + len(unknow_valset.classes)

    def read_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise Exception(
                'Can not read the file {} as an image'.format(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for trans in self.trans_list:
            img = trans(img)
        return img


