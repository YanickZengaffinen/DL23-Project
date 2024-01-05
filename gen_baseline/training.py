import torch
from gen_baseline.project import buildData, train_model
from mmpretrain.models import ResNet
# pip install -U openmim && mim install "mmpretrain>=1.0.0rc8"
# pip install mmpretrain

###### Main ######

# Get the data.
o_train_ds, o_val_ds, o_test_ds, m_train_ds, m_val_ds, m_test_ds = buildData()

# Train Omniglot.
feature_extractor = ResNet(depth=18, in_channels=1)
feature_s = 512
train_epochs = 50
num_classes = 1100
lr = 0.0005

print("Training on Omniglot.")
omni_model = train_model(feature_extractor, feature_s, o_train_ds, train_epochs, num_classes, lr)
torch.save(omni_model, './gen_baseline/models/omni_model.pt')

# Train MiniImageNet
feature_extractor = ResNet(depth=18, in_channels=3)
feature_s = 512*3*3
train_epochs = 50
num_classes = 64
lr = 0.0005

print("Training on MiniImageNet.")
mini_model = train_model(feature_extractor, feature_s, m_train_ds, train_epochs, num_classes, lr)
torch.save(mini_model, './gen_baseline/mini_model.pt')