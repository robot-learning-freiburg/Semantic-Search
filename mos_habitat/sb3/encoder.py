import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision import models


class EgocentricEncoders(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super(EgocentricEncoders, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # feature_size = 128
        self.use_uncertainty = "unc_map_small" in observation_space.spaces.keys()
        obs_keys = ["task_obs", "map_small", "map_large"]
        if self.use_uncertainty:
            obs_keys += ["unc_map_small", "unc_map_large"]

        assert list(observation_space.spaces.keys()) == list(obs_keys)
        for key, subspace in observation_space.spaces.items():
            feature_size = 128
            if key == "task_obs":
                # self.proprioceptive_dim = subspace.shape[0]
                extractors[key] = nn.Sequential(
                    nn.Linear(subspace.shape[0], feature_size), nn.ReLU()
                )
                # print("1",feature_size)
            elif key == "map_small":
                n_input_channels = subspace.shape[0]  # channel last
                if self.use_uncertainty:
                    n_input_channels += 1
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )

                test_tensor = th.zeros(
                    [n_input_channels, subspace.shape[1], subspace.shape[2]]
                )
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]

                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)

            elif key == "map_large":
                feature_size = 256
                n_input_channels = subspace.shape[0]  # channel last

                cnn = models.resnet18(pretrained=True)

                if self.use_uncertainty:
                    n_input_channels += 1
                    weight = cnn.conv1.weight.clone()
                    with th.no_grad():
                        cnn.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                        cnn.conv1.weight[:, :3] = weight
                        cnn.conv1.weight[:, 3] = weight.mean(dim=1)

                test_tensor = th.zeros(
                    [n_input_channels, subspace.shape[1], subspace.shape[2]]
                )
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]

                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)

            else:
                continue

            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        if self.use_uncertainty:
            observations["map_small"] = th.cat(
                [observations["map_small"], observations["unc_map_small"]], dim=1
            )
            observations["map_large"] = th.cat(
                [observations["map_large"], observations["unc_map_large"]], dim=1
            )

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)
