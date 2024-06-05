import numpy as np
import os
import torch
import uproot
from torch.utils.data import Dataset

# set our seed
torch.manual_seed(42)

# Define PDG/RECO IDs
MET_ID = 40
START_SEQ_ID = 50
SEPARATOR_ID = 51
END_SEQ_ID = 52
b_PDG_ID = 5
bbar_PDG_ID = -5

# Define sequence segment identification vectors
START_SEQ_VEC = [1, 0, 0, 0]
SEPARATOR_SEQ_VEC = [0, 1, 0, 0]
END_SEQ_VEC = [0, 0, 1, 0]


class GenDataset(Dataset):
    """
    The Generator level dataset used for training particle physics event encodings.
    Args:
        :param events: The uproot file containing all the events to be trained on
        :param gen_reco_ids (torch.Tensor): List containing True/False values on whether the particle is considered
            to be generated or not.
        :param standardize (bool, default=True): Whether or not to standardize the dataset
        :param pretraining (bool, default=True): Whether to randomly apply masking on the four vectors when grabbing an element out of the dataset.
            If false, the entire generator-level four vectors are masked.
        :param ids (torch.Tensor): A list containing the PDG/RECO IDs of each particle.
            MET = 40
            Start-of-sequence token = 50
            Separator token = 51
            End-of-sequence token = 52
    """
    def __init__(self, events, gen_reco_ids, standardize=True, pretraining=True, ids=None, mask_probability=0.15):
        self.events = events
        self.num_events = len(self.events['gen_lbar_pt'])
        self.mask_probability = mask_probability

        # TODO: This is hard-coded for dileptonic ttbar. Think of some way to make this more modular
        self.gen_four_vectors = torch.cat(
            (
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_pt'])[:, None],
                        torch.from_numpy(self.events['gen_lbar_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_b_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_nu_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_nubar_pt'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_lbar_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_b_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_nu_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_nubar_eta'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_lbar_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_b_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_nu_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_nubar_phi'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_mass'])[:, None], 
                        torch.from_numpy(self.events['gen_lbar_mass'])[:, None], 
                        torch.from_numpy(self.events['gen_b_mass'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_mass'])[:, None], 
                        torch.zeros(self.num_events)[:, None], 
                        torch.zeros(self.num_events)[:, None]
                    ),
                    axis=1
                )[:, :, None]
            ),
            axis=2
        )

        if standardize:
            self.gen_means = torch.mean(self.gen_four_vectors, axis=0)
            self.gen_stdevs = torch.std(self.gen_four_vectors, 0, True)
            self.gen_four_vectors = (self.gen_four_vectors - self.gen_means) / self.gen_stdevs

            # Entries set to 0 (MET mass/eta) and values with 0 stdev go to nan
            self.gen_four_vectors = torch.nan_to_num(self.gen_four_vectors)

        # Create the four vector input into the transformer
        self.four_vectors = torch.cat(
            (
                torch.Tensor([START_SEQ_VEC])[None, :, :].repeat(self.num_events, 1, 1),
                self.gen_four_vectors,
                torch.Tensor([END_SEQ_VEC])[None, :, :].repeat(self.num_events, 1, 1)
            ),        
            axis=1
        )
        
        # the PDG/RECO IDs for the particles
        # TODO: Again, hard-coded for ttbar dileptonic. Make more generalized
        self.ids = torch.cat(
            (
                torch.Tensor([START_SEQ_ID])[None, :].repeat(self.num_events, 1),
                torch.from_numpy(self.events['gen_l_pdgid'][:, None]),
                torch.from_numpy(self.events['gen_lbar_pdgid'][:, None]),
                torch.Tensor([b_PDG_ID])[None, :].repeat(self.num_events, 1),
                torch.Tensor([bbar_PDG_ID])[None, :].repeat(self.num_events, 1),
                torch.from_numpy(-1 * self.events['gen_l_pdgid'][:, None] - 1), # corresponding antineutrino
                torch.from_numpy(-1 * self.events['gen_lbar_pdgid'][:, None] + 1), # corresponding neutrino
                torch.Tensor([END_SEQ_ID])[None, :].repeat(self.num_events, 1)
            ),
            axis=1
        )
        self.ids += 40 # offset by 40 to ensure that negative PDG/RECO IDs are positive for look-up table embedding
        self.ids = self.ids.type(torch.int) # convert to integer for look-up table embedding

        # 0/1 for whether the particle is RECO/GEN
        self.gen_reco_ids = gen_reco_ids

        self.pretraining = pretraining

    def __len__(self):
        return self.num_events

    def __getitem__(self, index):
        """ Returns the masked four vectors, gen/reco IDs, and particle ids for the input of Bumblebee. 
            Labels are the unmasked four vectors. 50% of the time random particles' four vectors are masked with a 15% probability
            per particle. The other 50% of the time, the entire gen record of particles' four vectors are masked.
        :param index: The index of the event
        :return: A tuple where the first element is a 3-tuple of particle IDs, gen/reco IDs, and masked particle four vectors.
        The second element is the labels: unmasked four vectors.
        """
        # TODO: This is hard-coded for ttbar dileptonic decays. Generalize this.
        if self.pretraining: # pretraining
            dice_rolls = torch.rand(6)
            zerod_mask = ~(dice_rolls < self.mask_probability)
            zerod_mask = torch.cat(
                (torch.ones(1), zerod_mask, torch.ones(1)),
                dim=0
            )

        four_vector_mask = zerod_mask[:, None].repeat(1, 4)
        four_vector_mask[5:7, 3] = torch.ones(four_vector_mask[5:7, 3].shape)
        masked_four_vectors = (self.four_vectors[index] * four_vector_mask)

        return torch.cat((self.ids[index][:, None], self.gen_reco_ids[:, None], masked_four_vectors, zerod_mask[:, None]), axis=1), self.four_vectors[index]

    def get_standardization_params(self):
        params = (self.gen_means, self.gen_stdevs)
        return params


class GenRecoDataset(Dataset):
    """
    The Generator-Reconstruction level dataset used for training particle physics event encodings.
    Args:
        :param events: The uproot file containing all the events to be trained on
        :param gen_reco_ids (torch.Tensor): List containing True/False values on whether the particle is considered
            to be generated or not.
        :param standardize (bool, default=True): Whether or not to standardize the dataset
        :param pretraining (bool, default=True): Whether to randomly apply masking on the four vectors when grabbing an element out of the dataset.
            If false, the entire generator-level four vectors are masked.
        :param ids (torch.Tensor): A list containing the PDG/RECO IDs of each particle.
            MET = 40
            Start-of-sequence token = 50
            Separator token = 51
            End-of-sequence token = 52
    """
    def __init__(self, events, gen_reco_ids, standardize=True, pretraining=True, ids=None, mask_probability=0.15):
        self.events = events
        self.num_events = len(self.events['lbar_pt'])
        self.mask_probability = mask_probability

        # TODO: This is hard-coded for dileptonic ttbar. Think of some way to make this more modular
        self.reco_four_vectors = torch.cat(
            (
                torch.cat(
                    (
                        torch.from_numpy(self.events['l_pt'])[:, None],
                        torch.from_numpy(self.events['lbar_pt'])[:, None],
                        torch.from_numpy(self.events['b_pt'])[:, None], 
                        torch.from_numpy(self.events['bbar_pt'])[:, None], 
                        torch.from_numpy(self.events['met_pt'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['l_eta'])[:, None], 
                        torch.from_numpy(self.events['lbar_eta'])[:, None], 
                        torch.from_numpy(self.events['b_eta'])[:, None], 
                        torch.from_numpy(self.events['bbar_eta'])[:, None], 
                        torch.zeros(self.num_events)[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['l_phi'])[:, None], 
                        torch.from_numpy(self.events['lbar_phi'])[:, None], 
                        torch.from_numpy(self.events['b_phi'])[:, None], 
                        torch.from_numpy(self.events['bbar_phi'])[:, None], 
                        torch.from_numpy(self.events['met_phi'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['l_mass'])[:, None], 
                        torch.from_numpy(self.events['lbar_mass'])[:, None], 
                        torch.from_numpy(self.events['b_mass'])[:, None], 
                        torch.from_numpy(self.events['bbar_mass'])[:, None], 
                        torch.zeros(self.num_events)[:, None]
                    ),
                    axis=1
                )[:, :, None]
            ),
            axis=2
        )
        self.gen_four_vectors = torch.cat(
            (
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_pt'])[:, None],
                        torch.from_numpy(self.events['gen_lbar_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_b_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_nu_pt'])[:, None], 
                        torch.from_numpy(self.events['gen_nubar_pt'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_lbar_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_b_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_nu_eta'])[:, None], 
                        torch.from_numpy(self.events['gen_nubar_eta'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_lbar_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_b_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_nu_phi'])[:, None], 
                        torch.from_numpy(self.events['gen_nubar_phi'])[:, None]
                    ),
                    axis=1
                )[:, :, None],
                torch.cat(
                    (
                        torch.from_numpy(self.events['gen_l_mass'])[:, None], 
                        torch.from_numpy(self.events['gen_lbar_mass'])[:, None], 
                        torch.from_numpy(self.events['gen_b_mass'])[:, None], 
                        torch.from_numpy(self.events['gen_bbar_mass'])[:, None], 
                        torch.zeros(self.num_events)[:, None], 
                        torch.zeros(self.num_events)[:, None]
                    ),
                    axis=1
                )[:, :, None]
            ),
            axis=2
        )

        if standardize:
            self.reco_means = torch.mean(self.reco_four_vectors, axis=0)
            self.reco_stdevs = torch.std(self.reco_four_vectors, 0, True)
            self.gen_means = torch.mean(self.gen_four_vectors, axis=0)
            self.gen_stdevs = torch.std(self.gen_four_vectors, 0, True)
            self.reco_four_vectors = (self.reco_four_vectors - self.reco_means) / self.reco_stdevs
            self.gen_four_vectors = (self.gen_four_vectors - self.gen_means) / self.gen_stdevs

            # Entries set to 0 (MET mass/eta) and values with 0 stdev go to nan
            self.reco_four_vectors = torch.nan_to_num(self.reco_four_vectors)
            self.gen_four_vectors = torch.nan_to_num(self.gen_four_vectors)

        # Create the four vector input into the transformer
        self.four_vectors = torch.cat(
            (
                torch.Tensor([START_SEQ_VEC])[None, :, :].repeat(self.num_events, 1, 1),
                self.reco_four_vectors,
                torch.Tensor([SEPARATOR_SEQ_VEC])[None, :, :].repeat(self.num_events, 1, 1),
                self.gen_four_vectors,
                torch.Tensor([END_SEQ_VEC])[None, :, :].repeat(self.num_events, 1, 1)
            ),        
            axis=1
        )
        
        # the PDG/RECO IDs for the particles
        # TODO: Again, hard-coded for ttbar dileptonic. Make more generalized
        self.ids = torch.cat(
            (
                torch.Tensor([START_SEQ_ID])[None, :].repeat(self.num_events, 1),
                torch.from_numpy(self.events['l_pdgid'][:, None]),
                torch.from_numpy(self.events['lbar_pdgid'][:, None]),
                torch.Tensor([b_PDG_ID])[None, :].repeat(self.num_events, 1),
                torch.Tensor([bbar_PDG_ID])[None, :].repeat(self.num_events, 1),
                torch.Tensor([MET_ID])[None, :].repeat(self.num_events, 1),
                torch.Tensor([SEPARATOR_ID])[None, :].repeat(self.num_events, 1),
                torch.from_numpy(self.events['gen_l_pdgid'][:, None]),
                torch.from_numpy(self.events['gen_lbar_pdgid'][:, None]),
                torch.Tensor([b_PDG_ID])[None, :].repeat(self.num_events, 1),
                torch.Tensor([bbar_PDG_ID])[None, :].repeat(self.num_events, 1),
                torch.from_numpy(-1 * self.events['gen_l_pdgid'][:, None] - 1), # corresponding antineutrino
                torch.from_numpy(-1 * self.events['gen_lbar_pdgid'][:, None] + 1), # corresponding neutrino
                torch.Tensor([END_SEQ_ID])[None, :].repeat(self.num_events, 1)
            ),
            axis=1
        )
        self.ids += 40 # offset by 40 to ensure that negative PDG/RECO IDs are positive for look-up table embedding
        self.ids = self.ids.type(torch.int) # convert to integer for look-up table embedding

        # 0/1 for whether the particle is RECO/GEN
        self.gen_reco_ids = gen_reco_ids

        self.pretraining = pretraining

    def __len__(self):
        return self.num_events


    def __getitem__(self, index):
        """ Returns the masked four vectors, gen/reco IDs, and particle ids for the input of Bumblebee. 
            Labels are the unmasked four vectors. 50% of the time random particles' four vectors are masked with a 15% probability
            per particle. The other 50% of the time, the entire gen record of particles' four vectors are masked.
        :param index: The index of the event
        :return: A tuple where the first element is a 3-tuple of particle IDs, gen/reco IDs, and masked particle four vectors.
        The second element is the labels: unmasked four vectors.
        """
        # TODO: This is hard-coded for ttbar dileptonic decays. Generalize this.
        if self.pretraining: # pretraining
            task_roll = torch.rand(1)
            if task_roll < 0.5: # do the 15%-of-particles-are-masked task
                dice_rolls = torch.rand(11)
                zerod_mask = ~(dice_rolls < self.mask_probability)
                zerod_mask = torch.cat(
                    (torch.ones(1), zerod_mask[:4], torch.ones(1), zerod_mask[4:], torch.ones(1)),
                    dim=0
                )
            else: # do the mask-gen-particles task
                dice_roll = torch.rand(1)
                if dice_roll < 0.5: # mask all of the gen
                    zerod_mask = torch.cat((torch.ones(7), torch.zeros(6), torch.ones(1)))
                else: # don't mask anything
                    zerod_mask = torch.ones(14)
        else: # finetuning
            zerod_mask = torch.cat((torch.ones(7), torch.zeros(6), torch.ones(1)))

        four_vector_mask = zerod_mask[:, None].repeat(1, 4)
        four_vector_mask[11:13, 3] = torch.ones(four_vector_mask[11:13, 3].shape)
        masked_four_vectors = (self.four_vectors[index] * four_vector_mask)

        return torch.cat((self.ids[index][:, None], self.gen_reco_ids[:, None], masked_four_vectors, zerod_mask[:, None]), axis=1), self.four_vectors[index]


class DiscDataset(Dataset):
    def __init__(self, target_value, sb_dataset):
        super(DiscDataset, self).__init__()
        self.target_value = target_value
        self.sb_dataset = sb_dataset
        self.targets = torch.tensor(data=[target_value for _ in range(len(sb_dataset))])

    def __len__(self):
        return len(self.sb_dataset)

    def __getitem__(self, item):
        return self.sb_dataset[item][0], self.targets[item]