from src.training.lightningmodule.base_lightningmodule import BaseLightningModule
import torch

class DualPathDecoder(BaseLightningModule):
    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, wlen]
            'label_indices': batch_data_dict['cls_label']
        }
        output_dict = self.model(input_dict) # {'probabilities': [bs, nclasses]}
        target_dict = {
            'anechoic': batch_data_dict['anechoic'],
            'reverb': batch_data_dict['reverb'],
            'noise': batch_data_dict['noise'],
            'cls_label': batch_data_dict['cls_label'],
            'sed_label': batch_data_dict['sed_label'],
            'doa_label': batch_data_dict['doa_label'],
            'onoff_label': (torch.norm(batch_data_dict['doa_label'], dim=-1) > 0.5).float()
        }
        loss_dict, _ = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, wlen]
        }
        output_dict = self.model(input_dict) # {'probabilities': [bs, nclasses]}
        target_dict = {
            'anechoic': batch_data_dict['anechoic'],
            'reverb': batch_data_dict['reverb'],
            'noise': batch_data_dict['noise'],
            'cls_label': batch_data_dict['cls_label'],
            'sed_label': batch_data_dict['sed_label'],
            'doa_label': batch_data_dict['doa_label'],
            'onoff_label': (torch.norm(batch_data_dict['doa_label'], dim=-1) > 0.5).float()
        }
        loss_dict, perm_dict = self.loss_func(output_dict, target_dict)
        loss_dict = {k: v.item() for k,v in loss_dict.items()}

        active_mask = ((torch.norm(perm_dict['doa_label'], dim=-1) > 0.5) * (torch.sigmoid(perm_dict['onoff_label']) > 0.5)).float()
        cls_binary  = torch.zeros_like(perm_dict['cls_label']).scatter_(-1, torch.argmax(perm_dict['cls_label'], dim=-1, keepdim=True), 1)
        sed_out     = active_mask.unsqueeze(dim=-1) * cls_binary.unsqueeze(dim=2)
        perm_dict['sed_label'] = sed_out

        if self.metric_func: # add metrics
            metric = self.metric_func(perm_dict, target_dict)
            for k,v in metric.items(): # metric return [bs] for better calculation of mean
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict
