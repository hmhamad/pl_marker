import os

from models.wrapper import ModelWrapper
import os
from models.pl_marker.pl_marker_re_main import call_pl_marker_re
from models.pl_marker.pl_marker_ner_main import call_pl_marker_ner
from mapping import map_datafile

TRANSLATE_ARGS = {'model_path': 'model_name_or_path', 'train_path': 'train_file', 'valid_path': 'dev_file', 'dataset_path': 'data_file', 'log_path': 'output_dir'}

class PLMarkerWrapper(ModelWrapper):
    def __init__(self, exp_cfgs) -> None:
        super().__init__(exp_cfgs)

    def train(self, model_path, train_path, valid_path, output_path, trial=None):
        
        if trial:
            self.exp_cfgs.model_args.re_params.edit('num_train_epochs',trial.suggest_int('re_train_epochs', 5, 20))
            self.exp_cfgs.model_args.re_params.edit('learning_rate',trial.suggest_float('re_lr', 1e-7, 1e-4))
            self.exp_cfgs.model_args.re_params.edit('weight_decay',trial.suggest_float('re_weight_decay', 0.0, 0.1))
            self.exp_cfgs.model_args.re_params.edit('max_seq_length',trial.suggest_int('max_seq_length', 128, 256))
            
            self.exp_cfgs.model_args.ner_params.edit('num_train_epochs',trial.suggest_int('ner_train_epochs', 5, 30))
            self.exp_cfgs.model_args.ner_params.edit('learning_rate',trial.suggest_float('ner_lr', 1e-7, 1e-4))
            self.exp_cfgs.model_args.re_params.edit('weight_decay',trial.suggest_float('ner_weight_decay', 0.0, 0.1))
            self.exp_cfgs.model_args.ner_params.edit('max_mention_ori_length',trial.suggest_int('max_mention_ori_length', 8, 32))
        
        # First Train NER model and save NER results
        ner_exportargs = {}
        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                ner_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.ner_params.configs.items():
            ner_exportargs[key] = val
        
        ner_model_path = os.path.join(model_path,'ner_model') if 'best_model' in model_path else model_path
        ner_exportargs[TRANSLATE_ARGS['model_path']] = ner_model_path
        ner_exportargs[TRANSLATE_ARGS['train_path']] = train_path
        ner_exportargs[TRANSLATE_ARGS['valid_path']] = valid_path
        ner_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        ner_exportargs['do_train'] = True
        ner_exportargs['do_eval'] = True
        ner_exportargs['do_test'] = False
        ner_exportargs['do_predict'] = False


        call_pl_marker_ner(ner_exportargs)
        
        re_exportargs = {}
        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                re_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.re_params.configs.items():
            re_exportargs[key] = val

        valid_path = os.path.join(self.exp_cfgs.model_args.log_path,'ent_pred_dev.json')
        re_exportargs[TRANSLATE_ARGS['model_path']] = model_path
        re_exportargs[TRANSLATE_ARGS['train_path']] = train_path
        re_exportargs[TRANSLATE_ARGS['valid_path']] = valid_path
        re_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        re_exportargs['do_train'] = True
        re_exportargs['do_eval'] = True
        re_exportargs['do_test'] = False
        re_exportargs['do_predict'] = False



        eval_micro_f1 = call_pl_marker_re(re_exportargs, trial=trial)

        return eval_micro_f1

    def eval(self, model_path, dataset_path, output_path, data_label='test', save_embeddings = False, Temp_rel = 1.0, Temp_ent = 1.0):
           
        # First evaluate NER model and save NER results
        ner_exportargs = {}
        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                ner_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.ner_params.configs.items():
            ner_exportargs[key] = val

        ner_model_path = os.path.join(model_path,'ner_model') if 'best_model' in model_path else model_path
        ner_exportargs[TRANSLATE_ARGS['model_path']] = ner_model_path
        ner_exportargs[TRANSLATE_ARGS['dataset_path']] = dataset_path
        ner_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        ner_exportargs['do_test'] = True
        ner_exportargs['do_train'] = False
        ner_exportargs['do_eval'] = False
        ner_exportargs['do_predict'] = False
        ner_exportargs['data_label'] = data_label



        call_pl_marker_ner(ner_exportargs)

        re_exportargs = {}

        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                re_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.re_params.configs.items():
            re_exportargs[key] = val

        dataset_path = os.path.join(self.exp_cfgs.model_args.log_path,f'ent_pred_{data_label}.json')
        re_exportargs[TRANSLATE_ARGS['model_path']] = model_path
        re_exportargs[TRANSLATE_ARGS['dataset_path']] = dataset_path
        re_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        re_exportargs['do_test'] = True
        re_exportargs['do_train'] = False
        re_exportargs['do_eval'] = False
        ner_exportargs['do_predict'] = False
        re_exportargs['data_label'] = data_label
        re_exportargs['save_embeddings'] = save_embeddings

        call_pl_marker_re(re_exportargs)

        map_datafile(
        in_path=os.path.join(self.exp_cfgs.model_args.log_path,data_label+'_predictions.json'),
        out_path=os.path.join(self.exp_cfgs.model_args.log_path,data_label+'_predictions_standard.json'),
        from_format='cluster_jsonl',
        to_format='standard')
    
    def predict(self, model_path, dataset_path, output_path, data_label='unlabeled', save_embeddings = False, Temp_rel = 1.0, Temp_ent = 1.0):
        # First evaluate NER model and save NER results
        ner_exportargs = {}
        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                ner_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.ner_params.configs.items():
            ner_exportargs[key] = val

        ner_model_path = os.path.join(model_path,'ner_model') if 'best_model' in model_path else model_path
        ner_exportargs[TRANSLATE_ARGS['model_path']] = ner_model_path
        ner_exportargs[TRANSLATE_ARGS['dataset_path']] = dataset_path
        ner_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        ner_exportargs['do_test'] = False
        ner_exportargs['do_train'] = False
        ner_exportargs['do_eval'] = False
        ner_exportargs['do_predict'] = True
        ner_exportargs['data_label'] = data_label



        call_pl_marker_ner(ner_exportargs)

        re_exportargs = {}

        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                re_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.re_params.configs.items():
            re_exportargs[key] = val

        dataset_path = os.path.join(self.exp_cfgs.model_args.log_path,f'ent_pred_{data_label}.json')
        re_exportargs[TRANSLATE_ARGS['model_path']] = model_path
        re_exportargs[TRANSLATE_ARGS['dataset_path']] = dataset_path
        re_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        re_exportargs['do_test'] = False
        re_exportargs['do_train'] = False
        re_exportargs['do_eval'] = False
        re_exportargs['do_predict'] = True
        re_exportargs['data_label'] = data_label


        call_pl_marker_re(re_exportargs)

    def calibrate(self):
        pass