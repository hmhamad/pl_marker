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

    def train(self, model_path, train_path, valid_path, output_path, trial=None, curriculum_learning = False, train_ner = True, ner_train_path = None):
        
        if trial and not curriculum_learning:
            lr_range = [i*1e-6 for i in range(1, 10)] + [i*1e-5 for i in range(1, 10)]
            batch_size = trial.suggest_categorical('batch_size', [2,4,8])
            
            self.exp_cfgs.model_args.re_params.edit('learning_rate',trial.suggest_categorical('re_lr', lr_range))
            self.exp_cfgs.model_args.re_params.edit('per_gpu_train_batch_size',batch_size)
            
            self.exp_cfgs.model_args.ner_params.edit('learning_rate',trial.suggest_categorical('ner_lr', lr_range))
            self.exp_cfgs.model_args.ner_params.edit('per_gpu_train_batch_size',batch_size)
        
        if train_ner:
            # First Train NER model and save NER results
            ner_exportargs = {}
            for key,val in self.exp_cfgs.model_args.configs.items():
                if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                    ner_exportargs[key] = val

            for key,val in self.exp_cfgs.model_args.ner_params.configs.items():
                ner_exportargs[key] = val
            
            ner_model_path = os.path.join(model_path,'ner_model') if 'best_model' in model_path else model_path
            ner_exportargs[TRANSLATE_ARGS['model_path']] = ner_model_path
            ner_exportargs[TRANSLATE_ARGS['train_path']] = train_path if not ner_train_path else ner_train_path
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

        if train_ner:
            valid_path = os.path.join(self.exp_cfgs.model_args.log_path,'ent_pred_dev.json')
        else:
            valid_path = valid_path
        re_exportargs[TRANSLATE_ARGS['model_path']] = model_path
        re_exportargs[TRANSLATE_ARGS['train_path']] = train_path
        re_exportargs[TRANSLATE_ARGS['valid_path']] = valid_path
        re_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        re_exportargs['do_train'] = True
        re_exportargs['do_eval'] = True
        re_exportargs['do_test'] = False
        re_exportargs['do_predict'] = False



        eval_micro_f1 = call_pl_marker_re(re_exportargs, trial=trial)

        if curriculum_learning and not trial:
            return  valid_path
        else:
            return eval_micro_f1

    def eval(self, model_path, dataset_path, output_path, data_label='test', save_embeddings = False, Temp_rel = 1.0, Temp_ent = 1.0, ner_model_path = None):
           
        # First evaluate NER model and save NER results
        ner_exportargs = {}
        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                ner_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.ner_params.configs.items():
            ner_exportargs[key] = val

        if ner_model_path is None:
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