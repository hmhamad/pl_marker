import os

from models.wrapper import ModelWrapper
from mapping import map_datafile

import os
import json
from models.pl_marker.pl_marker_re_main import call_pl_marker_re
from models.pl_marker.pl_marker_ner_main import call_pl_marker_ner
from models.spert.args import train_argparser, eval_argparser, predict_argparser, ssl_argparser
from models.spert.config_reader import process_configs, _yield_configs
from models.spert.spert import input_reader, trainer
from models.spert.spert.spert_trainer import SpERTTrainer
from models.spert.spert_main import predict_for_ssl, train, eval
import shutil

TRANSLATE_ARGS = {'model_path': 'model_name_or_path', 'train_path': 'train_file', 'valid_path': 'dev_file', 'dataset_path': 'data_file', 'log_path': 'output_dir'}

class PLMarkerWrapper(ModelWrapper):
    def __init__(self, exp_cfgs) -> None:
        super().__init__(exp_cfgs)

    def train(self, model_path, train_path, valid_path, output_path):
        # First Train NER model and save NER results
        ner_exportargs = {}
        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                ner_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.ner_params.configs.items():
            ner_exportargs[key] = val
        
        ner_exportargs[TRANSLATE_ARGS['model_path']] = model_path
        ner_exportargs[TRANSLATE_ARGS['train_path']] = train_path
        ner_exportargs[TRANSLATE_ARGS['valid_path']] = valid_path
        ner_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        ner_exportargs['do_train'] = True
        ner_exportargs['do_eval'] = True
        ner_exportargs['no_test'] = True


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
        re_exportargs['no_test'] = True



        call_pl_marker_re(re_exportargs)

        return True

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
        ner_exportargs['do_train'] = False
        ner_exportargs['do_eval'] = False
        ner_exportargs['no_test'] = False



        call_pl_marker_ner(ner_exportargs)

        re_exportargs = {}

        for key,val in self.exp_cfgs.model_args.configs.items():
            if key not in TRANSLATE_ARGS and not isinstance(val,dict):
                re_exportargs[key] = val

        for key,val in self.exp_cfgs.model_args.re_params.configs.items():
            re_exportargs[key] = val

        dataset_path = os.path.join(self.exp_cfgs.model_args.log_path,'ent_pred_test.json')
        re_exportargs[TRANSLATE_ARGS['model_path']] = model_path
        re_exportargs[TRANSLATE_ARGS['dataset_path']] = dataset_path
        re_exportargs[TRANSLATE_ARGS['log_path']] = output_path
        re_exportargs['no_test'] = False
        re_exportargs['do_train'] = False
        re_exportargs['do_eval'] = False
        re_exportargs['data_label'] = data_label


        call_pl_marker_re(re_exportargs)

        # Change predicted data format if not already in Standard format
        if self.exp_cfgs.sim_args.dataset == 'scierc':
            map_datafile(os.path.join(self.exp_cfgs.model_args.log_path,data_label+'_predictions.json'),'scierc_josnl','standard',predicted=True)

    def predict(self, model_path, dataset_path, output_path, Temp_rel = 1.0, Temp_ent = 1.0):
        
        config_path = os.path.join(self.exp_cfgs.sim_args.log_dir,'spert_config.conf')
        with open(config_path,'w') as f:
            for key,val in self.exp_cfgs.model_args.configs.items():
                if key in PREDICT_ARGS_LIST:
                    f.write(str(key) + ' = ' + str(val) + '\n')
            f.write('model_path = ' + model_path + '\n')
            f.write('dataset_path = ' + dataset_path + '\n')
            f.write('types_path = ' + self.exp_cfgs.sim_args.dataset_paths.types_path + '\n')
            f.write('predictions_path = ' + os.path.join(self.exp_cfgs.sim_args.log_dir,'predictions.json')  + '\n')
            f.write('Temp_rel = ' + str(Temp_rel) + '\n')
            f.write('Temp_ent = ' + str(Temp_ent) + '\n')

        call_spert('predict',config_path)

    def calibrate(self, model_path, dataset_path, output_path, save_embeddings = False, embeddings_label='train'):
        
        config_path = os.path.join(self.exp_cfgs.sim_args.log_dir,'spert_config.conf')
        with open(config_path,'w') as f:
            for key,val in self.exp_cfgs.model_args.configs.items():
                if key in EVAL_ARGS_LIST:
                    f.write(str(key) + ' = ' + str(val) + '\n')
            f.write('model_path = ' + model_path + '\n')
            f.write('dataset_path = ' + dataset_path + '\n')
            f.write('log_path = ' + output_path + '\n')
            f.write('save_embeddings = ' + str(save_embeddings) + '\n')
            f.write('embeddings_label = ' + str(embeddings_label) + '\n')
            f.write('types_path = ' + self.exp_cfgs.sim_args.dataset_paths.types_path + '\n')

        Temp_ent, Temp_rel = call_spert('calibrate',config_path)

        return Temp_ent, Temp_rel

    def ssl(self, prediction_dataset_path, prediction_output_path, model_path=None):
        os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES']=str(self.model_configs.device_id)

        arg_parser = train_argparser()
        args, _ = arg_parser.parse_known_args()

        dataset_paths = self.main_configs.dataset_paths

        with open(os.path.join('models','spert','configs','temp.conf'),'w') as f:
            args_list = [arg for arg in vars(args)]
            for key,val in self.model_configs.configs.items():
                if key in args_list:
                    f.write(str(key) + ' = ' + str(val) + '\n')

        run_args, _, _ = next(_yield_configs(arg_parser, args, verbose=True))
        run_args.types_path = dataset_paths.types_path
        run_args.model_path = os.path.join(run_args.save_path, 'final_model')
        run_args.spacy_model = self.main_configs.spert_params.spacy_model
        
        
        predict_for_ssl(run_args, prediction_dataset_path, prediction_output_path, run_args.log_path, run_args.save_path, run_args.seed)

    # TODO: Delete later
    def semi_supervised_learning(run_args):
        ssl_iterations_start = 0
        ssl_train_path = run_args.train_path
        if not os.path.exists(f'{run_args.log_path}/train_status.json'):
            if os.path.exists(run_args.prediction_output_dir):
                shutil.rmtree(run_args.prediction_output_dir)

            os.makedirs(run_args.prediction_output_dir)
            trainer = SpERTTrainer(run_args, run_args.log_path, run_args.save_path, run_args.seed)
            trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path, types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)

            with open(f'{run_args.log_path}/train_status.json', 'w') as f:
                f.write(str(ssl_iterations_start))
        else:
            with open(f'{run_args.log_path}/train_status.json', 'r') as f:
                ssl_iterations_start = int(f.read())

            if ssl_iterations_start:
                ssl_train_path = f'{run_args.save_path}/ssl_train_path.json'

            print(f'Starting prediction from ssl prediction {ssl_iterations_start}')
            run_args.model_path = f'{run_args.save_path}/final_model'
            trainer = SpERTTrainer(run_args, run_args.log_path, run_args.save_path, run_args.seed)

        run_args.model_path = f'{run_args.save_path}/final_model'
        prediction_dataset_dir = run_args.prediction_dataset_dir
        prediction_output_dir = run_args.prediction_output_dir

        ssl_batch_size = run_args.ssl_batch_size
        with open(ssl_train_path, 'r') as f:
            gt_train = json.load(f)

        ssl_train_path = f'{run_args.save_path}/ssl_train_path.json'
        for i in range(ssl_iterations_start, run_args.ssl_iterations):
            print(f'Starting SSL itertation: {i}')
            print(f'Current GT size: {len(gt_train)}')
            for file_index in range(ssl_batch_size):
                file_to_process = f'{i * ssl_batch_size + file_index + 1}.json'

                prediction_dataset_path = os.path.join(prediction_dataset_dir, file_to_process)
                prediction_output_path = os.path.join(prediction_output_dir, file_to_process)

                print(f'Predicting file: {prediction_dataset_path}')
                trainer._args.predictions_path = prediction_output_path
                
                if not os.path.exists(prediction_output_path):
                    trainer.predict(dataset_path=prediction_dataset_path, types_path=run_args.types_path, input_reader_cls=input_reader.JsonPredictionInputReader)
                else:
                    print(f'Prediction found: {prediction_dataset_path}')

            semi_supervised_training_dataset = []
            print(f'Selecting {run_args.ssl_samples_per_relation} samples of each relation type')
            from old.summarize_json_output import JsonOutputSummarizer, RelationType
            summarizer = JsonOutputSummarizer(prediction_output_dir, i * ssl_batch_size + 1, i * ssl_batch_size + ssl_batch_size)
            for relation_type in RelationType:
                # Move query to configs
                summaries = summarizer.query(
                    take=run_args.ssl_samples_per_relation,
                    selection_criteria=run_args.ssl_selection_criteria,
                    sentence_agg=run_args.ssl_sentence_confidence_agg,
                    ssl_relation_confidences_threshold=run_args.ssl_relation_confidences_threshold,
                    ssl_entity_confidences_threshold=run_args.ssl_entity_confidences_threshold,
                    relation_types=[relation_type])

                print(f'Found {len(summaries)} of {relation_type.value}')
                semi_supervised_training_dataset.extend(summaries)

            gt_train.extend(semi_supervised_training_dataset)

            with open(ssl_train_path, 'w') as f:
                print(f'Current GT size: {len(gt_train)}')
                json.dump(gt_train, f)

            trainer = SpERTTrainer(run_args, run_args.log_path, run_args.save_path, run_args.seed)
            trainer.train(train_path=ssl_train_path, valid_path=run_args.valid_path, types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)
            with open(f'{run_args.log_path}/train_status.json', 'w') as f:
                f.write(str(i + 1))
