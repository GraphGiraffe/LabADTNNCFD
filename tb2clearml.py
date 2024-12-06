from pathlib import Path
import json
from pprint import pprint
from tqdm import tqdm

import numpy as np
np.string_ = np.bytes_
np.unicode_ = np.str_

from tensorboard.backend.event_processing import event_accumulator
from clearml import Task, OutputModel

from multiprocessing import Pool


def parse_tensorboard_events(event_file):
    """Считывает данные из TensorBoard events файла."""
    ea = event_accumulator.EventAccumulator(event_file,
                                            size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    metrics = {}
    for tag in ea.Tags().get('scalars', []):
        metrics[tag] = []
        for scalar_event in ea.Scalars(tag):
            metrics[tag].append((scalar_event.step, scalar_event.value))
    return metrics


def log_to_clearml(task, metrics):
    """Передает собранные метрики в ClearML."""
    logger = task.get_logger()
    for tag, values in metrics.items():
        ans = tag.split('/')
        if len(ans) == 2:
            tag, series = ans
        else:
            tag = ans[0]
            series = tag
        for step, value in values:
            logger.report_scalar(title=tag, series=series, iteration=step, value=value)

def process(event_file):
    log_dir = event_file.parent
    with open(log_dir / 'params.json', 'r') as f:
        loaded_params = json.load(f)
    loaded_params['utils'] = dict(server='CASCADE')
    # Сбор метрик из TensorBoard файла
    if event_file.exists():

        metrics = parse_tensorboard_events(str(event_file))

        # Отправка метрик в ClearML
        task = Task.init(project_name="DeepCFD_FC",
                            task_name=str(log_dir),
                            output_uri=False,
                            auto_resource_monitoring=False,
                            )

        model_p_dump = loaded_params.get('model')
        task.connect(loaded_params)
        output_model = OutputModel(task=task)
        output_model.update_design(config_dict=model_p_dump)
        
        log_to_clearml(task, metrics)
        task.close()

        print(f"DONE: {event_file}")

    else:
        print(f"Файл {event_file} не найден.")

if __name__ == "__main__":
    # Укажите путь к TensorBoard events файлу
    event_mask = '**/events.out.tfevents.*'
    p = Path(r'out_CASCADE').glob(event_mask)
    fp_list = [x for x in p if x.is_file()]
    pprint(fp_list)

    # for event_file in tqdm(fp_list):
    #     process(event_file)
    
    with Pool(len(fp_list)) as p:
        list(tqdm(p.imap(process, fp_list), total=len(fp_list)))

