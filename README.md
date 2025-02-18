# mT5

## Description

This is a repository for the finetune of the mT5 and Llama3.1-8B model on the Opus dataset for the translation task. The model is trained on the English to Korean translation task, and then English to Chinese translation task. The model is evaluated on the sacrebleu score metric.

More on the project can be found [here](https://github.com/Neetre/translator.git)

Model hf name: `Neetree/KoLama`

## Dataset

The dataset used for the training is the Opus dataset. The dataset is a collection of translated texts from various sources and is available in multiple languages. The dataset is available at [Opus](http://opus.nlpl.eu/).

## Model

The model used for this task is the mT5 model. The mT5 model is a variant of the T5 model that is pretrained on multiple languages. The model is available in the Hugging Face model hub.

At the end I'm using the Llama3.1-8B model for the fine-tuning. Thx to Unsloth and the LoRA method, I'm able to lower the time of the training and increase the performance of the model.

Try it with the `try_it.py` script in the `bin` folder.

## Training

The model is trained on the English to Korean translation task and then on the English to Chinese translation task. The model is trained using the `ko_finetune_mt5.py` script in the `bin` folder. The training is done on a single GPU (for now a 4090).

For the Llama3.1-8B model, the file is `unsloth_training.py` in the `bin` folder. This is also done on a single GPU.

## Evaluation

The model is evaluated on the sacrebleu score metric.
The evaluation is done during the training process.

## References

- [Opus](http://opus.nlpl.eu/)
- [mT5](https://huggingface.co/transformers/model_doc/mt5.html)
- [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu)
- [T5](https://huggingface.co/transformers/model_doc/t5.html)
- [Hugging Face](https://huggingface.co/)
- [mT5 Paper](https://arxiv.org/abs/2010.11934)
- [mT5 DDP](https://cloud.tencent.com/developer/ask/sof/107218421)  # For the DDP training of the mT5 model
