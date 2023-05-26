from transformers import BertForMaskedLM, BertTokenizer


def load_bert_model_and_tokenizer(model_config, cache_dir='../workdir/cache'):
    print('BERT configurations: ', model_config)
    bpe_tokenizer = BertTokenizer.from_pretrained(model_config, do_lower_case=model_config.endswith('uncased'))
    model = BertForMaskedLM.from_pretrained(model_config, cache_dir=cache_dir)
    res = model.cuda()
    return model, bpe_tokenizer
