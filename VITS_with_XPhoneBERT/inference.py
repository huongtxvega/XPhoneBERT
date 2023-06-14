import torch
import utils
from models import SynthesizerTrn
from transformers import AutoTokenizer
import soundfile as sf
from text2phonemesequence import Text2PhonemeSequence
from underthesea import word_tokenize


def get_inputs(text, model, tokenizer_xphonebert):
    text_segmented = word_tokenize(text, format="text")
    phones = model.infer_sentence(text_segmented)
    tokenized_text = tokenizer_xphonebert(phones)
    input_ids = torch.LongTensor(tokenized_text['input_ids'])
    attention_mask = torch.LongTensor(tokenized_text['attention_mask'])
    return input_ids, attention_mask


def inference(text, output, config, model_ckp, language='vie-n', cuda=True):
    device = torch.device("cuda:0") if cuda else torch.device("cpu") 
    hps = utils.get_hparams_from_file(config)
    tokenizer_xphonebert = AutoTokenizer.from_pretrained(hps.bert)
    # Load Text2PhonemeSequence
    model = Text2PhonemeSequence(language=language, is_cuda=cuda)
    net_g = SynthesizerTrn(
        hps.bert,
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to(device)
    
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_ckp, net_g, None)

    input_ids, attention_mask = get_inputs(text, model, tokenizer_xphonebert)
    
    with torch.no_grad():
        input_id = input_ids.to(device).unsqueeze(0)
        attention_mask = attention_mask.to(device).unsqueeze(0)
        audio = net_g.infer(input_id, 
                            attention_mask, 
                            noise_scale=.667, 
                            noise_scale_w=0.8, 
                            length_scale=1)
        audio = audio[0][0,0].data.cpu().float().numpy()

    sf.write(output, audio, hps.data.sampling_rate)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', dest="text", type=str, required=True)
    parser.add_argument('-o', '--output', dest="output", type=str, help='Output wav file')
    parser.add_argument('-c', '--config', dest="config", type=str, help='JSON file for configuration')
    parser.add_argument('-m', '--model-ckp', dest="model_ckp", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('-l', '--language', dest="language", type=str, default="vie-n", help='Language for phoneme tokenize')
    parser.add_argument('--cuda', dest="cuda", type=bool, default=True)
    
    args = parser.parse_args()

    inference(**args.__dict__)
